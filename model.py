import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.pool import global_mean_pool as gmp

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProteinSA(nn.Module): # is this too slow? is ESM-IF1 already contextualised? can always remove this if necessary
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.1):
        super(ProteinSA, self).__init__()
        self.multiheadattention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)

    def forward(self, x, mask=None):
        output, _ = self.multiheadattention(x, x, x, key_padding_mask=mask)
        return output
    
## FFN here?

# class DrugConv(nn.Module): # can afford to be cheap on drug side because UniMol is quite comprehensive
#     def __init__(self, input_dim, hidden_dims, dropout_rate=0.1):
#         super(DrugConv, self).__init__()
#         self.conv1 = nn.Conv1d(input_dim, hidden_dims[0], kernel_size=3, padding=1, stride=1)
#         self.conv2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1, stride=1)

#     def forward(self, x):
#         x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
#         return x

class DrugConv(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.1):
        super(DrugConv, self).__init__()
        self.conv1 = nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        return x

class CrossAttention(nn.Module): # refer to CAT-DTI
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.1):
        super(CrossAttention, self).__init__()
        self.CAp = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.CAd = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        ff_dim = embed_dim * 4
        self.ln_p1 = nn.LayerNorm(embed_dim)
        self.ln_p2 = nn.LayerNorm(embed_dim)
        self.ln_d1 = nn.LayerNorm(embed_dim)
        self.ln_d2 = nn.LayerNorm(embed_dim)
        self.ff_p = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.ff_d = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, protein_features, drug_features, protein_mask=None, drug_mask=None):
        attended_protein_features, attentionp = self.CAp(self.ln_p1(protein_features), drug_features, drug_features, key_padding_mask=drug_mask)
        protein_features = protein_features + self.dropout(attended_protein_features)
        protein_features = protein_features + self.dropout(self.ff_p(self.ln_p2(protein_features)))

        attended_drug_features, attentiond = self.CAd(self.ln_d1(drug_features), protein_features, protein_features, key_padding_mask=protein_mask)
        drug_features = drug_features + self.dropout(attended_drug_features)
        drug_features = drug_features + self.dropout(self.ff_d(self.ln_d2(drug_features)))

        return protein_features, drug_features
    
class CrossAttentionOld(nn.Module): # refer to CAT-DTI
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.1):
        super(CrossAttentionOld, self).__init__()
        self.CAp = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.CAd = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)

    def forward(self, protein_features, drug_features, protein_mask=None, drug_mask=None):
        attended_protein_features, attentionp = self.CAp(protein_features, drug_features, drug_features, key_padding_mask=drug_mask)
        attended_drug_features, attentiond = self.CAd(drug_features, protein_features, protein_features, key_padding_mask=protein_mask)
        return protein_features, drug_features

class Fusion(nn.Module): # get fixed length representations and concat
    def __init__(self, drug_embed_dim, drug_hidden_dims, protein_embed_dim, protein_hidden_dims, dropout_rate=0.2):
        super(Fusion, self).__init__()
        # 2 for drug but 1 for protein
        self.drug_linear = nn.Sequential(
            nn.Linear(drug_embed_dim, drug_hidden_dims[1]),
            nn.ReLU(),
        )
        self.protein_linear = nn.Sequential(
            nn.Linear(protein_embed_dim, protein_hidden_dims[0]),
            nn.ReLU(),
        )

    def forward(self, protein_features, drug_features, protein_mask, drug_mask): # TODO: Think about pooling strategies
        # # pooling strategy from evidti
        # protein_features = torch.mean(protein_features, dim=1)  # mean pooling
        # protein_features = self.protein_linear(protein_features) 

        # # mean pooling for drugs as recommended by unimol
        # drug_features = torch.mean(drug_features, dim=1)  # mean pooling
        # drug_features = self.drug_fc1(drug_features)
        # drug_features = self.drug_fc2(drug_features)

        if protein_mask is not None:
            valid_p = ~protein_mask                      # True where valid
            valid_p = valid_p.unsqueeze(-1)              # (B, Lp, 1)
            protein_sum = (protein_features * valid_p).sum(dim=1)  # (B, D)
            protein_count = valid_p.sum(dim=1).clamp(min=1)        # (B, 1)
            protein_features = protein_sum / protein_count
        else:
            protein_features = protein_features.mean(dim=1)

        protein_features = self.protein_linear(protein_features)

        if drug_mask is not None:
            valid_d = ~drug_mask                         # True where valid
            valid_d = valid_d.unsqueeze(-1)              # (B, Ld, 1)
            drug_sum = (drug_features * valid_d).sum(dim=1)        # (B, D)
            drug_count = valid_d.sum(dim=1).clamp(min=1)           # (B, 1)
            drug_features = drug_sum / drug_count
        else:
            drug_features = drug_features.mean(dim=1)

        drug_features = self.drug_linear(drug_features)
        
        # now both drug and protein features are of same dimension of 256
        res = torch.cat((protein_features, drug_features), dim=-1)
        return res


# class Fusion(nn.Module): # get fixed length representations and concat
#     def __init__(self, drug_embed_dim, drug_hidden_dims, protein_embed_dim, protein_hidden_dims, dropout_rate=0.2):
#         super(Fusion, self).__init__()
#         # 2 for drug but 1 for protein
#         self.drug_fc1 = nn.Linear(drug_embed_dim, drug_hidden_dims[0])
#         self.drug_fc2 = nn.Linear(drug_hidden_dims[0], drug_hidden_dims[1])

#         # self.protein_linear = nn.Sequential(
#         #     nn.Linear(protein_embed_dim, protein_hidden_dims[0]),
#         #     nn.Dropout(dropout_rate),
#         #     nn.ReLU(),
#         #     nn.BatchNorm1d(protein_hidden_dims[0]),
#         # )

#         self.protein_linear = nn.Linear(protein_embed_dim, protein_hidden_dims[0])

#     def forward(self, protein_features, drug_features, protein_mask, drug_mask): # TODO: Think about pooling strategies
#         # # pooling strategy from evidti
#         # protein_features = torch.mean(protein_features, dim=1)  # mean pooling
#         # protein_features = self.protein_linear(protein_features) 

#         # # mean pooling for drugs as recommended by unimol
#         # drug_features = torch.mean(drug_features, dim=1)  # mean pooling
#         # drug_features = self.drug_fc1(drug_features)
#         # drug_features = self.drug_fc2(drug_features)

#         if protein_mask is not None:
#             valid_p = ~protein_mask                      # True where valid
#             valid_p = valid_p.unsqueeze(-1)              # (B, Lp, 1)
#             protein_sum = (protein_features * valid_p).sum(dim=1)  # (B, D)
#             protein_count = valid_p.sum(dim=1).clamp(min=1)        # (B, 1)
#             protein_features = protein_sum / protein_count
#         else:
#             protein_features = protein_features.mean(dim=1)

#         protein_features = self.protein_linear(protein_features)

#         if drug_mask is not None:
#             valid_d = ~drug_mask                         # True where valid
#             valid_d = valid_d.unsqueeze(-1)              # (B, Ld, 1)
#             drug_sum = (drug_features * valid_d).sum(dim=1)        # (B, D)
#             drug_count = valid_d.sum(dim=1).clamp(min=1)           # (B, 1)
#             drug_features = drug_sum / drug_count
#         else:
#             drug_features = drug_features.mean(dim=1)

#         drug_features = self.drug_fc1(drug_features)
#         drug_features = self.drug_fc2(drug_features)
        
#         # now both drug and protein features are of same dimension of 256
#         res = torch.cat((protein_features, drug_features), dim=-1)
#         return res

class MLP1(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.out = nn.Linear(hidden_dims[2], hidden_dims[3])
        # self.dropout = nn.Dropout(dropout_rate)
        self.dropout = nn.AlphaDropout(dropout_rate)

    def forward(self, x):
        x = nn.SELU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.SELU()(self.fc2(x))
        x = self.dropout(x)
        x = nn.SELU()(self.fc3(x))
        x = self.dropout(x)
        x = self.out(x)
        # x = F.softplus(x) + 1 # !!

        return x

class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.out = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.protein_sa = ProteinSA(cfg.PROTEIN.EMBEDDING_DIM)
        self.drug_conv = DrugConv(cfg.DRUG.EMBEDDING_DIM, cfg.DRUG.CONV_DIMS)
        self.cross_attention = CrossAttentionOld(cfg.PROTEIN.EMBEDDING_DIM, dropout_rate=cfg.SOLVER.DROPOUT)
        self.fusion = Fusion(cfg.DRUG.EMBEDDING_DIM, cfg.DRUG.MLP_DIMS, cfg.PROTEIN.EMBEDDING_DIM, cfg.PROTEIN.DIMS, cfg.SOLVER.DROPOUT)
        self.mlp = MLP2(cfg.MLP.INPUT_DIM, cfg.MLP.DIMS, cfg.SOLVER.DROPOUT)
    def forward(self, protein_emb, drug_emb, protein_mask=None, drug_mask=None, mode="train"):
        # i should be able to easily turn off SA and the drug CNN
        # input is (B, L, D)
        # protein_features = self.protein_sa(protein_emb, mask=protein_mask)
        protein_features = protein_emb  # (B, L, D)

        # drug_features = self.drug_conv(drug_emb)
        drug_features = drug_emb  # (B, L, D)
        # Both (B, L, D)
        attended_protein_features, attended_drug_features = self.cross_attention(protein_features, drug_features, protein_mask=protein_mask, drug_mask=drug_mask)
        # attended_protein_features = protein_features
        # attended_drug_features = drug_features
        fused_features = self.fusion(attended_protein_features, attended_drug_features, protein_mask=protein_mask, drug_mask=drug_mask)
        # at this point, shape of (B, D)
        output = self.mlp(fused_features)

        return output




def _test_masks():
    """
    Quick sanity check for mask semantics:
    - Builds toy protein/drug sequences with different lengths.
    - Constructs masks in the same way as the collate_fn (True = PAD, False = valid).
    - Runs ProteinSA and CrossAttention and checks basic invariants.
    """
    torch.manual_seed(0)
    device_local = device

    # Toy batch
    B = 2
    Lp_max = 6
    Ld_max = 5
    D = 16  # small dim for a quick test

    # Example lengths: second sample is full length, first has padding
    prot_lens = torch.tensor([4, 6], device=device_local)
    drug_lens = torch.tensor([3, 5], device=device_local)

    # Build masks in the same way as collate_fn (True = padding, False = valid)
    prot_mask = torch.arange(Lp_max, device=device_local).unsqueeze(0).expand(B, Lp_max) >= prot_lens.unsqueeze(1)
    drug_mask = torch.arange(Ld_max, device=device_local).unsqueeze(0).expand(B, Ld_max) >= drug_lens.unsqueeze(1)

    print("[_test_masks] protein lengths:", prot_lens.tolist())
    print("[_test_masks] protein mask (True = PAD):")
    print(prot_mask.cpu().numpy())
    print("[_test_masks] drug lengths:", drug_lens.tolist())
    print("[_test_masks] drug mask (True = PAD):")
    print(drug_mask.cpu().numpy())
    print(~drug_mask)

    # Random token embeddings
    protein_emb = torch.randn(B, Lp_max, D, device=device_local, requires_grad=True)
    drug_emb = torch.randn(B, Ld_max, D, device=device_local, requires_grad=True)

    # Instantiate small versions of the modules
    protein_sa = ProteinSA(embed_dim=D, num_heads=4, dropout_rate=0.0).to(device_local)
    cross_attn = CrossAttention(embed_dim=D, num_heads=4, dropout_rate=0.0).to(device_local)
    fusion = Fusion(
        drug_embed_dim=D,
        drug_hidden_dims=[32, 8],
        protein_embed_dim=D,
        protein_hidden_dims=[8],
        dropout_rate=0.0,
    ).to(device_local)

    protein_sa.train()
    cross_attn.train()
    fusion.train()

    # Forward through ProteinSA with masks
    sa_out = protein_sa(protein_emb, mask=prot_mask)
    print("[_test_masks] ProteinSA output shape:", tuple(sa_out.shape))

    # Forward through CrossAttention with masks
    attn_prot, attn_drug = cross_attn(sa_out, drug_emb, protein_mask=prot_mask, drug_mask=drug_mask)
    print("[_test_masks] CrossAttention protein/drug shapes:", tuple(attn_prot.shape), tuple(attn_drug.shape))

    # Forward through Fusion with masks
    fused = fusion(attn_prot, attn_drug, protein_mask=prot_mask, drug_mask=drug_mask)
    print("[_test_masks] Fusion output shape:", tuple(fused.shape))

    # Simple grad check to ensure masks don't block all gradients
    loss = fused.pow(2).mean()
    loss.backward()
    has_grad = any(p.grad is not None for p in list(protein_sa.parameters()) +
                   list(cross_attn.parameters()) +
                   list(fusion.parameters()))
    assert has_grad, "[_test_masks] No gradients flowed with masks applied."
    print("[_test_masks] Gradients OK with masks.")

# -------------------------
# Sanity checks
# -------------------------

def _test_fusion_module():
    torch.manual_seed(42)
    device_local = device

    B = 2
    Lp = 300   # protein residues
    Ld = 60    # drug atoms/tokens
    D  = 512

    # Random token features (simulate post-CA tokens)
    protein_tokens = torch.randn(B, Lp, D, device=device_local, requires_grad=True)
    drug_tokens    = torch.randn(B, Ld, D, device=device_local, requires_grad=True)

    # Instantiate Fusion with your current config choices
    fusion = Fusion(
        drug_embed_dim=D,
        drug_hidden_dims=[1024, 256],
        protein_embed_dim=D,
        protein_hidden_dims=[256],
        dropout_rate=0.1
    ).to(device_local)
    fusion.train()

    # Forward
    fused = fusion(protein_tokens, drug_tokens, protein_mask=None, drug_mask=None)  # expect [B, 512]
    print("[Fusion] output shape:", tuple(fused.shape))

    # Shape check
    assert fused.shape == (B, 512), f"Expected (B,512) got {tuple(fused.shape)}"

    # Simple gradient check
    loss = fused.pow(2).mean()
    loss.backward()

    # Ensure gradients exist on a representative parameter
    has_grad = any(p.requires_grad and p.grad is not None for p in fusion.parameters())
    assert has_grad, "No gradients flowed through Fusion parameters."
    print("[Fusion] gradients ok.")

def _smoke_test():
    from config.cfg import get_cfg_defaults
    cfg = get_cfg_defaults()

    model = Model(cfg).to(device)
    model.eval()

    B = 32
    Lp = 256
    Ld = 48
    Dp = cfg.PROTEIN.EMBEDDING_DIM
    Dd = cfg.DRUG.EMBEDDING_DIM

    # Random embeddings (simulate ESM-IF1 and UniMol outputs)
    protein_emb = torch.randn(B, Lp, Dp, device=device)
    drug_emb = torch.randn(B, Ld, Dd, device=device)

    with torch.no_grad():
        out = model(protein_emb, drug_emb)
        print(out)
    print("[Model smoke test] Forward pass successful. Output shape:", tuple(out.shape))

def _test_model_forward():
    import torch
    from torch.utils.data import DataLoader
    from dataset import MyDataset  # change this import path
    from dataset import collate_fn
    from config.cfg import get_cfg_defaults
    cfg = get_cfg_defaults()

    model = Model(cfg).to(device)
    model.eval()

    ds = MyDataset(cfg.DATA.TRAIN_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
    dl = DataLoader(ds, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=True)

    B = cfg.SOLVER.BATCH_SIZE

    s = next(iter(dl))

    with torch.no_grad():
        out = model(s["protein_emb"].to(device), s["drug_emb"].to(device), s["protein_mask"].to(device), s["drug_mask"].to(device))
        print(out)
    print("[Model forward test] Forward pass successful. Output shape:", tuple(out.shape))

if __name__ == "__main__":
    from config.cfg import get_cfg_defaults
    model = Model(get_cfg_defaults()).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    _test_masks()
    # _test_fusion_module()
    # _smoke_test()
    # _test_model_forward()
