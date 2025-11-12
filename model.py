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

class DrugConv(nn.Module): # can afford to be cheap on drug side because UniMol is quite comprehensive
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.1):
        super(DrugConv, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dims[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        return x

class CrossAttention(nn.Module): # refer to CAT-DTI
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.1):
        super(CrossAttention, self).__init__()
        self.CA = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)

    def forward(self, protein_features, drug_features, protein_mask=None, drug_mask=None):
        attended_protein_features, attentionp = self.CA(protein_features, drug_features, drug_features, key_padding_mask=drug_mask)
        attended_drug_features, attentiond = self.CA(drug_features, protein_features, protein_features, key_padding_mask=protein_mask)
        return attended_protein_features, attended_drug_features

class Fusion(nn.Module): # get fixed length representations and concat
    def __init__(self, drug_embed_dim, drug_hidden_dims, protein_embed_dim, protein_hidden_dims, dropout_rate=0.2):
        super(Fusion, self).__init__()
        # 2 for drug but 1 for protein
        self.drug_fc1 = nn.Linear(drug_embed_dim, drug_hidden_dims[0])
        self.drug_fc2 = nn.Linear(drug_hidden_dims[0], drug_hidden_dims[1])

        self.protein_linear = nn.Sequential(
            nn.Linear(protein_embed_dim, protein_hidden_dims[0]),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.BatchNorm1d(protein_hidden_dims[0]),
        )

    def forward(self, protein_features, drug_features): # TODO: Think about pooling strategies
        # pooling strategy from evidti
        protein_features = torch.mean(protein_features, dim=1)  # mean pooling
        protein_features = self.protein_linear(protein_features) 

        # mean pooling for drugs as recommended by unimol
        drug_features = torch.mean(drug_features, dim=1)  # mean pooling
        drug_features = self.drug_fc1(drug_features)
        drug_features = self.drug_fc2(drug_features)

        # now both drug and protein features are of same dimension of 256

        res = torch.cat((protein_features, drug_features), dim=-1)
        return res

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.out = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = nn.SELU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.SELU()(self.fc2(x))
        x = self.dropout(x)
        x = nn.SELU()(self.fc3(x))
        x = self.dropout(x)
        x = self.out(x)
        x = F.softplus(x) + 1 # !!

        return x

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.protein_sa = ProteinSA(cfg.PROTEIN.EMBEDDING_DIM)
        self.drug_conv = DrugConv(cfg.DRUG.EMBEDDING_DIM, cfg.DRUG.CONV_DIMS)
        self.cross_attention = CrossAttention(cfg.PROTEIN.EMBEDDING_DIM)
        self.fusion = Fusion(cfg.DRUG.EMBEDDING_DIM, cfg.DRUG.MLP_DIMS, cfg.PROTEIN.EMBEDDING_DIM, cfg.PROTEIN.DIMS)
        self.mlp = MLP(cfg.MLP.INPUT_DIM, cfg.MLP.DIMS)
    def forward(self, protein_emb, drug_emb, mode="train"):
        # i should be able to easily turn off SA and the drug CNN
        # input is (B, L, D)
        protein_features = self.protein_sa(protein_emb)

        # drug_features = self.drug_conv(drug_emb)
        drug_features = drug_emb  # (B, L, D)
        # Both (B, L, D)
        attended_protein_features, attended_drug_features = self.cross_attention(protein_features, drug_features)
        fused_features = self.fusion(attended_protein_features, attended_drug_features)
        # at this point, shape of (B, D)
        output = self.mlp(fused_features)

        return output




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
    fused = fusion(protein_tokens, drug_tokens)  # expect [B, 512]
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
    from main import MyDataset  # change this import path
    from main import collate_fn
    from config.cfg import get_cfg_defaults
    cfg = get_cfg_defaults()

    model = Model(cfg).to(device)
    model.eval()

    ds = MyDataset(cfg.DATA.TRAIN_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
    dl = DataLoader(ds, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=True)

    B = cfg.SOLVER.BATCH_SIZE

    s = next(iter(dl))

    with torch.no_grad():
        out = model(s["protein_emb"].to(device), s["drug_emb"].to(device))
        print(out)
    print("[Model forward test] Forward pass successful. Output shape:", tuple(out.shape))

if __name__ == "__main__":
    _test_fusion_module()
    _smoke_test()
    _test_model_forward()
