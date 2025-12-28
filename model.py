import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.pool import global_mean_pool as gmp

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProteinSA(nn.Module): # is this too slow? is ESM-IF1 already contextualised? can always remove this if necessary
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0):
        super(ProteinSA, self).__init__()
        self.multiheadattention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)

    def forward(self, x, mask=None):
        output, _ = self.multiheadattention(x, x, x, key_padding_mask=mask)
        return output
    
class ProteinSAtransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0, num_layers=2):
        super(ProteinSAtransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        if mask is not None:
            # Transformer expects mask with shape (batch_size, seq_len)
            # where True indicates positions to be masked
            transformer_mask = mask
        else:
            transformer_mask = None
        output = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)
        return output

class ProteinSAnew(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        
        ff_dim = embed_dim * 4
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        attn_in = self.ln1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, key_padding_mask=mask)
        x = x + self.dropout(attn_out)

        ff_in = self.ln2(x)
        ff_out = self.ff(ff_in)
        x = x + self.dropout(ff_out)

        return x

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

class DrugCNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, num_layers=1, kernel_size=3, dropout_rate=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim  # keep same dim for simplicity

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

        in_c = embed_dim
        for i in range(num_layers):
            out_c = hidden_dim
            conv = nn.Conv1d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # keep length
                stride=1,
            )
            self.layers.append(conv)
            in_c = out_c

        # if hidden_dim == embed_dim we can do residuals
        self.use_residual = (hidden_dim == embed_dim)

    def forward(self, x, mask=None):
        """
        x:    (B, L, D) drug embeddings from UniMol
        mask: (B, L) bool, True = PAD (optional, we just pass it through)
        """
        B, L, D = x.shape
        h = x.transpose(1, 2)  # (B, D, L) for Conv1d

        for conv in self.layers:
            h_in = h
            h = conv(h)
            h = self.activation(h)
            h = self.dropout(h)
            if self.use_residual and h.shape == h_in.shape:
                h = h + h_in   # simple residual

        h = h.transpose(1, 2)  # back to (B, L, D or hidden_dim)

        # we don't need to touch the mask here; downstream modules still use it
        return h

class DrugSA(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0):
        super(DrugSA, self).__init__()
        self.multiheadattention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)

    def forward(self, x, mask=None):
        output, _ = self.multiheadattention(x, x, x, key_padding_mask=mask)
        return output

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

    def forward(self, protein_features, drug_features, protein_mask, drug_mask, return_attention=False):
        attended_protein_features, attentionp = self.CAp(self.ln_p1(protein_features), drug_features, drug_features, key_padding_mask=drug_mask)
        protein_features = protein_features + self.dropout(attended_protein_features)
        protein_features = protein_features + self.dropout(self.ff_p(self.ln_p2(protein_features)))

        protein_features = protein_features.masked_fill(
            protein_mask.unsqueeze(-1), 0.0
        )

        attended_drug_features, attentiond = self.CAd(self.ln_d1(drug_features), protein_features, protein_features, key_padding_mask=protein_mask)
        drug_features = drug_features + self.dropout(attended_drug_features)
        drug_features = drug_features + self.dropout(self.ff_d(self.ln_d2(drug_features)))

        drug_features = drug_features.masked_fill(drug_mask.unsqueeze(-1), 0.0)

        if return_attention:
            return protein_features, drug_features, attentionp, attentiond

        return protein_features, drug_features

class CrossAttentionAdaptive(nn.Module): # LayerScale
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.1):
        super(CrossAttentionAdaptive, self).__init__()
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
        self.alpha_p_attn = nn.Parameter(torch.ones(1) * 1e-4)
        self.alpha_p_ff   = nn.Parameter(torch.ones(1) * 1e-4)
        self.alpha_d_attn = nn.Parameter(torch.ones(1) * 1e-4)
        self.alpha_d_ff   = nn.Parameter(torch.ones(1) * 1e-4)

    def forward(self, protein_features, drug_features, protein_mask=None, drug_mask=None):
        attended_protein_features, attentionp = self.CAp(self.ln_p1(protein_features), drug_features, drug_features, key_padding_mask=drug_mask)
        protein_features = protein_features + self.alpha_p_attn * self.dropout(attended_protein_features)
        protein_features = protein_features + self.alpha_p_ff * self.dropout(self.ff_p(self.ln_p2(protein_features)))

        attended_drug_features, attentiond = self.CAd(self.ln_d1(drug_features), protein_features, protein_features, key_padding_mask=protein_mask)
        drug_features = drug_features + self.alpha_d_attn * self.dropout(attended_drug_features)
        drug_features = drug_features + self.alpha_d_ff * self.dropout(self.ff_d(self.ln_d2(drug_features)))

        return protein_features, drug_features

class Fusion(nn.Module): # get fixed length representations and concat
    def __init__(self, drug_embed_dim, drug_hidden_dims, protein_embed_dim, protein_hidden_dims, dropout_rate=0.2):
        super(Fusion, self).__init__()
        # 2 for drug but 1 for protein
        self.drug_linear1 = nn.Sequential(
            nn.Linear(drug_embed_dim, drug_hidden_dims[0]),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )
        self.drug_linear2 = nn.Sequential(
            nn.Linear(drug_hidden_dims[0], drug_hidden_dims[1]),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )

        self.protein_linear = nn.Sequential(
            nn.Linear(protein_embed_dim, protein_hidden_dims[0]),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )

    def forward(self, protein_features, drug_features, protein_mask, drug_mask): # TODO: Think about pooling strategies

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

        # if protein_mask is not None:
        #     # mask padded positions with -inf so they don't affect max
        #     prot_mask_exp = protein_mask.unsqueeze(-1)  # (B, Lp, 1)
        #     prot_masked = protein_features.masked_fill(prot_mask_exp, float("-inf"))
        #     protein_features, _ = prot_masked.max(dim=1)  # (B, D)
        # else:
        #     protein_features, _ = protein_features.max(dim=1)  # (B, D)

        # protein_features = self.protein_linear(protein_features)  # (B, protein_hidden_dims[0])

        # # ----- Drug: masked max pooling over sequence -----
        # if drug_mask is not None:
        #     drug_mask_exp = drug_mask.unsqueeze(-1)  # (B, Ld, 1)
        #     drug_masked = drug_features.masked_fill(drug_mask_exp, float("-inf"))
        #     drug_features, _ = drug_masked.max(dim=1)  # (B, D)
        # else:
        #     drug_features, _ = drug_features.max(dim=1)  # (B, D)

        drug_features = self.drug_linear1(drug_features)
        drug_features = self.drug_linear2(drug_features)
        
        # now both drug and protein features are of same dimension of 256
        res = torch.cat((protein_features, drug_features), dim=-1)
        return res

class LearnedMixedPool(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # Per-dimension mixing coefficients; sigmoid → (0,1)
        self.alpha = nn.Parameter(torch.zeros(1, embed_dim))  # init → 0.5 mean / 0.5 max

    def forward(self, x, mask=None):
        """
        x:    (B, L, D)
        mask: (B, L) bool, True = PAD, False = valid  OR  None
        returns: (B, D) pooled representation
        """
        B, L, D = x.shape

        if mask is not None:
            valid = ~mask                      # True where valid
            valid_exp = valid.unsqueeze(-1)    # (B, L, 1)

            # ---- mean pooling over valid tokens ----
            x_for_mean = x * valid_exp         # zero out pads
            count = valid_exp.sum(dim=1).clamp(min=1)   # (B, 1)
            mean = x_for_mean.sum(dim=1) / count        # (B, D)

            # ---- max pooling over valid tokens ----
            pad_exp = mask.unsqueeze(-1)       # (B, L, 1), True where PAD
            x_for_max = x.masked_fill(pad_exp, float("-inf"))
            max_vals, _ = x_for_max.max(dim=1)          # (B, D)

            # if a row is all PAD (shouldn't really happen), fix -inf
            all_pad = ~valid.any(dim=1)        # (B,)
            if all_pad.any():
                max_vals[all_pad] = 0.0
        else:
            mean = x.mean(dim=1)               # (B, D)
            max_vals, _ = x.max(dim=1)         # (B, D)

        # ---- learned mixing ----
        # alpha in (0,1), shape (1, D) → broadcast over batch
        alpha = torch.sigmoid(self.alpha)      # (1, D)
        pooled = alpha * max_vals + (1.0 - alpha) * mean  # (B, D)

        return pooled

class FusionNew(nn.Module): # 2 for both drug and protein
    def __init__(self, drug_embed_dim, drug_hidden_dims, protein_embed_dim, protein_hidden_dims, dropout_rate=0.2):
        super(FusionNew, self).__init__()
        # 2 for drug but 1 for protein
        self.drug_linear1 = nn.Sequential(
            nn.Linear(drug_embed_dim, drug_hidden_dims[0]),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )
        self.drug_linear2 = nn.Sequential(
            nn.Linear(drug_embed_dim, drug_hidden_dims[1]),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )

        self.protein_linear1 = nn.Sequential(
            nn.Linear(protein_embed_dim, protein_hidden_dims[0]),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )
        self.protein_linear2 = nn.Sequential(
            nn.Linear(protein_embed_dim, protein_hidden_dims[1]),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )

    def forward(self, protein_features, drug_features, protein_mask, drug_mask): # TODO: Think about pooling strategies

        if protein_mask is not None:
            valid_p = ~protein_mask                      # True where valid
            valid_p = valid_p.unsqueeze(-1)              # (B, Lp, 1)
            protein_sum = (protein_features * valid_p).sum(dim=1)  # (B, D)
            protein_count = valid_p.sum(dim=1).clamp(min=1)        # (B, 1)
            protein_features = protein_sum / protein_count
        else:
            protein_features = protein_features.mean(dim=1)

        # protein_features = self.protein_linear1(protein_features)
        protein_features = self.protein_linear2(protein_features)

        if drug_mask is not None:
            valid_d = ~drug_mask                         # True where valid
            valid_d = valid_d.unsqueeze(-1)              # (B, Ld, 1)
            drug_sum = (drug_features * valid_d).sum(dim=1)        # (B, D)
            drug_count = valid_d.sum(dim=1).clamp(min=1)           # (B, 1)
            drug_features = drug_sum / drug_count
        else:
            drug_features = drug_features.mean(dim=1)

        # drug_features = self.drug_linear1(drug_features)
        drug_features = self.drug_linear2(drug_features)
        
        # now both drug and protein features are of same dimension of 256
        res = torch.cat((protein_features, drug_features), dim=-1)
        return res


class AdaptiveFusion(nn.Module):
    def __init__(self, drug_embed_dim, drug_hidden_dims, protein_embed_dim, protein_hidden_dims, dropout_rate=0.2):
        super(AdaptiveFusion, self).__init__()
        # 2 for drug but 1 for protein
        self.drug_pooling = LearnedMixedPool(drug_embed_dim)
        self.protein_pooling = LearnedMixedPool(protein_embed_dim)
        self.drug_linear = nn.Sequential(
            nn.Linear(drug_embed_dim, drug_hidden_dims[1]),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )
        self.protein_linear = nn.Sequential(
            nn.Linear(protein_embed_dim, protein_hidden_dims[1]),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )

    def forward(self, protein_features, drug_features, protein_mask, drug_mask):
        protein_features = self.protein_pooling(protein_features, mask=protein_mask)
        protein_features = self.protein_linear(protein_features)
        drug_features = self.drug_pooling(drug_features, mask=drug_mask)
        drug_features = self.drug_linear(drug_features)
        res = torch.cat((protein_features, drug_features), dim=-1)
        return res
        

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
        x = F.softplus(x) + 1 # !!

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
        # x = F.softplus(x) + 1 # !!
        return x

class MLP3(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.fc5 = nn.Linear(hidden_dims[3], hidden_dims[4])
        self.out = nn.Linear(hidden_dims[4], hidden_dims[5])
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = self.activation(self.fc4(x))
        x = self.dropout(x)
        x = self.activation(self.fc5(x))
        x = self.out(x)
        # x = F.softplus(x) + 1 # !!
        return x

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # self.protein_sa = ProteinSAnew(cfg.PROTEIN.EMBEDDING_DIM, num_heads=4, dropout_rate=cfg.SOLVER.DROPOUT)
        self.protein_sa = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.PROTEIN.EMBEDDING_DIM,
                nhead=4,
                dropout=cfg.SOLVER.DROPOUT,
                batch_first=True,
                # activation="gelu",
            ), num_layers=1)
        # self.drug_sa = DrugSA(cfg.DRUG.EMBEDDING_DIM)
        # self.drug_conv = DrugConv(cfg.DRUG.EMBEDDING_DIM, cfg.DRUG.CONV_DIMS)
        # self.drug_cnn = DrugCNN(cfg.DRUG.EMBEDDING_DIM, hidden_dim=cfg.DRUG.EMBEDDING_DIM, num_layers=2, dropout_rate=cfg.SOLVER.DROPOUT)
        self.drug_sa = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.DRUG.EMBEDDING_DIM,
                nhead=4,
                dropout=cfg.SOLVER.DROPOUT,
                batch_first=True,
                # activation="gelu",
            ), num_layers=4)
        self.cross_attention = CrossAttention(cfg.PROTEIN.EMBEDDING_DIM, dropout_rate=cfg.SOLVER.DROPOUT, num_heads=4)

        self.fusion = FusionNew(cfg.DRUG.EMBEDDING_DIM, cfg.DRUG.MLP_DIMS, cfg.PROTEIN.EMBEDDING_DIM, cfg.PROTEIN.MLP_DIMS, cfg.SOLVER.DROPOUT)
        self.mlp = MLP2(cfg.MLP.INPUT_DIM, cfg.MLP.DIMS, cfg.SOLVER.DROPOUT)
    def forward(self, protein_emb, drug_emb, protein_mask=None, drug_mask=None, mode="train", return_attention=False):
        # i should be able to easily turn off SA and the drug CNN
        # input is (B, L, D)
        protein_features = self.protein_sa(protein_emb, src_key_padding_mask=protein_mask)
        # protein_features = protein_emb  # (B, L, D)

        # drug_features = self.drug_cnn(drug_emb)
        # drug_features = self.drug_sa(drug_emb, mask=drug_mask)
        drug_features = self.drug_sa(drug_emb, src_key_padding_mask=drug_mask)  # (B, L, D)
        # Both (B, L, D)
        if return_attention:
            attended_protein_features, attended_drug_features, attentionp, attentiond = self.cross_attention(protein_features, drug_features, protein_mask=protein_mask, drug_mask=drug_mask, return_attention=True)
        else:
            attended_protein_features, attended_drug_features = self.cross_attention(protein_features, drug_features, protein_mask=protein_mask, drug_mask=drug_mask, return_attention=False)
        fused_features = self.fusion(attended_protein_features, attended_drug_features, protein_mask=protein_mask, drug_mask=drug_mask)
        # at this point, shape of (B, D)
        output = self.mlp(fused_features)

        if return_attention:
            return output, attentionp, attentiond
        else:
            return output
    
class stdModel(nn.Module):
    def __init__(self, cfg):
        super(stdModel, self).__init__()
        self.protein_sa = ProteinSA(cfg.PROTEIN.EMBEDDING_DIM)
        # self.drug_sa = DrugSA(cfg.DRUG.EMBEDDING_DIM)
        # self.drug_conv = DrugConv(cfg.DRUG.EMBEDDING_DIM, cfg.DRUG.CONV_DIMS)
        self.cross_attention = CrossAttention(cfg.PROTEIN.EMBEDDING_DIM, dropout_rate=cfg.SOLVER.DROPOUT)
        self.fusion = Fusion(cfg.DRUG.EMBEDDING_DIM, cfg.DRUG.MLP_DIMS, cfg.PROTEIN.EMBEDDING_DIM, cfg.PROTEIN.MLP_DIMS, cfg.SOLVER.DROPOUT)
        self.mlp = MLP2(cfg.MLP.INPUT_DIM, cfg.MLP.DIMS, cfg.SOLVER.DROPOUT)
    def forward(self, input):
        protein_emb = input["protein_emb"]
        drug_emb = input["drug_emb"]
        protein_mask = input["protein_mask"]
        drug_mask = input["drug_mask"]
        # i should be able to easily turn off SA and the drug CNN
        # input is (B, L, D)
        protein_features = self.protein_sa(protein_emb, mask=protein_mask)
        # protein_features = protein_emb  # (B, L, D)

        # drug_features = self.drug_conv(drug_emb)
        # drug_features = self.drug_sa(drug_emb, mask=drug_mask)
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
    protein_sa = ProteinSA(embed_dim=D, num_heads=4, dropout_rate=0).to(device_local)
    cross_attn = CrossAttention(embed_dim=D, num_heads=4, dropout_rate=0).to(device_local)
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
