import torch
import numpy as np

# this is data, i don't want to load as weights
reps = torch.load("drug/embeddings/DB00006_unimol.pt", weights_only=False)
print(np.array(reps))  # (N_atoms, 512)
print(np.array(reps).shape)  # (N_atoms, 512)
