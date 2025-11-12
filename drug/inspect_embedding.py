import torch
import numpy as np

# this is data, i don't want to load as weights
reps = torch.load("drug/embeddings_atomic/DB00006_unimol.pt", weights_only=False)
print(np.array(reps["atomic_reprs"]).reshape(-1, 512))  # (N_atoms, 512)
print(np.array(reps["atomic_reprs"]).reshape(-1, 512).shape)  # (N_atoms, 512)
