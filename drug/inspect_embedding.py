import torch
import numpy as np

# this is data, i don't want to load as weights
reps = torch.load("drug/embeddings_atomic/DB00482_unimol.pt", weights_only=False)
print(reps.keys())
print(np.array(reps["atomic_reprs"]))  # (N_atoms, 512)
print(np.array(reps["atomic_reprs"]).shape)  # (N_atoms, 512)
print(np.array(reps["atomic_symbol"]))  # (N_atoms, 512)
print(np.array(reps["atomic_symbol"]).shape)  # (N_atoms, 512)

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# Create a molecule object from a SMILES string (e.g., ethanol)
mol = Chem.MolFromSmiles("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F")
mol = Chem.AddHs(mol)
if mol is not None:
    # Iterate over each atom in the molecule using GetAtoms()
    for atom in mol.GetAtoms():
        print(f"Atom index: {atom.GetIdx()}, Symbol: {atom.GetSymbol()}, Atomic Number: {atom.GetAtomicNum()}")
else:
    print("Invalid SMILES string, molecule could not be generated.")

# ----------------------------
# RDKit 3D conformer generation + saving with labels
# ----------------------------
out_dir = "drug/rdkit_conformers"
import os
os.makedirs(out_dir, exist_ok=True)

# Build 3D conformer (ETKDG) and optimize geometry
mol3d = Chem.MolFromSmiles(
    "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F"
)
if mol3d is None:
    raise ValueError("Invalid SMILES; cannot generate conformer")

mol3d = Chem.AddHs(mol3d)
params = AllChem.ETKDGv3()
params.randomSeed = 0xC0FFEE  # deterministic
conf_id = AllChem.EmbedMolecule(mol3d, params)
if conf_id < 0:
    raise RuntimeError("RDKit failed to embed a 3D conformer")

# Try MMFF first; fall back to UFF
try:
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol3d, mmffVariant="MMFF94s")
    if mmff_props is not None:
        AllChem.MMFFOptimizeMolecule(mol3d, mmff_props)
    else:
        AllChem.UFFOptimizeMolecule(mol3d)
except Exception:
    AllChem.UFFOptimizeMolecule(mol3d)

# Label atoms in the 3D file via AtomMapNum (many viewers display as :idx)
for a in mol3d.GetAtoms():
    a.SetAtomMapNum(a.GetIdx())

# Save 3D conformer as SDF and PDB
sdf_path = os.path.join(out_dir, "DB00482_rdkit3d_labeled.sdf")
pdb_path = os.path.join(out_dir, "DB00482_rdkit3d.pdb")

w = Chem.SDWriter(sdf_path)
w.write(mol3d)
w.close()

Chem.MolToPDBFile(mol3d, pdb_path)

print(f"\nSaved RDKit 3D conformer:\n  SDF (labeled via AtomMapNum): {sdf_path}\n  PDB: {pdb_path}")

# Also save a 2D depiction with atom indices drawn (clean for reports)
# (2D depiction is separate from 3D coordinates.)
mol2d = Chem.Mol(mol3d)
AllChem.Compute2DCoords(mol2d)

png_path = os.path.join(out_dir, "DB00482_atom_indices.png")
img = Draw.MolToImage(mol2d, size=(900, 600), legend="DB00482 atom indices")
img.save(png_path)
print(f"Saved 2D labeled depiction: {png_path}")