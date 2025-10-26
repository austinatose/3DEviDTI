from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
mol = Chem.MolFromSmiles('CC(C(=O)NC)N(C)CC1=C(C=C2C(=C1)C(=NC=N2)NC3=C(C(=CC=C3)Cl)F)OC')
print(Chem.MolToMolBlock(mol))

import tensorflow as tf

# Check TensorFlow version
print(tf.__version__)

# Perform a basic operation
hello = tf.constant("Hello, TensorFlow!")
tf.print(hello)