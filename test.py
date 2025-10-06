from rdkit import Chem
mol = Chem.MolFromSmiles('C1CCC1C2CC2')
print(Chem.MolToMolBlock(mol))