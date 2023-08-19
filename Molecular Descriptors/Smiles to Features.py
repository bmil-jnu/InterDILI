import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import QED
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdMolDescriptors

from sklearn.preprocessing import StandardScaler

# smiles
smiles_list = []
for i in range(len(train_df)):
    train_df['smiles'][i] = str(train_df['smiles'][i])

# transform smiles to mol
mols = [Chem.MolFromSmiles(smiles) for smiles in train_df["smiles"]]

# if smiles don't transform to mol, add to non_list
none_list = []
for i in range(len(mols)):
    if mols[i] is None:
        none_list.append(i)
        print('add to none_list')

reg_idx = 0
for i in none_list:
    del mols[i - reg_idx]
    reg_idx += 1

# modify index
if len(none_list) != 0:
    train_df = train_df.drop(none_list, axis=0)
    train_df = train_df.reset_index(drop=True)


# create fingerprint
bit_info_list = []  # bit vector
bit_info = {}  # bit vector
fps = []

b = 0

# mol to fingerprint Bit Vector
for a in mols:
    fps.append(AllChem.GetMorganFingerprintAsBitVect(a, 3, nBits=1024, bitInfo=bit_info))
    bit_info_list.append(bit_info.copy())

# to array
arr_list = list()
for i in range(len(fps)):
    array = np.zeros((0,), dtype=np.int8)
    arr_list.append(array)

for i in range(len(fps)):
    bit = fps[i]
    DataStructs.ConvertToNumpyArray(bit, arr_list[i])

x = np.stack([i.tolist() for i in arr_list])
x = x.astype(np.float32)
finprt = pd.DataFrame(x)


# create physicochemical properties
qe = [QED.properties(mol) for mol in mols]
qe = pd.DataFrame(qe)

#QED datapreprocessing
ss = StandardScaler()
ss.fit(qe)
qe_scaled = ss.transform(qe)
qe_scaled = pd.DataFrame(qe_scaled)
qe_scaled.columns =['MW','ALOGP','HBA','HBD','PSA','ROTB','AROM','ALERTS']
final = pd.concat([finprt,qe_scaled,train_t],axis=1)