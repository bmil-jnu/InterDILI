#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import kerastuner
import tensorflow
import pandas as pd

from tensorflow import keras


from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdMolDescriptors, QED

from keras.optimizers import Adam
from sklearn import datasets, metrics
from sklearn.metrics import auc, roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from tensorflow.keras.layers import Embedding, Dense,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K 
from tensorflow.keras import initializers


# In[ ]:


inp = pd.read_csv('Total_dataset.csv')
train_df = inp.iloc[452:,:].reset_index(drop=True)
test_df = inp.iloc[:452,:]
train_s = train_df.iloc[:,:-1]
train_t = train_df['toxicity']
test_s = test_df.iloc[:,:-1]
test_t = test_df['toxicity']


# In[4]:


inp.groupby('toxicity')['ref'].value_counts()


# # Train

# In[ ]:


# smiles 
smiles_list = []
for i in range(len(train_df)):
    train_df['smiles'][i] = str(train_df['smiles'][i])

# mol to smiles
mols = [Chem.MolFromSmiles(smiles) for smiles in train_df["smiles"]]

# if mol isn't transform to smiles, to none_list
none_list = []
for i in range(len(mols)):
    if mols[i] is None :
        none_list.append(i)
        print('add to none_list')
    
reg_idx = 0
for i in none_list :
    del mols[i - reg_idx]
    reg_idx += 1
    
# modify index
if len(none_list) != 0 :
    train_df = train_df.drop(none_list, axis=0)
    train_df = train_df.reset_index(drop = True)


# In[9]:


train_df['toxicity'].value_counts()


# In[10]:


# create fingerprint 
bit_info_list = [] 
bit_info = {} 
fps = []

b = 0

# mol to fingerprint Bit Vector
for a in mols :
    fps.append(AllChem.GetMorganFingerprintAsBitVect(a, 3, nBits = 1024, bitInfo = bit_info))
    bit_info_list.append(bit_info.copy()) 
    
# to array
arr_list = list()
for i in range(len(fps)):
    array = np.zeros((0,), dtype = np.int8)
    arr_list.append(array)
    
for i in range(len(fps)):
    bit = fps[i]
    DataStructs.ConvertToNumpyArray(bit, arr_list[i])
    
x = np.stack([i.tolist() for i in arr_list])
x = x.astype(np.float32)
finprt = pd.DataFrame(x)


# In[11]:


# physicochemical properties

qe = [QED.properties(mol) for mol in mols]
qe = pd.DataFrame(qe)
qe


# In[13]:


#QED datapreprocessing 
ss = StandardScaler()

ss.fit(qe)
qe_scaled = ss.transform(qe) 


qe_scaled = pd.DataFrame(qe_scaled)
qe_scaled.columns =['MW','ALOGP','HBA','HBD','PSA','ROTB','AROM','ALERTS']
qe_scaled


# In[14]:


from pickle import dump

#dump(ss, open('./standard_scaler.pkl','wb'))


# In[ ]:


final = pd.concat([finprt,qe_scaled,train_t],axis=1)
x_train = final.iloc[:,:1032]
y_train = final['toxicity']


# ## Test 

# In[ ]:


# smiles 
smiles_list = []
for i in range(len(test_df)):
    test_df['smiles'][i] = str(test_df['smiles'][i])

# to mol 
mols = [Chem.MolFromSmiles(smiles) for smiles in test_df["smiles"]]

# if smiles don't transform to mol, add to none_list 
none_list = []
for i in range(len(mols)):
    if mols[i] is None :
        none_list.append(i)
        print('add to none_list')
    
reg_idx = 0
for i in none_list :
    del mols[i - reg_idx]
    reg_idx += 1
    
# modify index
if len(none_list) != 0 :
    test_df = test_df.drop(none_list, axis=0)
    test_df = test_df.reset_index(drop = True)


# create fingerprint 
bit_info_list = [] 
bit_info = {} 
fps = []

b = 0

# mol to fingerprint Bit Vector 
for a in mols :
    fps.append(AllChem.GetMorganFingerprintAsBitVect(a, 3, nBits = 1024, bitInfo = bit_info))
    bit_info_list.append(bit_info.copy())
    
# to array 
arr_list = list()
for i in range(len(fps)):
    array = np.zeros((0,), dtype = np.int8)
    arr_list.append(array)
    
for i in range(len(fps)):
    bit = fps[i]
    DataStructs.ConvertToNumpyArray(bit, arr_list[i])
    
x = np.stack([i.tolist() for i in arr_list])
x = x.astype(np.float32)
finprt = pd.DataFrame(x)


# In[19]:


# physicochemical properties

qe = [QED.properties(mol) for mol in mols]
qe = pd.DataFrame(qe)
qe


# In[24]:


from pickle import load 

#QED datapreprocessing 
load_ss = load(open('./standard_scaler.pkl','rb'))
qe_scaled = load_ss.transform(qe)
qe_scaled = pd.DataFrame(qe_scaled, columns=['MW','ALOGP','HBA','HBD','PSA','ROTB','AROM','ALERTS'])
qe_scaled


# In[ ]:


final = pd.concat([finprt,qe_scaled,test_t],axis=1)
x_test = final.iloc[:,:1032]
y_test = final['toxicity']


# # Model train

# In[23]:


x = x_train
y = y_train


# In[ ]:


# split train validation set 

x_train, x_val, y_train,y_val = train_test_split(x,y,train_size=0.8, random_state=42)


# In[ ]:


# input data dimension
input_dim = x_train.shape[1]

# Layer weight initializers 
initializer = tf.keras.initializers.HeNormal()

# L2 regularizer 
from tensorflow.keras import regularizers
regularizer = regularizers.l2(0.001)

#model hyperparameter
epochs = 100
batch_size = 32

#callbacks
callbacks = [
    tensorflow.keras.callbacks.ModelCheckpoint(
        "DILIattention.h5", save_best_only=True, monitor="val_loss"
    ),
    tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, verbose=1),
]


# In[140]:


# PregTaboo
from keras.layers import Dense, Dropout, MultiHeadAttention

# Model
inputs = tf.keras.layers.Input(shape=(input_dim,))
dense_v = tf.keras.layers.Dense(input_dim, activation = None)(inputs)
attn_score = tf.keras.layers.Softmax(axis = -1)(dense_v)
cal_score = tf.math.multiply(inputs, attn_score)
Dense1 = tf.keras.layers.Dense(512, activation = 'relu', 
                          kernel_initializer = initializer)(cal_score)
Dense1_BN = tf.keras.layers.BatchNormalization()(Dense1)
Dropout = Dropout(rate=0.25)(Dense1_BN)
outputs = tf.keras.layers.Dense(1, activation = 'sigmoid')(Dropout)


# In[141]:


model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()


# In[142]:




model.compile(
    optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
    loss="binary_crossentropy",
    metrics=["accuracy"],
    )
    
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data = (x_val, y_val),
    callbacks = callbacks
    )


# In[144]:


# ROC curve
from sklearn.metrics import roc_auc_score,roc_curve,auc
preds = model.predict(x_test) #x_test: DILIrank dataset

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

roc = pd.DataFrame({
    'FPR': fpr,
    'TPRate': tpr,
    'Threshold': threshold
})

optimal_idx=np.argmax(tpr-fpr)
optimal_threshold= threshold[optimal_idx]


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[147]:


from sklearn.metrics import precision_recall_curve

# precision, recall, threshold 
precision, recall, thresholds = precision_recall_curve(y_test, preds)
# pr_auc 
pr_auc = auc(recall, precision)
# F1 score 
F1 = 2*(precision*recall)/(precision+recall)

plt.title('PR curve')
plt.plot(recall, precision, 'b', label = 'AUC = %0.2f' % pr_auc)
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()


# In[149]:


d_t = pd.DataFrame(y_test)
d_t = d_t.reset_index(drop=True)
d_preds = pd.DataFrame(preds)
pred_proba = pd.concat([d_t,d_preds],axis=1)
pred_proba.columns=['toxicity','prediction']
pred_proba.sort_values(by='prediction',ascending=False)
preds_1d = preds.flatten()
pred_class = np.where(preds_1d> optimal_threshold,1,0)


# In[198]:


print('idx:',optimal_idx,', threshold:',optimal_threshold )

auc_score = roc_auc_score(y_test, preds)
accuracy_score = accuracy_score(y_test,pred_class)
Precision = precision_score(y_test,pred_class)
Recall = recall_score(y_test,pred_class)
print('AUC:',auc_score)
F1_score = 2*(Precision*Recall)/(Precision+Recall)


# In[199]:


# model performance
test_preds = model.predict(x_test)

test_preds[test_preds >= optimal_threshold] = 1
test_preds[test_preds < optimal_threshold] = 0

Precision = precision_score(y_test, test_preds)
Recall = recall_score(y_test, test_preds)
model_roc = model.predict(x_test)
F1_score = 2*(Precision*Recall)/(Precision+Recall)

print('accuracy : {0}'.format(accuracy_score(y_test, test_preds)))
print('Precision : {0}'.format(Precision))
print('Recall : {0}'.format(Recall))
print('ROC_score : {0}'.format(roc_auc_score(y_test, model_roc)))
print("F1 score : {0}".format(F1_score))


# In[200]:


tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
specificity = tn / (tn+fp)
specificity


# In[ ]:




