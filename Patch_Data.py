#!/usr/bin/env python
# coding: utf-8

# # Dictionary Learning Demo
# #### Simple pytorch implementation of a Dictionary Learning demo employing stochastic gradient descent, on MNIST. 
# 
# See Readme.me

# Importing packages and Defining hyper-parameters

# In[80]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time
import importlib
import src.dataloader as D
from torchvision import transforms
# import matplotlib
# matplotlib.use('TkAgg')
import ipdb

# Hyper Parameters
EPOCH = 50
BATCH_SIZE = 1000
LR = .5  # learning rate
MNT = 0.9   # momentum variable
DOWNLOAD_Dataset = False
N_TEST_IMG = 5
dictionary_size = 250     # Dictionary Size

hyperparameters = {
        "K": dictionary_size,
        "m": 100000,  # number of sub_patch images of size (filter_size x filter_size)
        "filter_size": 16,
        "eps": 0.1,
        "n_channels": 1,  # 1 when grayscale
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Defining and importing training and testing data - using MNIST digits for this example

# In[17]:


# train_data = torchvision.datasets.MNIST(
#     root='../data',
#     train=True,                                     # this is training data
#     transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                               torchvision.transforms.Normalize((0.1307,), (0.3081,)),
#                              ]),    
#     download=DOWNLOAD_Dataset,                        # download it if you don't have it
# )
# # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
# train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 
# test_data = torchvision.datasets.MNIST(
#     root='../data',
#     train=False,                                     # this is testing data
#     transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                               torchvision.transforms.Normalize((0.1307,), (0.3081,)),
#                              ]),
#     download=DOWNLOAD_Dataset,                        # download it if you don't have it
# )
# test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)





data_transform = transforms.Compose([D.ToTensor()])
dict_dataset = D.DictionaryDatasetSingleImage(img_path='/home/vagrawal/Projects/MaLTT/data/slides/02B_H&E_Control.jpeg',
                                               hyperparameters=hyperparameters,
                                               transform=data_transform)
dataloader = D.DataLoader(dict_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# The dictionary model, as well as the dictionary learning method, is defined in Simple_DL.py file.
# Here the model is initialized, we define the optimizer and the loss function to be the $\ell_2$ loss

# In[63]:


# import importlib
# importlib.reload(modelDL)


# In[64]:


import Dictionary_Model as modelDL

n_features = hyperparameters['filter_size']**2 * hyperparameters['n_channels']
model = modelDL.DictLearn(dictionary_size, n_features).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MNT)
loss_func = nn.MSELoss()


# Beginning Training!

# In[65]:


# import importlib
# importlib.reload(modelDL)

# Training
Error = np.zeros((EPOCH,))
Nnz = np.zeros((EPOCH,))
Loss = np.zeros((EPOCH,))
N = len(dict_dataset)
DISPLAY_FREQ = 1 ; 
TSHOW = np.round(DISPLAY_FREQ * N/BATCH_SIZE) # times per EPOCH to display information
t0 = time.perf_counter()
SC = 'IHT' # 'fista' or 'IHT'
K = 1       # sparsity parameter: float numer if 'fista' or cardinality constraint if 'IHT'

Err = []


# In[81]:


for epoch in range(EPOCH):
    for step, sample in enumerate(dataloader):
        x = sample['image']
        b_x = (x.view(x.shape[0], -1)).double().to(device)   # batch x, shape (batch, 28*28)
        
        b_y = b_x.clone()

        decoded, encoded, errIHT = model(b_x, SC, K)
        
        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients
        model.zero_grad()
        
        if SC == 'IHT': Loss[epoch] = Loss[epoch] + loss.data.item()
        elif SC == 'fista': Loss[epoch] = Loss[epoch] + loss.data.item() + K * np.sum(np.abs(encoded.cpu().numpy()))
        decoded = decoded.detach()
        err = np.linalg.norm( (decoded-b_x).cpu().numpy() ,'fro') / np.linalg.norm( b_x.cpu().numpy() ,'fro')
        Error[epoch] = Error[epoch] + errIHT[-1]
        Err = np.append(Err,errIHT[-1])
        Nnz[epoch] = Nnz[epoch] + np.count_nonzero(encoded.cpu().numpy())/encoded.cpu().numpy().size
        
#         # for debugging:
#         print(Error[epoch]/(step+1))
#         if step%50==0:
#             plt.plot(errIHT); plt.show()
        
    Loss[epoch] /= len(dataloader)
    Error[epoch] /= (step+1)
    Nnz[epoch] /= (step+1)
    print('Epoch: ', epoch, ', Error: ', Error[epoch], ', | train loss: %.3e' % Loss[epoch], ' NNZ/(1-sparsity): ', Nnz[epoch] )
    


# In[89]:


import pickle
with open('dict_learner.pkl', 'wb') as fid:
    pickle.dump(model, fid)


# In[86]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(Err); plt.grid(); plt.title('Reconstruction Error'); plt.xlabel('mini-bach / dict update');
plt.subplot(1,2,2)
plt.plot(Loss); plt.grid(); plt.title('Loss Evolution'); plt.xlabel('epoch')
plt.show(); 
plt.savefig("reconstruction.jpg")


# In[88]:


from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

W = model.cpu().W.data.numpy().copy()

M1 = modelDL.showFilters(W,10,20)
plt.figure(figsize=(15,15))
plt.imshow(rescale(M1,4,mode='constant'),cmap='gray')
plt.axis('off')
plt.show()
plt.savefig("dict.jpg")


# In[90]:



def load_dictionary():
    with open('dict_learner.pkl', 'rb') as fid:
        dict_learner = pickle.load(fid)
    return dict_learner


# In[ ]:




