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
import dict_learning.Dictionary_Model as modelDL
import yaml
import os
import pickle
from pathlib import Path
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean


print("Cuda Available: %s, GPU Count: %s" % (torch.cuda.is_available(), torch.cuda.device_count()))


def load_config(config_file_path):
    with open(config_file_path, 'r') as f:
        return yaml.safe_load(f)


class Dictionary_Learning(object):

    def __init__(self, config_file, train=True, load_model_type=None):
        torch.cuda.empty_cache()
        if type(config_file) is not dict:
            self.config_file = load_config(config_file)
        else:
            self.config_file = config_file

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if train:
            self.hyperparameters = self.set_hyperparameters()
            self.dataset, self.dataloader = self.torch_data_loader()

            self.model = self.define_model()

            self.train_model()
            self.save_model()
            self.save_config()

        else:
            if load_model_type == "gpu":
                self.model = self.load_model(return_cpu_model=False)
            else:
                self.model = self.load_model(return_cpu_model=True)

        torch.cuda.empty_cache()

    def define_model(self):
        n_features = self.config_file['dictionary']['filter_size'] ** 2 * self.config_file['file']['n_channels']
        dictionary_size = self.config_file['dictionary']['dictionary_size']
        model = modelDL.DictLearn(dictionary_size, n_features).to(self.device)
        return model

    def torch_data_loader(self):
        project_dir = os.getenv("PROJECT_DIR")
        file_name = self.config_file['file']['name']
        batch_size = self.config_file['dictionary']['batch_size']

        data_transform = transforms.Compose([D.ToTensor()])
        dict_dataset = D.DictionaryDatasetSingleImage(
            # img_path=f'{project_dir}/data/slides/{file_name}',
            img_path=f'data/slides/{file_name}',
            hyperparameters=self.hyperparameters,
            transform=data_transform)
        dataloader = D.DataLoader(dict_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        return dict_dataset, dataloader

    def set_hyperparameters(self):
        config = self.config_file
        hyperparameters = {
            "K": config['dictionary']['dictionary_size'],
            "m": config['dictionary']['number_of_patches'],  # number of sub_patch images of size (filter_size x filter_size)
            "filter_size": config['dictionary']['filter_size'],
            "eps": config['dictionary']['whitening_eps'],
            "n_channels": config['file']['n_channels'],  # 1 when grayscale
            "epoch": config['dictionary']['epochs'],
            "batch_size": config['dictionary']['batch_size'],
            "learning_rate": config['dictionary']['learning_rate'],
            "momentum": config['dictionary']['momentum'],
            "file_name": config['file']['name']
        }
        return hyperparameters

    def train_model(self):
        LR = self.config_file['dictionary']['learning_rate']
        MNT = self.config_file['dictionary']['momentum']
        optimizer = torch.optim.SGD(self.model.parameters(), lr=LR, momentum=MNT)
        loss_func = nn.MSELoss()

        # Training
        EPOCH = self.config_file['dictionary']['epochs']
        BATCH_SIZE = self.config_file['dictionary']['batch_size']
        Error = np.zeros((EPOCH,))
        Nnz = np.zeros((EPOCH,))
        Loss = np.zeros((EPOCH,))
        N = len(self.dataset)
        DISPLAY_FREQ = 1
        TSHOW = np.round(DISPLAY_FREQ * N / BATCH_SIZE)  # times per EPOCH to display information
        t0 = time.perf_counter()
        SC = self.config_file['dictionary']['optimizer']  # 'fista' or 'IHT'
        K = self.config_file['dictionary']['max_nonzero']  # sparsity parameter: float numer if 'fista' or cardinality constraint if 'IHT'

        Err = []

        for epoch in range(EPOCH):
            print(f"EPOCH: {epoch}")
            for step, sample in enumerate(self.dataloader):
                x = sample['image']
                b_x = (x.view(x.shape[0], -1)).double().to(self.device)  # batch x, shape (batch, 28*28)

                b_y = b_x.clone()

                decoded, encoded, errIHT = self.model(b_x, SC, K)

                loss = loss_func(decoded, b_y)  # mean square error
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                self.model.zero_grad()

                if SC == 'IHT':
                    Loss[epoch] = Loss[epoch] + loss.data.item()
                elif SC == 'fista':
                    Loss[epoch] = Loss[epoch] + loss.data.item() + K * np.sum(np.abs(encoded.cpu().numpy()))
                decoded = decoded.detach()
                err = np.linalg.norm((decoded - b_x).cpu().numpy(), 'fro') / np.linalg.norm(b_x.cpu().numpy(), 'fro')
                Error[epoch] = Error[epoch] + errIHT[-1]
                Err = np.append(Err, errIHT[-1])
                Nnz[epoch] = Nnz[epoch] + np.count_nonzero(encoded.cpu().numpy()) / encoded.cpu().numpy().size

            #         # for debugging:
            #         print(Error[epoch]/(step+1))
            #         if step%50==0:
            #             plt.plot(errIHT); plt.show()

            Loss[epoch] /= len(self.dataloader)
            Error[epoch] /= (step + 1)
            Nnz[epoch] /= (step + 1)
            print('Epoch: ', epoch, ', Error: ', Error[epoch], ', | train loss: %.3e' % Loss[epoch],
                  ' NNZ/(1-sparsity): ', Nnz[epoch])

        ##### save_plots
        # Error Evolution
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(Err); plt.grid(); plt.title('Reconstruction Error'); plt.xlabel('mini-bach / dict update');
        plt.subplot(1, 2, 2)
        plt.plot(Loss); plt.grid(); plt.title('Loss Evolution'); plt.xlabel('epoch')
        plt.show();
        file_name = self.config_file['file']['name'][:-5]
        path = Path(f"saved/dictionary/{file_name}/plots/")
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(path / f"loss_evolution_{EPOCH}_epochs.jpg"))

        # Sample of Trained Dictionary Elements
        W = self.model.cpu().W.data.numpy().copy()
        M1 = modelDL.showFilters(W, 10, 20)
        plt.figure(figsize=(15, 15))
        plt.imshow(rescale(M1, 4, mode='constant'), cmap='gray')
        plt.axis('off')
        plt.show()
        plt.savefig(f"dict_{EPOCH}_epochs.jpg")

    def save_model(self):
        file_name = self.config_file['file']['name'][:-5]
        EPOCH = self.config_file['dictionary']['epochs']
        path_gpu = Path(f"saved/dictionary/{file_name}/gpu/")
        path_gpu.mkdir(parents=True, exist_ok=True)
        path_cpu = Path(f"saved/dictionary/{file_name}/cpu/")
        path_cpu.mkdir(parents=True, exist_ok=True)

        with open(str(path_gpu / f'dict_model_{EPOCH}_epochs.pkl'), 'wb') as f:
            pickle.dump(self.model, f)

        with open(str(path_cpu / f'dict_model_{EPOCH}_epochs.pkl'), 'wb') as f:
            pickle.dump(self.model.to('cpu'), f)

        project_dir = Path(os.getenv('PROJECT_DIR'))

        self.config_file['saved']['gpu_dict_model'] = str(project_dir / path_gpu / f'dict_learner_{EPOCH}_epochs.pkl')
        self.config_file['saved']['cpu_dict_model'] = str(project_dir / path_cpu / f'dict_learner_{EPOCH}_epochs.pkl')

    def save_config(self):
        file_name = self.config_file['file']['name'][:-5]  # name without extention
        with open(f"configs/{file_name}.yml", "w") as f:
            yaml.dump(self.config_file, f, default_flow_style=False)

    def load_model(self, return_cpu_model=False):
        path_gpu = self.config_file['saved']['gpu_dict_model']
        path_cpu = self.config_file['saved']['cpu_dict_model']

        if return_cpu_model is False:
            with open(str(path_gpu), 'rb') as f:
                dict_model_gpu = pickle.load(f)
            return dict_model_gpu

        else:
            with open(str(path_cpu), 'rb') as f:
                dict_model_cpu = pickle.load(f)
            return dict_model_cpu


#
# # Hyper Parameters
# EPOCH = 150
# BATCH_SIZE = 1000
# LR = .5  # learning rate
# MNT = 0.9   # momentum variable
# DOWNLOAD_Dataset = False
# N_TEST_IMG = 5
# dictionary_size = 250     # Dictionary Size
#
# hyperparameters = {
#         "K": dictionary_size,
#         "m": 100000,  # number of sub_patch images of size (filter_size x filter_size)
#         "filter_size": 16,
#         "eps": 0.1,
#         "n_channels": 1,  # 1 when grayscale
# }
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# data_transform = transforms.Compose([D.ToTensor()])
# dict_dataset = D.DictionaryDatasetSingleImage(img_path='/home/vagrawal/Projects/MaLTT/data/slides/02B_H&E_Control.jpeg',
#                                                hyperparameters=hyperparameters,
#                                                transform=data_transform)
# dataloader = D.DataLoader(dict_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#
#
#
#
#
# # The dictionary model, as well as the dictionary learning method, is defined in Simple_DL.py file.
# # Here the model is initialized, we define the optimizer and the loss function to be the $\ell_2$ loss
#
# # In[63]:
#
#
# # import importlib
# # importlib.reload(modelDL)
#
#
# # In[64]:
#
#
#
#
# n_features = hyperparameters['filter_size']**2 * hyperparameters['n_channels']
# model = modelDL.DictLearn(dictionary_size, n_features).to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MNT)
# loss_func = nn.MSELoss()
#
#
# # Beginning Training!
#
# # In[65]:
#
#
# # import importlib
# # importlib.reload(modelDL)
#
# # Training
# Error = np.zeros((EPOCH,))
# Nnz = np.zeros((EPOCH,))
# Loss = np.zeros((EPOCH,))
# N = len(dict_dataset)
# DISPLAY_FREQ = 1 ;
# TSHOW = np.round(DISPLAY_FREQ * N/BATCH_SIZE) # times per EPOCH to display information
# t0 = time.perf_counter()
# SC = 'IHT' # 'fista' or 'IHT'
# K = 1       # sparsity parameter: float numer if 'fista' or cardinality constraint if 'IHT'
#
# Err = []
#
#
# # In[81]:
#
#
# for epoch in range(EPOCH):
#     for step, sample in enumerate(dataloader):
#         x = sample['image']
#         b_x = (x.view(x.shape[0], -1)).double().to(device)   # batch x, shape (batch, 28*28)
#
#         b_y = b_x.clone()
#
#         decoded, encoded, errIHT = model(b_x, SC, K)
#
#         loss = loss_func(decoded, b_y)      # mean square error
#         optimizer.zero_grad()               # clear gradients for this training step
#         loss.backward()                     # backpropagation, compute gradients
#         optimizer.step()                    # apply gradients
#         model.zero_grad()
#
#         if SC == 'IHT': Loss[epoch] = Loss[epoch] + loss.data.item()
#         elif SC == 'fista': Loss[epoch] = Loss[epoch] + loss.data.item() + K * np.sum(np.abs(encoded.cpu().numpy()))
#         decoded = decoded.detach()
#         err = np.linalg.norm( (decoded-b_x).cpu().numpy() ,'fro') / np.linalg.norm( b_x.cpu().numpy() ,'fro')
#         Error[epoch] = Error[epoch] + errIHT[-1]
#         Err = np.append(Err,errIHT[-1])
#         Nnz[epoch] = Nnz[epoch] + np.count_nonzero(encoded.cpu().numpy())/encoded.cpu().numpy().size
#
# #         # for debugging:
# #         print(Error[epoch]/(step+1))
# #         if step%50==0:
# #             plt.plot(errIHT); plt.show()
#
#     Loss[epoch] /= len(dataloader)
#     Error[epoch] /= (step+1)
#     Nnz[epoch] /= (step+1)
#     print('Epoch: ', epoch, ', Error: ', Error[epoch], ', | train loss: %.3e' % Loss[epoch], ' NNZ/(1-sparsity): ', Nnz[epoch] )
#
#
#
# # In[89]:
#
#
# import pickle
# with open(f'dict_learner_{EPOCH}_epochs.pkl', 'wb') as fid:
#     pickle.dump(model, fid)
#
#
# # In[86]:
#
#
# plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
# plt.plot(Err); plt.grid(); plt.title('Reconstruction Error'); plt.xlabel('mini-bach / dict update');
# plt.subplot(1,2,2)
# plt.plot(Loss); plt.grid(); plt.title('Loss Evolution'); plt.xlabel('epoch')
# plt.show();
# plt.savefig(f"reconstruction_{EPOCH}_epochs.jpg")
#
#
# # In[88]:
#
#
# from skimage import data, color
# from skimage.transform import rescale, resize, downscale_local_mean
#
# W = model.cpu().W.data.numpy().copy()
#
# M1 = modelDL.showFilters(W,10,20)
# plt.figure(figsize=(15,15))
# plt.imshow(rescale(M1,4,mode='constant'),cmap='gray')
# plt.axis('off')
# plt.show()
# plt.savefig(f"dict_{EPOCH}_epochs.jpg")
#
#
# # In[90]:
#
#
#
# def load_dictionary():
#     with open('dict_learner.pkl', 'rb') as fid:
#         dict_learner = pickle.load(fid)
#     return dict_learner
#
#
# # In[ ]:
#



