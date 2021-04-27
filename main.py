import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from autoencoder import VAE
from loss import kl_loss,reconstruction_loss
from trainer import trainer

device = torch.device("cuda:0")
torch.cuda.empty_cache()

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

vae = VAE().to(device)

trainer(vae,train_loader,test_loader,batch_size_train,batch_size_test)
