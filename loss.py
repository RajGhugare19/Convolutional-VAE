import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0")

def reconstruction_loss(recon_x,x):
    
    r_loss = F.binary_cross_entropy(recon_x,x)
    return r_loss

def kl_loss(mu,logvar):
    
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss