import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0")

class VAE(nn.Module):

    def __init__(self):
        
        super(VAE, self).__init__()
        
        self.z = 10
        self.h = 64*11*11

        self.conv1 = nn.Conv2d(1,32,kernel_size=3)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3)
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(self.h,2*self.z)
        self.fc2 = nn.Linear(self.z,self.h)
        
        self.tconv1 = nn.ConvTranspose2d(64,32,kernel_size=3)  
        self.tconv2 = nn.ConvTranspose2d(32, 1,kernel_size=3)
        self.upsamp = nn.Upsample(scale_factor = 2, mode='nearest')
        
        self.optimizer = optim.Adam(self.parameters(),lr = 0.001)

    def forward(self, x):

        x, mu, logvar = self.encode(x)
        x = self.decode(x)

        return x, mu, logvar
    
    def flatten(self, x):
        
        return x.view(x.size(0),-1)
    
    def unflatten(self,x):

        return x.view(x.size(0),64,11,11)

    def encode(self,x):

        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        mu, logvar = torch.chunk(self.fc1(x),2,1)
        z = self.reparameterize(mu,logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    
    def decode(self,x):

        x = torch.relu(self.fc2(x))
        x = self.unflatten(x)
        x = torch.relu(self.tconv1(x))
        x = self.upsamp(x)
        x = torch.sigmoid(self.tconv2(x))

        return x