import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def trainer(vae,train_loader,test_loader,batch_size_train,batch_size_test):

    for epoch in range(20):
        print("epoch number ", epoch)
        train_loss = 0
        val_loss = 0
        for idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            recon_images, mu, logvar = vae(images)
            r_loss = reconstruction_loss(recon_images, images)
            k_loss = kl_loss(mu, logvar)
            loss = r_loss + k_loss
            vae.optimizer.zero_grad()
            loss.backward()
            vae.optimizer.step()
            train_loss += loss.item()
        print("train loss ", train_loss/(len(train_loader)*batch_size_train))
        for idx, (images, _) in enumerate(test_loader):
            images = images.to(device)
            recon_images, mu, logvar = vae(images)
            r_loss = reconstruction_loss(recon_images, images)
            k_loss = kl_loss(mu, logvar)
            loss = r_loss + k_loss
            val_loss += loss.item()
        print("val loss ", val_loss/(len(test_loader)*batch_size_test))

        save_model(vae, chk_pnt=epoch)
    
def save_model(model,chk_pnt,best_model = False):
    
    if best_model == False:
        save_path = 'saved_models/' + str(chk_pnt) + '.pt'
    else:
        save_path = 'saved_models/best.pt'
    
    torch.save(model,save_path)