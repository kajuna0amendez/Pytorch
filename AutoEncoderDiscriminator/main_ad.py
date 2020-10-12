# -*- coding: utf-8 -*-
#!/usr/bin/env python
__author__ = "Andres Mendez-Vazquez"
__copyright__ = "Copyright 2019"
__credits__ = ["Andres Mendez-Vazquez"]
__license__ = "closed"
__version__ = "v1.0.0"
__maintainer__ = "Andres Mendez-Vazquez"
__email =  "kajuna0kajuna@gmail.com"
__status__ = "Development"

import random
import torch
from data_loader import data_loader, data_secondary_gen
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from discriminator import Discriminator
from autoencoder import AutoEncoder
import numpy as np
from config import ngpu, lr, beta1, beta2, num_epochs, manualSeed,\
                   plot_some_images, plot_machines, plot_results,\
                   activation
from torch import nn
from Tools import weights_init
import torch.optim as optim

if __name__ == '__main__':
    
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)    

    # Get data
    dataloader, dataloader, device = data_loader()
    
    # Generate Autoencoder
    netAE = AutoEncoder(ngpu, activation).to(device)
    
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netAE.apply(weights_init)
    
    if plot_machines == True:
        # Print the model
        print(netAE)
    
    # Create the Discriminators
    netD = Discriminator(ngpu, activation).to(device) 
    
    netSD =  Discriminator(ngpu, activation).to(device) 
    
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    
    if plot_machines == True:    
        # Print the model
        print(netD)
    
    
    # Plot some training images
    if plot_some_images == True:
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0][0].to(device)[:64],\
                                                 padding=2, normalize=True).cpu(),\
                                                 (1,2,0)))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Occluded Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0][1].to(device)[:64],\
                                                 padding=2, normalize=True).cpu(),\
                                                 (1,2,0)))
        plt.show(block=True)
    
    
    
    # Initialize Loss function
    MSE = nn.MSELoss(reduction='sum')
    #MSE = nn.L1Loss()
    
    BCE1 = nn.BCELoss()
    BCE2 = nn.BCELoss()
    
    # Establish convention for real and fake labels during training
    # For the Discriminator
    real_label = 1
    fake_label = 0
    
    # Setup Adam optimizers for both AE and D
    optimizerAE = optim.Adam(netAE.parameters(),\
                            lr=lr, betas=(beta1, beta2),\
                                amsgrad = True)
    optimizerD = optim.Adam(netD.parameters(),\
                            lr=lr, betas=(beta1, beta2),\
                                amsgrad = True) 
        
    optimizerSD = optim.Adam(netD.parameters(),\
                            lr=lr, betas=(beta1, beta2),\
                                amsgrad = True) 
        
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))
    
    
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        print('Epoch %d'%(epoch))
        for i, data in enumerate(dataloader, 0):

            # Clean the discriminator and generator
            optimizerD.zero_grad() 
            optimizerAE.zero_grad()
            optimizerSD.zero_grad()
            
            # Format batch and move to device
            image_remove = data[0][1].to(device)
        
        
            # labels are images 
            label = data[0][0].to(device)
            
            # Forward pass real batch through AE
            output_fake_ae = netAE(image_remove)

            image_gen = output_fake_ae.cpu()
            
            real_section = data[0][2].to(device)
            
            a = data[0][3][0]
            
            b = data[0][3][1]
            
            gen_image_sec = data_secondary_gen(image_gen, a, b).to(device)
   
    
            # Output Real Labels
            output_real_ae = netAE(label)
            output_real_d = netD(output_real_ae)
            
            # Train the Discriminator True Label
            r_size = output_real_d.size()
            rlabel = torch.full(r_size, real_label, device=device)
                    
            errD_real = BCE1(output_real_d, rlabel)
                        
            output_fake_d = netD(output_fake_ae)
            
            # Train the Discriminator False Label
            f_size = output_fake_d.size()
            flabel = torch.full(f_size, fake_label, device=device)

            errD_fake = BCE1(output_fake_d, flabel)
            
            DCost = errD_real + errD_fake 
            
            #DCost.backward(retain_graph=True)
            
            # Train the Secondary Discriminator
            output_real_sd = netSD(real_section)
            r_size_sd = output_real_sd.size()
            rlabel_sd = torch.full(r_size_sd, real_label, device=device)
            errSD_real = BCE2(output_real_sd, rlabel_sd)
            output_fake_sd = netSD(gen_image_sec)
            f_size_gen = output_fake_sd.size()
            flabel_sd = torch.full(f_size_gen, fake_label, device=device)
            errSD_fake = BCE2(output_fake_sd, flabel_sd)
            
            SDCost = errSD_real + errSD_fake 
            
            #SDCost.backward(retain_graph=True)
            
            # Calculate loss
            errAE_opt = MSE(output_fake_ae, label)
            errAE_opt_real = MSE(output_real_ae, label)
            AE_x = errAE_opt.item()
            AE_real = errAE_opt_real.item()
            
            df_size = output_fake_d.size()
            dtlabel = torch.full(df_size, real_label, device=device)
            
            g_cost_d = BCE1(output_fake_d,  dtlabel) 
            
            g_cost_rd = BCE1(output_real_d,  dtlabel)
            
            secr_size_sd = output_fake_sd.size()
            seclabel_sd = torch.full(secr_size_sd, real_label, device=device)
            g_cost_sec = BCE2(output_fake_sd, seclabel_sd)
            
            
            # Final Cost Function 
            AECost = errAE_opt + g_cost_d + errAE_opt_real + g_cost_rd + g_cost_sec

            
            #Calculate gradients for AE in backward pass
            AECost.backward(retain_graph=True)
            
            DCost.backward()
            
            SDCost.backward()
            
            optimizerAE.step()
            
            optimizerD.step()
            
            optimizerSD.step()

            # Output training stats
            if i % 1 == 0:
                #\t AE_MSE Real: %f'\
                print('[%d/%d][%d/%d]\tLoss_AE_MSE Occlussion: %f\t AE_MSE Real: %f'\
                      % (epoch, num_epochs, i, len(dataloader),\
                          AE_x,  AE_real))
                #,  AE_real 
            
        
    # Plot some testing images
    if plot_results == True:# and i%100==0:
        real_batch = next(iter(dataloader))

        
        images_transform = real_batch[0][1].to(device)
        
        Result = netAE(images_transform)
        
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Testing Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0][0].to(device)[:4],\
                                                 padding=2, normalize=True).cpu(),\
                                                 (1,2,0)))
        
        
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Testing Images with Occlussion")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0][1].to(device)[:4],\
                                                 padding=2, normalize=True).cpu(),\
                                                 (1,2,0)))
        
        
            
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Rebuild Images")
        plt.imshow(np.transpose((vutils.make_grid(Result[:4],\
                                                 padding=2, normalize=True).cpu()).detach(),\
                                                 (1,2,0)))
 
        plt.show(block=True)            
            