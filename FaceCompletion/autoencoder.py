# -*- coding: utf-8 -*-
#!/usr/bin/env python
__author__ = "Andres Mendez-Vazquez"
__copyright__ = "Copyright 2020"
__credits__ = ["Andres Mendez-Vazquez"]
__version__ = "v1.0.0"
__maintainer__ = "Andres Mendez-Vazquez"
__email =  "kajuna0kajuna@gmail.com"
__status__ = "Development"

from torch import nn
from config import nc, ngf, k_size, k_padding, image_size

class AutoEncoder(nn.Module):
    def __init__(self, ngpu, activation):
        
        super(AutoEncoder, self).__init__()
        self.activation = activation
        self.ngpu = ngpu
        
        # ENCODER SECTION REDUCING IMAGE SIZE
        
        # HERE STRIDES FOR REDUCTION INSTEAD OF 
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels = nc, out_channels = ngf, stride = 1,\
                      kernel_size=k_size, padding=k_padding),
            nn.Conv2d(in_channels = ngf, out_channels = 2*ngf, stride = 2,\
                      kernel_size=k_size, padding=k_padding),
            nn.BatchNorm2d(2*ngf),
            self.activation_func()
            )
            
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels = 2*ngf, out_channels = ngf*2, stride = 1,\
                      kernel_size=k_size, padding=k_padding),
            nn.Conv2d(in_channels = 2*ngf, out_channels = 4*ngf, stride = 2,\
                      kernel_size=k_size, padding=k_padding),
            nn.BatchNorm2d(4*ngf),
            self.activation_func()
            )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels = ngf*4, out_channels = ngf*4, stride = 1,\
                      kernel_size=k_size, padding=k_padding),
            nn.Conv2d(in_channels = 4*ngf, out_channels = 8*ngf, stride = 2,\
                      kernel_size=k_size, padding=k_padding),
            nn.BatchNorm2d(8*ngf),
            self.activation_func()
            )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(in_channels = ngf*8, out_channels = ngf*8, stride = 1,\
                      kernel_size=k_size, padding=k_padding),
            nn.Conv2d(in_channels = 8*ngf, out_channels = 16*ngf, stride = 2,\
                      kernel_size=k_size, padding=k_padding),
            nn.BatchNorm2d(16*ngf),
            self.activation_func()
            )        

        # FULLY CONNECTED LAYER FOR THE CODIFICATION OF THE IMAGE
            
        self.Fully_Connected1 = nn.Sequential(
                nn.BatchNorm2d(ngf*16),
                nn.Linear(in_features = image_size//(2**4),\
                           out_features = image_size//(2**4),\
                          bias=True),
                nn.Linear(in_features = image_size//(2**4),\
                         out_features = image_size//(2**4),\
                          bias=True),
                nn.BatchNorm2d(ngf*16)
                )
            
        # DECODE REBUILD THE IMAGE FOR THE CODIFICATION USING ConvTranspose2d 

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = ngf*16, out_channels = ngf*16, stride = 1,\
                      kernel_size=k_size, padding=k_padding),
            nn.ConvTranspose2d(in_channels = 16*ngf, out_channels = 8*ngf, stride = 2,\
                      kernel_size=k_size, padding=k_padding, output_padding = 1),
            nn.BatchNorm2d(8*ngf),
            self.activation_func()
            ) 

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = ngf*8, out_channels = ngf*8, stride = 1,\
                      kernel_size=k_size, padding=k_padding),
            nn.ConvTranspose2d(in_channels = 8*ngf, out_channels = 4*ngf, stride = 2,\
                      kernel_size=k_size, padding=k_padding, output_padding = 1),
            nn.BatchNorm2d(4*ngf),
            self.activation_func()
            )
        
            
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 4*ngf, out_channels = ngf*4, stride = 1,\
                      kernel_size=k_size, padding=k_padding),
            nn.ConvTranspose2d(in_channels = 4*ngf, out_channels = 2*ngf, stride = 2,\
                      kernel_size=k_size, padding=k_padding, output_padding = 1),
            nn.BatchNorm2d(2*ngf),
            self.activation_func()
            )
            
        self.decoder4 = nn.Sequential(
            #nn.BatchNorm2d(2*ngf),
            nn.ConvTranspose2d(in_channels = 2*ngf, out_channels = 2*ngf, stride = 1,\
                      kernel_size=k_size, padding=k_padding),
            nn.ConvTranspose2d(in_channels = 2*ngf, out_channels = nc, stride = 2,\
                      kernel_size=k_size, padding=k_padding, output_padding = 1)
            )
             
        self.TANH = nn.Tanh()

    def activation_func(self):
        return  nn.ModuleDict([
            ['relu', nn.ReLU(inplace=True)],
            ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
            ['selu', nn.SELU(inplace=True)],
            ['none', nn.Identity()]
        ])[self.activation]        


    
    def forward(self, x):
        

        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = self.Fully_Connected1(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x) 
        x = self.decoder4(x)
        #x = self.Fully_Connected2(x)
        
        x = self.TANH(x)
        
        return x