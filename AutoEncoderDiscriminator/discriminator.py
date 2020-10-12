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

from torch import nn
from config import nc, ndf, k_size, k_padding, p_size

class Discriminator(nn.Module):
    def __init__(self, ngpu, activation):
        
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.activation = activation
        self.encoder1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.BatchNorm2d(nc),
            nn.Conv2d(in_channels = nc, out_channels = ndf, stride = 2,\
                      kernel_size=k_size, padding=k_padding),
            nn.Conv2d(in_channels = ndf, out_channels = ndf, stride = 1,\
                      kernel_size=k_size, padding=k_padding),
            nn.LeakyReLU(0.2, inplace=True)
            )
            #nn.MaxPool2d(p_size),
            
            # state size. (ndf) x 32 x 32
        self.encoder2 = nn.Sequential(
            nn.BatchNorm2d(ndf),
            nn.Conv2d(in_channels = ndf, out_channels = ndf*2, stride = 2,\
                      kernel_size=k_size, padding=k_padding),
            nn.Conv2d(in_channels = ndf*2, out_channels = ndf*2, stride = 1,\
                      kernel_size=k_size, padding=k_padding),
            self.activation_func()
            )
            #n.MaxPool2d(p_size),
            
            # state size. (ndf*2) x 16 x 16
        self.encoder3 = nn.Sequential(
            nn.BatchNorm2d(ndf*2),
            nn.Conv2d(in_channels = ndf*2, out_channels = ndf*4, stride = 2,\
                      kernel_size=k_size, padding=k_padding),
            nn.Conv2d(in_channels = ndf*4, out_channels = ndf*4, stride = 1,\
                      kernel_size=k_size, padding=k_padding),
            self.activation_func()
            )
            #nn.MaxPool2d(2),
            # state size. (ndf*4) x 8 x 8
        self.encoder4 = nn.Sequential(
            nn.BatchNorm2d(ndf*4),
            nn.Conv2d(in_channels = ndf*4, out_channels = ndf*8, stride = 2,\
                      kernel_size=k_size, padding=k_padding),
            nn.Conv2d(in_channels = ndf*8, out_channels = ndf*8, stride = 1,\
                      kernel_size=k_size, padding=k_padding),
            self.activation_func()
            )
            #nn.MaxPool2d(2),
            
            #Here the output to a sigmoid function
            
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        
        x = self.encoder1(x)
        #print(x.size())
        x = self.encoder2(x) 
        #print(x.size())
        x = self.encoder3(x) 
        #print(x.size())
        x = self.encoder4(x)
        #print(x.size())
        x = self.sigmoid(x) 
        
        return x
    
    def activation_func(self):
        return  nn.ModuleDict([
            ['relu', nn.ReLU(inplace=True)],
            ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
            ['selu', nn.SELU(inplace=True)],
            ['none', nn.Identity()]
        ])[self.activation]  
