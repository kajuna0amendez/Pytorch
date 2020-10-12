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

import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from config import ngpu, image_size, nc, batch_size, workers, dataroot, lower, upper
import torch

class Center_Random_Noise(object):
    """
    Generate the random center noise

    """

    def __init__(self, lower, upper):
        
        self.lower = lower
        self.upper = upper
        
    def __call__(self, sample):
                
        
        temp = torch.from_numpy(np.random.rand(3,self.upper-\
                                                 self.lower+1,
                                                 self.upper-\
                                                 self.lower+1))
        #rvar = np.random.randint(0, image_size-(upper)-1, size = 1)
        rcent = np.random.randint(0, image_size//8, size = 1)
    
        rsample = sample.clone()
    
        centerx = image_size//2 + rcent[0]
        
        a = centerx - (self.upper-self.lower+1)//2
        b = centerx + (self.upper-self.lower+1)//2
    

        rsample[:, a:b, a:b ] = temp
        
        tranrez = transforms.Resize(image_size)
        
        windowr = transforms.ToTensor()(tranrez(transforms.ToPILImage()(rsample[:, a:b, a:b ]))) #.clone()
        
        return [sample, rsample, windowr, (a,b)]                                                      

class Normalization_three_images(object):
    """
    Generate normalization
    """

    def __init__(self, mean, std):
        
        self.mean = mean
        self.std = std    

    def __call__(self, sample):
        
        s1 = transforms.Normalize(self.mean, self.std)(sample[0])
        s2 = transforms.Normalize(self.mean, self.std)(sample[1])
        s3 = transforms.Normalize(self.mean, self.std)(sample[2])
        
        return [s1, s2, s3, sample[3]]

def data_loader():

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset real
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   Center_Random_Noise(lower, upper),
                                   Normalization_three_images((0.5, 0.5, 0.5),\
                                                        (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    return dataset, dataloader, device


def data_secondary_gen(image_gen, a, b):
    """
    Extract Sections 
    """
    nimg = image_gen.size(0) 
            
    new_image = torch.zeros(batch_size, nc, image_size, image_size)
    
    tranrez = transforms.Resize(image_size)
    
    for i in range(nimg):
        new_image[i,:,:,:] = transforms.ToTensor()(tranrez(transforms.ToPILImage()(image_gen[i,:,a[i]:b[i], a[i]:b[i]])))
        
    return new_image