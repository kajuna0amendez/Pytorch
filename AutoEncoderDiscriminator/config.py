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



# Root directory for dataset
dataroot = #low resolution path images
#dataroot = #high resolution

# Number of workers for dataloader
workers = 4

# Batch size during training
batch_size = 32

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128#256 #256

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Kernel Size
k_size = 3

# Padding 
k_padding = 1
d_padding = 1

# Number of training epochs
num_epochs = 4

# Learning rate for optimizers
lr = 0.0001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.90
beta2 = 0.999

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Reduction pooling
p_size = 2

# Noise 
lower = 16
upper = 45

# Flag to plot images
plot_some_images = False
plot_machines = False
plot_results = True

# Set random seed for reproducibility
manualSeed = 999

# We have the mixing factor
alpha = 1.0
beta  = 1.0

# Which Activation Function
activation = 'leaky_relu'