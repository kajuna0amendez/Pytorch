# -*- coding: utf-8 -*-
#!/usr/bin/env python
__author__ = "Andres Mendez-Vazquez"
__copyright__ = "Copyright 2018"
__credits__ = ["Andres Mendez-Vazquez"]
__license__ = "Closed"
__version__ = "v1.0.0"
__maintainer__ = "Andres Mendez-Vazquez"
__email =  "kajuna0kajuna@gmail.com"
__status__ = "Development"

from sklearn import preprocessing
from MultiLayerPerceptron import MultiLayerPerceptron_gpu,\
                                 MultiLayerPerceptron_cpu
import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import torch
import time

if __name__ == '__main__':

    # Laod parameters for the network
    input_dim = 784
    hidden_dim = 1500
    output_dim = 1
    learning_rate = 0.0001
    alpha = 1.0
    num_epochs = 10 #2000
    
    # Load data
    X, Y = loadlocal_mnist( images_path='train-images.idx3-ubyte', labels_path='train-labels.idx1-ubyte')

    X01 = X[np.logical_or(Y==0 ,  Y==1)]
    Y01 = Y[np.logical_or(Y==0 ,  Y==1)]
    
    # Run CPU first
    X = torch.tensor(preprocessing.normalize(X01), dtype=torch.float)
    Y = torch.tensor(Y01.reshape(-1, 1), dtype=torch.float)
        
    model_cpu = MultiLayerPerceptron_cpu(input_dim, hidden_dim, output_dim, learning_rate, alpha)
    
    time1 = time.clock()
    for epoch in range(num_epochs):
        # Get our predictions
        Y_hat = model_cpu(X)
        
        loss = torch.mean((Y_hat-Y).pow(2)).detach().item()
            
        # Print our mean cross entropy loss
        if epoch % 2 == 0:
            print('Epoch {} | Mean Loss: {}'.format(epoch, loss))
        
        # Train
        model_cpu.train(X, Y)
    time2 = time.clock()
    print('Time CPU: ', time2-time1)
    
    num_epochs = 1000
    #GPU second
    if torch.cuda.is_available():
        
        torch.cuda.empty_cache()
    
        cuda = torch.device('cuda')
    
        X = torch.tensor(preprocessing.normalize(X01),\
                         dtype=torch.float, device = cuda)
        Y = torch.tensor(Y01.reshape(-1, 1), dtype=torch.float,\
                         device = cuda)
                     
        one_gpu = torch.tensor(1.0, dtype=torch.float, device = cuda)
        
        model_gpu = MultiLayerPerceptron_gpu(input_dim, hidden_dim, output_dim, learning_rate, alpha)
        
        time1 = time.clock()
        for epoch in range(num_epochs):
            # Get our predictions
            Y_hat = model_gpu(X)
                
            loss = torch.mean((Y_hat-Y).pow(2))#.detach().item()
                    
            # Print our mean square loss
            if epoch % 200 == 0:
                print('Epoch {} | Mean Loss: {}'.format(epoch, loss))
            
            # Train
            model_gpu.train(X, Y)
            
        time2 = time.clock()
        print('Time GPU: ', time2-time1)
        
        plt.figure()
        fpr, tpr, _ = roc_curve(Y01, Y_hat.to('cpu'))
        plt.plot(fpr, tpr, lw=3, alpha=0.9)
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()