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

import sys
import os
from MultiLayerPerceptronOpt import MultiLayerPerceptronOptim
import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from sklearn import preprocessing
import torch.optim as optim
from torch.utils import data 
import torch
import time

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from tools.data_processing import Data


if __name__ == '__main__':

    # Laod parameters for the network
    input_dim = 784
    hidden_dim = 1500
    output_dim = 1
    learning_rate = 0.001
    alpha = 0.1
    nfolds = 10
    num_epochs = 10
    
    # Load data
    X, Y = loadlocal_mnist(images_path='../data/train-images.idx3-ubyte',\
                           labels_path='../data/train-labels.idx1-ubyte')

    X01 = X[np.logical_or(Y==0 ,  Y==1)]
    Y01 = Y[np.logical_or(Y==0 ,  Y==1)]
    
    if torch.cuda.is_available():

        # Empty cached memory in CUDA device        
        torch.cuda.empty_cache()      
        
        # Moving parameters to the GPU accelerate    
        cuda = torch.device('cuda')          
        one_gpu = torch.tensor(1.0, dtype=torch.float, device = cuda)
        
        # Cross Validation Data
        cv = StratifiedKFold(n_splits=nfolds, shuffle = True)
        tprs = list()
        aucs = list()
        mean_fpr = np.linspace(0, 1, 100)
        
        time1 = time.clock()
        Fold = 1
        plt.figure()
        for train, test in cv.split(X01, Y01):
            print('Fold being used: ', Fold)
            
            # Instanciate model
            model = MultiLayerPerceptronOptim(input_dim, hidden_dim, output_dim, alpha)
            
            # Call the optimization for the model
            optimizer = optim.Adam(model.parameters(), lr = learning_rate)
            
            # Using a extended class for our own data processing into min_batch 
            # for ADAM  
            DataSamples = Data(X01[train], Y01[train])
            
            dataload = data.DataLoader(DataSamples, batch_size = 30,\
                                       shuffle = True, num_workers = 4)
            
            for epoch in range(num_epochs):
                for i_b, (x_minb, y_minb) in enumerate(dataload):
                    x_minb_cuda = x_minb.to('cuda', dtype=torch.float, non_blocking=True)
                    y_minb_cuda = y_minb.to('cuda', dtype=torch.float, non_blocking=True)
                    # Clean grads
                    optimizer.zero_grad()
                    # Get our predictions
                    Y_hat = model(x_minb_cuda)
                    loss = model.loss(Y_hat, y_minb_cuda)
                          
                    # Print our mean square loss
                    if epoch % 2 == 0 and i_b == 20:
                        #print('Size {}'.format(x_minb_cuda.shape[0]))
                        print('Epoch {} | Mean Loss: {}'.format(epoch, loss))
                    
                    # Train
                    model.backward(x_minb_cuda, y_minb_cuda, Y_hat)
                    # Update parameters using ADAM
                    optimizer.step()
            print("Estimation")
            X = torch.tensor(preprocessing.normalize(X01[test]),\
                             dtype=torch.float, device = cuda)        
            estimation = model(X)
            
            # Plotting the roc curve
            fpr, tpr, _ = roc_curve(Y01[test], estimation.to('cpu').detach().numpy())
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            # Get the AUC
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,\
                             label='ROC fold %d (AUC = %0.2f)' % (Fold, roc_auc))
            Fold += 1
        time2 = time.clock()
        print('Time GPU: ', time2-time1)
        
        # Final mean ROC
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper,\
                         color='grey', alpha=.2,\
                         label=r'$\pm$ 1 std. dev.')
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


