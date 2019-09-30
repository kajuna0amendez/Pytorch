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
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
import torch
import time

if __name__ == '__main__':

    # Laod parameters for the network
    input_dim = 784
    hidden_dim = 1500
    output_dim = 1
    learning_rate = 0.00001
    alpha = 0.1
    nfolds = 10
    num_epochs_cpu = 10
    num_epochs_gpu = 3000
    
    # Load data
    X, Y = loadlocal_mnist( images_path='data/train-images.idx3-ubyte', labels_path='data/train-labels.idx1-ubyte')

    X01 = X[np.logical_or(Y==0 ,  Y==1)]
    Y01 = Y[np.logical_or(Y==0 ,  Y==1)]
    
#    # Run CPU first
#    X = torch.tensor(preprocessing.normalize(X01), dtype=torch.float)
#    Y = torch.tensor(Y01.reshape(-1, 1), dtype=torch.float)
#        
#    model_cpu = MultiLayerPerceptron_cpu(input_dim, hidden_dim, output_dim, learning_rate, alpha)
#    
#    time1 = time.clock()
#    for epoch in range(num_epochs):
#        # Get our predictions
#        Y_hat = model_cpu(X)
#        
#        loss = torch.mean((Y_hat-Y).pow(2)).detach().item()
#            
#        # Print our mean cross entropy loss
#        if epoch % 2 == 0:
#            print('Epoch {} | Mean Loss: {}'.format(epoch, loss))
#        
#        # Train
#        model_cpu.train(X, Y)
#    time2 = time.clock()
#    print('Time CPU: ', time2-time1)
    
    
    #GPU second
    if torch.cuda.is_available():
        
        torch.cuda.empty_cache()
    
        cuda = torch.device('cuda')
    
        X = torch.tensor(preprocessing.normalize(X01),\
                         dtype=torch.float, device = cuda)
        Y = torch.tensor(Y01.reshape(-1, 1), dtype=torch.float,\
                         device = cuda)
                     
        one_gpu = torch.tensor(1.0, dtype=torch.float, device = cuda)
        
        cv = StratifiedKFold(n_splits=nfolds, shuffle = True)
        tprs = list()
        aucs = list()
        mean_fpr = np.linspace(0, 1, 100)
        
        
        time1 = time.clock()
        Fold = 1
        plt.figure()
        for train, test in cv.split(X01, Y01):
            print('Fold being used: ', Fold)
            model_gpu = MultiLayerPerceptron_gpu(input_dim, hidden_dim, output_dim, learning_rate, alpha)
            for epoch in range(num_epochs_gpu):
                # Get our predictions
                Y_hat = model_gpu(X[train])
                    
                loss = torch.mean((Y_hat-Y[train]).pow(2))
                        
                # Print our mean square loss
                if epoch % 100 == 0:
                    print('Epoch {} | Mean Loss: {}'.format(epoch, loss))
                
                # Train
                model_gpu.train(X[train], Y[train])
            estimation = model_gpu(X[test])
            fpr, tpr, _ = roc_curve(Y01[test], estimation.to('cpu'))
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,\
                             label='ROC fold %d (AUC = %0.2f)' % (Fold, roc_auc))
            Fold += 1
        time2 = time.clock()
        print('Time GPU: ', time2-time1)
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


        
#        plt.figure()
#        fpr, tpr, _ = roc_curve(Y01, Y_hat.to('cpu'))
#        plt.plot(fpr, tpr, lw=3, alpha=0.9)
#        
#        plt.xlim([-0.05, 1.05])
#        plt.ylim([-0.05, 1.05])
#        plt.xlabel('False Positive Rate')
#        plt.ylabel('True Positive Rate')
#        plt.show()