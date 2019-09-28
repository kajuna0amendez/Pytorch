
# -*- coding: utf-8 -*-
#!/usr/bin/env python
__author__ = 'Andres Mendez-Vazquez'
__copyright__ = 'Copyright 2019'
__credits__ = ['Andres Mendez-Vazquez', 'Ana Aguilar']
__license__ = 'Closed'
__version__ = '1.0.0'
__maintainer__ = 'Andres Mendez-Vazquez'
__email__ = 'andresm@amcoonline.net'
__status__ = 'Development'


import numpy as np
from scipy.special import expit 

def sigmoid(eval):
    return expit(eval)




def Neural_Training(Y01, Labels01 , eta , Epochs):
    """
    6 year Old code for a Neural Network 
    """
    d,samplenumb = Y01.shape

    # Random [-1,1] init from Haykin  
    WIH = 2*np.mat(np.random.rand(2*d,d)) -1.0
    WHO = 2*np.mat(np.random.rand(1,2*d)) -1.0 
    difft =  Labels01.astype(np.float64) 
    
    for i in range(1,Epochs):
        
        #Get the input to the output layer
        y_j_temp = sigmoid(WIH*Y01)
        netk = WHO*y_j_temp
        zk = sigmoid(netk)
        
        # Creating Delta Wk
        diff1 = difft - zk
        tDeltaWk = eta*np.multiply(diff1,np.multiply(sigmoid(netk),1.0-sigmoid(netk)))
        tDeltaWk = np.tile(tDeltaWk,(2*d,1))
        DeltaWk = np.multiply( y_j_temp,tDeltaWk)
        DeltaWk = np.transpose(np.sum(DeltaWk,1))
        
        # New Weights
        WHO = WHO + DeltaWk
        
        #Creating Delta Wj        
        dnetj = np.multiply(y_j_temp,1.0-y_j_temp)
        tprodsumk = np.multiply(np.transpose(DeltaWk),np.transpose(WHO))
        tprodsumk = np.tile(tprodsumk, (1,samplenumb) )
        tprodsumk = eta*np.multiply(tprodsumk,dnetj)
        DeltaWj = tprodsumk * np.transpose(Y01)
        
        # New Weights
        WIH = WIH + DeltaWj
            
        
    return WIH, WHO 

if __name__ == '__main__':
    # Number of samples
    N= 60000
    
    #Number of Epochs
    Epochs = 20
    
    #Learning Rate
    eta = 0.001
    
    # opening images for [r]eading as [b]inary
    in_file = open("train-images.idx3-ubyte", "rb") 
    in_file.read(16)
    Data = in_file.read()
    in_file.close()
    
    # Transform the data stream
    X = np.fromstring(Data, dtype=np.uint8) 
    X = X.astype(np.float64)
    X = np.mat(X)
    X = X.reshape(N,784)
    X = np.transpose(X)
    
    # Now the labels:
    in_file = open("train-labels.idx1-ubyte", "rb") 
    in_file.read(8)
    DLabel = in_file.read()
    in_file.close()
    
    # Transform the label
    Labels = np.fromstring(DLabel, dtype=np.uint8)
    
    Labels01 =  Labels[ np.logical_or(Labels[:]==0 , Labels[:]==1) ] 
    
    #Extract Data 01
    X01 = X[:, Labels[ np.logical_or(Labels[:]==0 , Labels[:]==1) ] ]
    dummy, N1 =  X01.shape
    
    #Mean Creation
    Xmean= X01.sum(axis=1)
    Xmean = (1.0/(N1*1.0))*Xmean
    # Thus we create data with zero mean 
    X01= X01-Xmean
    
    #Covariance
    C_X = (1.0/(N1*1.0-1.0))*(X01*np.transpose(X01))
    
    D, E = np.linalg.eigh(C_X)
    idx = D.argsort()[::-1]   
    D = D[idx]
    E = E[:,idx]
    
    P =  np.transpose(E[:,0:100])
    
    Y01 = P*X01
    
    # Run the Training
    WIH, WHO =  Neural_Training(Y01, Labels01 , eta, Epochs)
    
    # Create the results
    Results = sigmoid(WHO*sigmoid(WIH*Y01))
    
    # Find the decisions
    d,samplenumb = Y01.shape
    index = np.ones(samplenumb)
    Results = np.asarray(Results).reshape(-1)
    
    # Generate the Confusion Matrix using a hard threshold half of minimum and maximum
    
    t1 = np.max(Results[Labels01[:]==0])
    t2 = np.max(Results[Labels01[:]==1])
    
    tr = (t2-t1)/2.0
    
    # print threshold
    
    print(tr)
    
    R11 = np.sum(index[ np.logical_and(Results[:]<tr, Labels01[:]==0 )])
    R22 = np.sum(index[ np.logical_and(Results[:]>tr, Labels01[:]==1 )])
    R12 = np.sum(index[ np.logical_and(Results[:]>=tr, Labels01[:]==0 ) ])
    R21 = np.sum(index[ np.logical_and(Results[:]<=tr, Labels01[:]==1 ) ])
    
    ConfusionMatrix =  np.matrix([[R11 ,R12] , [R21 , R22]])
    
    # Print the results
    print("Confusion Matrix \n", ConfusionMatrix)
    
    # Print some Labels
    for i in range(0,15):
        print("Gen Label {} = {}  Real Label {}".format(i , round(Results[i],2), Labels01[i]))
    
        
