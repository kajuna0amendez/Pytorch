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

import torch
import torch.nn as nn

class MultiLayerPerceptronOptim(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, alpha):
        """
        Initalization
        """
        # Init the above class 
        super(MultiLayerPerceptronOptim, self).__init__()
        
        # Get Cuda device 
        cuda = torch.device('cuda')

        self.alpha = torch.tensor(alpha, dtype=torch.float, device = cuda)
        self.one = torch.tensor(1.0, dtype=torch.float, device = cuda)
        
        # Dimensions for input, hidden and output
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
                
        # Layers - using randn for the -1 and 1 selection
        torch.cuda.seed_all()
                
        self.WIH = torch.nn.Parameter(torch.randn(self.input_dim, self.hidden_dim, device=cuda,\
                                       requires_grad=True))
        self.WHO = torch.nn.Parameter(torch.randn(self.hidden_dim, self.output_dim, device=cuda,\
                                       requires_grad=True))


    def sigmoid(self, x):
        
        return self.one / (self.one + torch.exp(-self.alpha*x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (self.one - self.sigmoid(x))
    
    def forward(self, X):
        """
        Forward pass
        """
        # Split first layer liner
        self.Y1 = torch.matmul(X, self.WIH)
        
        # Apply the activation function
        self.Y2 = self.sigmoid(self.Y1)
        
        # Hidden Layer linear function
        self.Y3 = torch.matmul(self.Y2, self.WHO)
    
        # Last activation
        Y4 = self.sigmoid(self.Y3)
            
        return Y4
    
    def loss(self, Y_hat, Y):
        """
        The loss being used in the function
        """
        return torch.mean(torch.pow(Y_hat-Y, 2)) 
        
        
    def backward(self, X, Y, Y4):
        """
        Backward Pass
        """  
        # Gradient for Output to Hidden Backpropagation
        self.dL_dy4 = (Y4 - Y)
        
        self.dy4_dy3  = self.dsigmoid(self.Y3)
        self.dy3_dWHO = self.Y2
        
        # Delta
        self.y4_delta_k = (self.dL_dy4 * self.dy4_dy3)

        # The parts for the Gradient of w2: delta dy3_dWHO
        dL_dWHO = torch.matmul(torch.t(self.dy3_dWHO), self.y4_delta_k)


        # Gradient for Hidden to Input Backpropagation
        self.dy3_dy2 = self.WHO
        self.dy2_dy1 = self.dsigmoid(self.Y1)

        # Y2 delta: (dL_dy4 dy4_dy3) dy3_dy2 dy2_dy1
        self.y2_delta_j = (torch.matmul(self.y4_delta_k,\
                           torch.t(self.dy3_dy2)) * self.dy2_dy1)

        # Gradients for w1: (dC_dy4 dy4_dy3) dy3_dy2 dy2_dy1 dy1_dw1
        dL_dWIH = torch.matmul(torch.t(X), self.y2_delta_j)

        # Change the grad data
        self.WIH.grad = dL_dWIH
        self.WHO.grad = dL_dWHO
            

        
        

    