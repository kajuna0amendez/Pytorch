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

class MultiLayerPerceptron_gpu(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate, alpha):
        """
        Initalization
        """
        # Init the above class 
        super().__init__()
        
        # Get Cuda device 
        cuda = torch.device('cuda')

        self.alpha = torch.tensor(alpha, dtype=torch.float, device = cuda)
        self.one = torch.tensor(1.0, dtype=torch.float, device = cuda)
        
        # Dimensions for input, hidden and output
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Learning Rate
        self.learning_rate = torch.tensor(learning_rate, dtype=torch.float, device=cuda)
        
        # Layers - using randn for the -1 and 1 selection
        self.WIH = torch.randn(self.input_dim, self.hidden_dim, device=cuda)
        self.WHO = torch.randn(self.hidden_dim, self.output_dim, device=cuda)

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
        self.dL_dWHO = torch.matmul(torch.t(self.dy3_dWHO), self.y4_delta_k)


        # Gradient for Hidden to Input Backpropagation
        self.dy3_dy2 = self.WHO
        self.dy2_dy1 = self.dsigmoid(self.Y1)

        # Y2 delta: (dL_dy4 dy4_dy3) dy3_dy2 dy2_dy1
        self.y2_delta_j = (torch.matmul(self.y4_delta_k,\
                           torch.t(self.dy3_dy2)) * self.dy2_dy1)

        # Gradients for w1: (dC_dy4 dy4_dy3) dy3_dy2 dy2_dy1 dy1_dw1
        self.dL_dWIH = torch.matmul(torch.t(X), self.y2_delta_j)

        # Gradient descent on the weights from our 2 linear layers
        self.WIH -= (self.learning_rate * self.dL_dWIH)
        self.WHO -= (self.learning_rate * self.dL_dWHO)
            

    def train(self, X, Y):
        # Forward propagation
        output = self.forward(X)

        # Backward propagation and gradient descent
        self.backward(X, Y, output)
        
        
class MultiLayerPerceptron_cpu(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate, alpha):
        """
        Initalization
        """
        # Init the above class 
        super().__init__()
        
        # Get Cuda device 
        cuda = torch.device('cuda')
        
        self.alpha = torch.tensor(alpha, dtype=torch.float)
        self.one = torch.tensor(1.0, dtype=torch.float)
        
        # Dimensions for input, hidden and output
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Learning Rate
        self.learning_rate = torch.tensor(learning_rate, dtype=torch.float)
        
        # Layers - using randn for the -1 and 1 selection
        self.WIH = torch.randn(self.input_dim, self.hidden_dim)
        self.WHO = torch.randn(self.hidden_dim, self.output_dim)

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
        
    def backward(self, X, Y, Y4):
        """
        Backward Pass
        """
        # Given the regression 1/2(o-t)^2
        
        # Gradient for Output to Hidden Backpropagation
        self.dL_dy4 = Y4 - Y
        
        self.dy4_dy3  = (self.dsigmoid(self.Y3))
        self.dy3_dWHO = (self.Y2)
        
        # Delta
        self.y4_delta_k = (self.dL_dy4 * self.dy4_dy3)

        # The parts for the Gradient of w2: delta dy3_dWHO
        self.dL_dWHO = torch.matmul(torch.t(self.dy3_dWHO), self.y4_delta_k)


        # Gradient for Hidden to Input Backpropagation
        self.dy3_dy2 = (self.WHO)
        self.dy2_dy1 = (self.dsigmoid(self.Y1))

        # Y2 delta: (dL_dy4 dy4_dy3) dy3_dy2 dy2_dy1
        self.y2_delta_j = (torch.matmul(self.y4_delta_k, torch.t(self.dy3_dy2)) * self.dy2_dy1)

        # Gradients for w1: (dC_dy4 dy4_dy3) dy3_dy2 dy2_dy1 dy1_dw1
        self.dL_dWIH = (torch.matmul(torch.t(X), self.y2_delta_j))

        # Gradient descent on the weights from our 2 linear layers
        self.WIH -= (self.learning_rate * self.dL_dWIH)
        self.WHO -= (self.learning_rate * self.dL_dWHO)
        

    def train(self, X, Y):
        # Forward propagation
        output = self.forward(X)

        # Backward propagation and gradient descent
        self.backward(X, Y, output)        
    
    