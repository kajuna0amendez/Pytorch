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
from sklearn import preprocessing
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, X, Y):
        cuda = torch.device('cuda')
    
        self.x = preprocessing.normalize(X)
                 #torch.tensor(preprocessing.normalize(X),\
                 #             dtype=torch.float, device = cuda)
        self.y = Y.reshape(-1, 1)
                 #torch.tensor(Y.reshape(-1, 1), dtype=torch.float,\
                 #             device = cuda)
        self.len=self.x.shape[0]
    def __getitem__(self,index):    
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len