"""
Adaptive Asynchronous Temporal Fields Base model
"""
import torch.nn as nn
import torch
from torch.autograd import Variable

class BasicModule(nn.Module):
    def __init__(self, inDim, outDim, hidden_dim = 1000, dp_rate = 0.3):
        super(BasicModule, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(inDim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = dp_rate),
            nn.Linear(hidden_dim, outDim)
            )
        
    def forward(self, x):
        return self.layers(x)
        

class AsyncTFBase(nn.Module):
    def __init__(self, dim, s_classes, o_classes, v_classes, _BaseModule = BasicModule):
        super(AsyncTFBase, self).__init__()
        self.s_classes = s_classes
        self.o_classes = o_classes
        self.v_classes = v_classes
        
        self.num_low_rank = 5

        self.s = nn.Sequential(
            nn.Linear(dim, 1000),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(1000, self.s_classes)
            )
        self.o = nn.Linear(dim, self.o_classes)
        self.v = nn.Linear(dim, self.v_classes)
        
        
    def forward(self, rgb_feat):
        s = self.s(rgb_feat)
        o = self.o(rgb_feat)
        v = self.v(rgb_feat)
        
        return s, o, v