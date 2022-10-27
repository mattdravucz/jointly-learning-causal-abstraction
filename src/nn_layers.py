import torch
import torch.nn as nn

class BinaryLinear(nn.Module):
    def __init__(self, in_dim, out_dim, T, initialised=None):
        super(BinaryLinear, self).__init__()
        if type(initialised) == type(None):
            self.weight = nn.Parameter(torch.randn((in_dim, out_dim)))
        else:
            self.weight = nn.Parameter(initialised)
        self.T = T
    
    def binarize_weights(self):
        return torch.exp(self.weight/self.T) / torch.unsqueeze(torch.sum(torch.exp(self.weight/self.T),axis=0),dim=0)
    
    def get_binarized_weights(self):
        return self.binarize_weights()
    
    def forward(self,x):
        return torch.matmul(x, self.binarize_weights().T)