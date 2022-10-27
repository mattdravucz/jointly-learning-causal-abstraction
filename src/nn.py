
import src.utils as ut
import src.evaluationsets as es

from src.nn_layers import BinaryLinear
from src.SCMMappings_1_1 import Abstraction

import torch
import torch.nn as nn


class JointNeuralNet(nn.Module):
    def __init__(self, M0, M1, R, a, J, T=0.1, frozen_alphas={}, initialised_alphas={}):
        super(JointNeuralNet, self).__init__()
        
        self.M0 = M0
        self.M1 = M1
        self.R = R
        self.a = a
        self.A = Abstraction(self.M0,self.M1,self.R,self.a)
        self.frozen_alphas = frozen_alphas
        self.initialised_alphas = initialised_alphas
        
        self.T = T
        
        self.M1_diagrams = J
        self.M0_diagrams = [[ut.inverse_fx(self.a,d[0]), ut.inverse_fx(self.a,d[1])] for d in self.M1_diagrams]

        self.lower_paths = []
        self.higher_paths = []

        self.alphas_to_learn = []
        for i in range(len(self.M0_diagrams)):
            left_alpha = self.M1_diagrams[i][0]
            right_alpha = self.M1_diagrams[i][1]
            self.alphas_to_learn = self.alphas_to_learn + left_alpha + right_alpha

            low_path = (left_alpha, torch.tensor(self.A.compute_mechanisms(M1,self.M1_diagrams[i][0],self.M1_diagrams[i][1]),dtype=torch.float32))
            high_path = (torch.tensor(self.A.compute_mechanisms(M0,self.M0_diagrams[i][0],self.M0_diagrams[i][1]),dtype=torch.float32),right_alpha)
            self.lower_paths.append(low_path)
            self.higher_paths.append(high_path)
        
        self.alphas_to_learn = list(set(self.alphas_to_learn) - set(frozen_alphas.keys()))
        self.alpha_dims = {k: self.A.get_cardinalities_alpha(k) for k in self.alphas_to_learn}
        self.alpha_index = {}
        
        self.layers = nn.ModuleList()
        for key, value in self.alpha_dims.items():
            if key in self.initialised_alphas:
                self.layers.append(BinaryLinear(value[1], value[0], self.T, self.initialised_alphas[key]))
            else:
                self.layers.append(BinaryLinear(value[1], value[0], self.T))
            self.alpha_index[key] = len(self.layers)-1    
    
    def get_Wmatrix_by_name(self,alphakey):
        return self.layers[self.alpha_index[alphakey]].get_binarized_weights()
    
    def flat_tensor_product(self,x,y):
        """
        pytorch implementation of src.utils.flat_tensor_product()
        """
        tensor = torch.einsum('ij,kl->ikjl',x,y)
        return tensor.reshape((tensor.shape[0]*tensor.shape[1],tensor.shape[2]*tensor.shape[3]))

    def tensorize_list(self,tensor,l):
        """
        pytorch implementation of src.utils.tensorize_list()
        """ 
        if tensor is None:
            if len(l)>1:
                tensor = self.flat_tensor_product(l[0],l[1])
                return self.tensorize_list(tensor,l[2:])
            else:
                return l[0]
        else:
            if len(l)>0:
                tensor = self.flat_tensor_product(tensor,l[0])
                return self.tensorize_list(tensor,l[1:])
            else:
                return tensor    
    
    def forward(self):        
        lower_distrib = []
        for path in self.lower_paths:           
            low_mechanism = path[1]            
            left_alpha_matrices = []
            for name in path[0]:
                if name in self.alphas_to_learn:
                    left_alpha_matrices.append( self.get_Wmatrix_by_name(name) )
                else:
                    left_alpha_matrices.append( self.frozen_alphas[name] )
            left_abstraction = self.tensorize_list(None,left_alpha_matrices)
            low_path = torch.matmul(low_mechanism, left_abstraction)
            lower_distrib.append(low_path)
        
        higher_distrib = []
        for path in self.higher_paths:
            right_alpha_matrices = []
            for name in path[1]:
                if name in self.alphas_to_learn:
                    right_alpha_matrices.append( self.get_Wmatrix_by_name(name) )
                else:
                    right_alpha_matrices.append( self.frozen_alphas[name] )
            right_abstraction = self.tensorize_list(None,right_alpha_matrices)
            high_mechanism = path[0]
            
            high_path = torch.matmul(right_abstraction, high_mechanism)
            higher_distrib.append(high_path)

        Ws = [layer.get_binarized_weights() for layer in self.layers]
        return lower_distrib, higher_distrib, Ws
    
    
class IndepNeuralNets():
    def __init__(self, M0, M1, R, a, J, T=0.1):
        self.nns = []
        for j in J:
            self.nns.append(JointNeuralNet(M0,M1,R,a,T=0.1,J=[j]))
            
    
    