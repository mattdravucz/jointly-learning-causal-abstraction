import torch
import torch.nn as nn
    
class JSD_loss(nn.Module):
    def __init__(self):
        super(JSD_loss, self).__init__()
    
    def kl(self,p, q): 
        return torch.sum(p*torch.log((p/q)+1e-7),0)
    
    def jsd(self,p,q,W):
        m = (1./2.)*(p + q)
        return torch.max(W*torch.sqrt((1./2.)*self.kl(p,m)+(1./2.)*self.kl(q,m)))
    
    def forward(self, ps,qs, Ws=None):
        if not Ws:
            Ws = torch.ones(len(ps))
        
        losses_commutativity = 0.
        for p,q,W in zip(ps,qs,Ws):
            losses_commutativity += self.jsd(p.T,q.T,W)
        return losses_commutativity

    
class RowMax_penalty(nn.Module):
    def __init__(self):
        super(RowMax_penalty, self).__init__()
    
    def rowmax(self,W):
        return torch.sum(torch.ones(W.shape[0]) - torch.max(W.T,dim=0).values)
     
    def forward(self,Ws):
        losses_penalty = 0.
        for W in Ws:
            losses_penalty += self.rowmax(W)
        
        return losses_penalty