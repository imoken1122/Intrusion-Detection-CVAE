import torch as th
from torch import nn
from torch.autograd import Variable as V
from torch.nn import functional as F

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(116,500)
        self.hidden = nn.Linear(500,500)
        self.mu = nn.Linear(500,25)
        self.sigma = nn.Linear(500,25)
        
        self.fc2 = nn.Linear(25,495)
        self.fc3 = nn.Linear(500,116)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def encoder(self,x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.hidden(h))
        h = self.hidden(h)
        return self.mu(h),self.sigma(h)
    
    def revize_parameter(self,mu,logsigma):

        sigma = th.exp(0.5*logsigma)
        eps = V(th.randn(sigma.size()))
        return sigma.mul(eps) + mu

    
    def decoder(self,z,oh_label):
        h = self.relu(self.fc2(z))

        h = th.cat((h,oh_label),dim = 1)
        h = self.fc3(self.relu(self.hidden(h)))
        return self.sigmoid(h)
    
    def forward(self,x,label):
        mu,sigma = self.encoder(x)
        z = self.revize_parameter(mu,sigma)
        output = self.decoder(z,label)
        return output,mu,sigma