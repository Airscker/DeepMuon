

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class DetNet(nn.Module):
    def __init__(self):
        super(DetNet,self).__init__()
        self.bottleneck=nn.Sequential(
            nn.Conv3d(3,3,kernel_size=3,stride=1,padding=2,dilation=2,bias=False),
            nn.BatchNorm3d(3),
            nn.ReLU(inplace=True),
            nn.Conv3d(3,3,1,bias=False),
            nn.BatchNorm3d(3)
        )
        self.flatten=nn.Flatten()
        self.lin=nn.Sequential(
            nn.Linear(12000,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,3)
        )
    def forward(self,x):
        x=self.bottleneck(x)
        x=self.flatten(x)
        x=self.lin(x)
        return F.normalize(x)

