import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math

class SPP(nn.Module):

    def __init__(self):
        super(SPP, self).__init__()

        self.output_num=[3,2,1]
        
        self.conv=nn.Sequential(
            # nn.BatchNorm3d(3),
            nn.Conv3d(3,8,(4,4,5),1,1,bias=False),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
            nn.Conv3d(8,16,(4,4,5),1,1,bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16,32,(4,4,5),1,1,bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(1152,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,3)
        )

    def forward(self,x):
        x=self.conv(x)
        num,c,d,h,w=x.size()
        out_pool_size=self.output_num
        for i in range(len(out_pool_size)):
            d_wid = int(math.ceil(d/out_pool_size[i]))
            h_wid = int(math.ceil(h/out_pool_size[i]))
            w_wid = int(math.ceil(w/out_pool_size[i]))
            d_pad = int(math.floor((d_wid*out_pool_size[i]-d+1)/2))
            h_pad = int(math.floor((h_wid*out_pool_size[i]-h+1)/2))
            w_pad = int(math.floor((w_wid*out_pool_size[i]-w+1)/2))
            tensor=F.max_pool3d(x,(d_wid,h_wid,w_wid),stride=(d_wid,h_wid,w_wid),padding=(d_pad,h_pad,w_pad))
            if(i==0):
                spp=tensor.view(num,-1)
            else:
                spp=torch.cat((spp,tensor.view(num,-1)),1)
        x=spp
        x=self.linear_relu_stack(x)
        return 100*F.normalize(x)
