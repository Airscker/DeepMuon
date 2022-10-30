import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math

class SPP_NET(nn.Module):

    def __init__(self):
        super(SPP_NET, self).__init__()

        self.output_num=[3,2,1]
        
        self.conv=nn.Sequential(
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
        spp=spatial_pyramid_pool(x,1,[int(x.size(2)),int(x.size(3)),int(x.size(4))],self.output_num)
        x=self.linear_relu_stack(x)
        return 100*F.normalize(x)

def spatial_pyramid_pool(self,previous_conv,num_sample,previous_conv_size,out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    for i in range(len(out_pool_size)):
        d_wid = int(math.ceil(previous_conv_size[0]/out_pool_size[i]))
        h_wid = int(math.ceil(previous_conv_size[1]/out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[2]/out_pool_size[i]))
        d_pad = int(math.floor((d_wid*out_pool_size[i]-previous_conv_size[0]+1)/2))
        h_pad = int(math.floor((h_wid*out_pool_size[i]-previous_conv_size[1]+1)/2))
        w_pad = int(math.floor((w_wid*out_pool_size[i]-previous_conv_size[2]+1)/2))
        maxpool=nn.MaxPool3d((d_wid,h_wid,w_wid),stride=(d_wid,h_wid,w_wid),padding=(d_pad,h_pad,w_pad))
        x=maxpool(previous_conv)
        if(i==0):
            spp=x.view(num_sample,-1)
        else:
            spp=torch.cat((spp,x.view(num_sample,-1)),1)
    return spp