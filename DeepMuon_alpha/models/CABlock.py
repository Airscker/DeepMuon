
#Coordinate Attention

import torch
from torch import nn
import torch.nn.functional as F
 
class ECANet(nn.Module):
    def __init__(self):
        super(ECANet,self).__init__()
        self.avgpool=nn.AdaptiveAvgPool3d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=3,padding=1,bias=False)
        self.sigmoid=nn.Sigmoid()
        
        self.pool=nn.AvgPool3d(kernel_size=(3,3,5),stride=1)

        self.flatten=nn.Flatten()
        self.linear=nn.Sequential(
            nn.Linear(3*8*8*36,2048),
            nn.ReLU(), 
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512,3)
        )
    def forward(self,x):
        y=self.avgpool(x)
        y=self.conv(y.squeeze(-1).squeeze(-1).transpose(-1,-2))
        y=y.transpose(-1,-2).unsqueeze(-1).unsqueeze(-1)
        y=self.sigmoid(y)
        x=x*y.expand_as(x)
        x=self.pool(x)
        x=self.flatten(x)
        x=self.linear(x)
        return F.normalize(x)



class CABlock(nn.Module):
    def __init__(self):
        super(CABlock,self).__init__()
 
        self.avg_pool_x=nn.AdaptiveAvgPool3d((10,1,1))
        self.avg_pool_y=nn.AdaptiveAvgPool3d((1,10,1))
        self.avg_pool_z=nn.AdaptiveAvgPool3d((1,1,40))

        self.conv=nn.Conv3d(3,1,kernel_size=1,stride=1,bias=False)
        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm3d(1)
        self.F=nn.Conv3d(1,3,kernel_size=1,stride=1,bias=False)
        self.sigmoid=nn.Sigmoid()

        self.flatten=nn.Flatten()
        self.linear=nn.Sequential(
            nn.Linear(3*10*10*40,4096),
            nn.ReLU(), 
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512,3)
        )
 
    def forward(self, x):
        #b,c,d,h,w
        x_d=self.avg_pool_x(x).permute(0,1,4,3,2)
        #b,c,1,1,d
        x_h=self.avg_pool_y(x).permute(0,1,2,4,3)
        #b,c,1,1,h
        x_w=self.avg_pool_z(x)
        #b,c,1,1,w
        x_cat_conv_relu=self.relu(self.conv(torch.cat((x_d,x_h,x_w),4)))
        #b,c,1,1,d+h+w
        x_d,x_h,x_w=x_cat_conv_relu.split([10,10,40],4)
        #b,c,1,1,h    b,c,1,1,w
        x_d=self.sigmoid(self.F(x_h.permute(0,1,4,3,2)))
        x_h=self.sigmoid(self.F(x_h.permute(0,1,2,4,3)))
        x_w=self.sigmoid(self.F(x_w))
        x=x*x_d.expand_as(x)*x_h.expand_as(x)*x_w.expand_as(x)

        x=self.flatten(x)
        x=self.linear(x)
        return F.normalize(x)
