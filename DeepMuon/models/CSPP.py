'''
Author: airscker
Date: 2022-10-13 07:51:47
LastEditors: airscker
LastEditTime: 2022-12-03 23:41:01
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import ResidualUnit
torch.set_default_tensor_type(torch.DoubleTensor)

class Plane(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10*10, 52),
            nn.BatchNorm1d(52),
            nn.LeakyReLU(),
            nn.Linear(52, 2)
        )
        self.linear_relu_stack2= nn.Sequential(
            nn.Linear(3*40, 52),
            nn.BatchNorm1d(52),
            nn.LeakyReLU(),
            nn.Linear(52, 3),
            HailingDirectNorm()
        )
    def forward(self, x):
        for i in range(40):
            x_plane=x[:,:,:,:,i].to()
            if torch.equal(x_plane,torch.zeros(x_plane.shape)):
                x_plane=torch.zero(3)
            else:
                x_plane=self.flatten(x_plane) 
                x_plane=self.linear_relu_stack(x_plane)
                x_plane=torch.cat((x_plane,i),0)
            if i==0:
                logits=x_plane
            else:
                logits=torch.cat((logits,x_plane),dim=0)
        track=self.flatten(logits)
        track=self.linear_relu_stack2(logits)
        logits=torch.cat((logits,track),dim=0)
        return logits


class UCSPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode1=nn.Sequential(
            nn.AdaptiveMaxPool3d((10,10,20)),
            ResidualUnit(spatial_dims=3,in_channels=3,out_channels=6,kernel_size=5),
            nn.BatchNorm3d(6),
            nn.LeakyReLU(),
            ResidualUnit(spatial_dims=3,in_channels=6,out_channels=12,kernel_size=3),
            nn.BatchNorm3d(12),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d((2,2,5))
        )
        self.encode2=nn.Sequential(
            nn.AdaptiveMaxPool3d((5,5,10)),
            ResidualUnit(spatial_dims=3,in_channels=3,out_channels=6,kernel_size=5),
            nn.BatchNorm3d(6),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d((2,2,5))
        )
        self.encode3=nn.Sequential(
            nn.AdaptiveMaxPool3d((2,2,5)),
        )
        self.mlp=nn.Sequential(
            nn.Flatten(),
            nn.Linear(420,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,3),
            HailingDirectNorm()
        )
    def forward(self,x):
        feature1=self.encode1(x)
        feature2=self.encode2(x)
        feature3=self.encode3(x)
        x=torch.cat((feature1,feature2,feature3),1)
        return self.mlp(x)

class ResMax(nn.Module):
    def __init__(self,mlp_drop_rate=0,res_dropout=0):
        super().__init__()

        self.output_num=[5,4,3,2]
        self.pools=nn.ModuleList([nn.AdaptiveMaxPool3d(x) for x in self.output_num])
        self.conv=nn.Sequential(
            # nn.BatchNorm3d(3),
            # nn.Conv3d(3,8,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3,in_channels=3,out_channels=3,kernel_size=5,dropout=res_dropout),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d((8,8,30)),
            # nn.Conv3d(8,16,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3,in_channels=3,out_channels=3,kernel_size=5,dropout=res_dropout),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d((6,6,20)),
            # nn.Conv3d(16,32,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3,in_channels=3,out_channels=3,kernel_size=5,dropout=res_dropout),
            nn.BatchNorm3d(3),
            nn.LeakyReLU()
        )
        self.linear_relu_stack=nn.Sequential(
            nn.Flatten(),
            nn.Linear(672,512),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512,128),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,3)
            # HailingDirectNorm()
        )
    def forward(self,x):
        batch=x.shape[0]
        # pos=torch.where(torch.count_nonzero(x,(0,1,2,3))>0)[0]
        # x=x[:,:,:,:,pos[0]:pos[-1]+1]
        x=self.conv(x)
        for i in range(len(self.pools)):
            if i==0:
                feature=self.pools[i](x).view(batch,-1)
            else:
                feature=torch.cat((feature,self.pools[i](x).view(batch,-1)),1)
        x=self.linear_relu_stack(feature)
        return x


class DResMax(nn.Module):
    def __init__(self,mlp_drop_rate=0,res_dropout=0):
        super().__init__()
        self.output_num=[4,3,2,1]
        self.pools=nn.ModuleList([nn.AdaptiveMaxPool3d(x) for x in self.output_num])
        self.conv=nn.Sequential(
            # nn.Conv3d(3,8,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3,in_channels=3,out_channels=3,kernel_size=5,act='PRELU',norm='INSTANCE',subunits=2,dropout=res_dropout),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d((8,8,30)),
            # nn.Conv3d(8,16,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3,in_channels=3,out_channels=3,kernel_size=5,act='PRELU',norm='INSTANCE',subunits=2,dropout=res_dropout),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d((6,6,20)),
            # nn.Conv3d(16,32,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3,in_channels=3,out_channels=3,kernel_size=5,act='PRELU',norm='INSTANCE',subunits=2,dropout=res_dropout),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
        )
        self.hidden_size=[512,128]
        self.linear_relu_stack=nn.Sequential(
            nn.Flatten(),
            nn.Linear(300,self.hidden_size[0]),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(self.hidden_size[0]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size[0],self.hidden_size[1]),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(self.hidden_size[1]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size[1],3),
            HailingDirectNorm()
        )
    def forward(self,x):
        batch=x.shape[0]
        # pos=torch.where(torch.count_nonzero(x,(0,1,2,3))>0)[0]
        # x=x[:,:,:,:,pos[0]:pos[-1]+1]
        x=self.conv(x)
        for i in range(len(self.pools)):
            if i==0:
                feature=self.pools[i](x).view(batch,-1)
            else:
                feature=torch.cat((feature,self.pools[i](x).view(batch,-1)),1)
        x=self.linear_relu_stack(feature)
        return x

class CSPP(nn.Module):
    
    def __init__(self,mlp_drop_rate=0):
        super().__init__()

        self.output_num=[3,2,1]
        self.pools=nn.ModuleList([nn.AdaptiveMaxPool3d(x) for x in self.output_num])
        self.conv=nn.Sequential(
            # nn.BatchNorm3d(3),
            # nn.Conv3d(3,8,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3,in_channels=3,out_channels=3,kernel_size=5),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d((8,8,30)),
            # nn.Conv3d(8,16,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3,in_channels=3,out_channels=3,kernel_size=5),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d((6,6,20)),
            # nn.Conv3d(16,32,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3,in_channels=3,out_channels=3,kernel_size=5),
            nn.BatchNorm3d(3),
            nn.LeakyReLU()
        )
        self.linear_relu_stack=nn.Sequential(
            nn.Flatten(),
            nn.Linear(108,512),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512,128),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,3),
            HailingDirectNorm()
        )
    def forward(self,x):
        batch=x.shape[0]
        # pos=torch.where(torch.count_nonzero(x,(0,1,2,3))>0)[0]
        # x=x[:,:,:,:,pos[0]:pos[-1]+1]
        x=self.conv(x)
        for i in range(len(self.pools)):
            if i==0:
                feature=self.pools[i](x).view(batch,-1)
            else:
                feature=torch.cat((feature,self.pools[i](x).view(batch,-1)),1)
        x=self.linear_relu_stack(feature)
        return x

class ResMax_2(nn.Module):
    def __init__(self,mlp_drop_rate=0):
        super().__init__()

        self.output_num=[(4,4,10),(3,3,6),(2,2,4),(1,1,2)]
        self.pools=nn.ModuleList([nn.AdaptiveMaxPool3d(x) for x in self.output_num])
        self.conv=nn.Sequential(
            # nn.BatchNorm3d(3),
            # nn.Conv3d(3,8,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3,in_channels=3,out_channels=3,kernel_size=5),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d((8,8,30)),
            # nn.Conv3d(8,16,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3,in_channels=3,out_channels=3,kernel_size=5),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d((6,6,20)),
            # nn.Conv3d(16,32,(4,4,5),1,1,bias=False),
            ResidualUnit(spatial_dims=3,in_channels=3,out_channels=3,kernel_size=5),
            nn.BatchNorm3d(3),
            nn.LeakyReLU()
        )
        self.linear_relu_stack=nn.Sequential(
            nn.Flatten(),
            nn.Linear(696,512),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512,128),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,3),
            HailingDirectNorm()
        )
    def forward(self,x):
        batch=x.shape[0]
        # pos=torch.where(torch.count_nonzero(x,(0,1,2,3))>0)[0]
        # x=x[:,:,:,:,pos[0]:pos[-1]+1]
        x=self.conv(x)
        for i in range(len(self.pools)):
            if i==0:
                feature=self.pools[i](x).view(batch,-1)
            else:
                feature=torch.cat((feature,self.pools[i](x).view(batch,-1)),1)
        x=self.linear_relu_stack(feature)
        return x


class HailingDirectNorm(nn.Module):
    def __init__(self) -> None:
        '''
        ## Customized Layer, Normalize the Direction Vector of Hailing Data Derived from _Direct Models
        - Input: [N,3], info: [px,py,pz]
        - Output: [N,3], info: [px,py,pz](Normalized)

        N is the batch size, and the output direction vector is normalized to 1
        '''
        super().__init__()
    def forward(self,x):
        return F.normalize(x)
