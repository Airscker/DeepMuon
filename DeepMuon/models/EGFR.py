'''
Author: airscker
Date: 2022-12-26 21:36:52
LastEditors: airscker
LastEditTime: 2023-03-14 17:04:36
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from monai.networks.blocks.convolutions import ResidualUnit


class ResMax_C1(nn.Module):
    def __init__(self, mlp_drop_rate=0, res_dropout=0):
        super().__init__()
        self.output_num = [32, 16, 8]
        self.pools = nn.ModuleList([nn.AdaptiveMaxPool3d(x)
                                   for x in self.output_num])
        self.conv1 = nn.Sequential(
            ResidualUnit(spatial_dims=2, in_channels=3, out_channels=6, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool2d(self.output_num[0]))
        self.conv2=nn.Sequential(
            ResidualUnit(spatial_dims=2, in_channels=6, out_channels=6, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool2d(self.output_num[1]))
        self.conv3=nn.Sequential(
            ResidualUnit(spatial_dims=2, in_channels=6, out_channels=6, kernel_size=5,
                         act='PRELU', norm='INSTANCE', subunits=2, dropout=res_dropout),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool2d(self.output_num[2]))
        self.hidden_size = [1024, 256]
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(int(np.sum(np.array(self.output_num)**2)*6), self.hidden_size[0]),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(self.hidden_size[0]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.Dropout(mlp_drop_rate),
            nn.BatchNorm1d(self.hidden_size[1]),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size[1], 2),
            nn.Softmax(1)
        )

    def freeze_stages(self):
        pass

    def forward(self, x:torch.Tensor):
        batch=x.shape[0]
        conv_f1=self.conv1(x)
        conv_f2=self.conv2(conv_f1)
        conv_f3=self.conv3(conv_f2)
        linear_feature=torch.cat([conv_f1.view(batch,-1),conv_f2.view(batch,-1),conv_f3.view(batch,-1)],1)
        linear_feature=self.linear_relu_stack(linear_feature)
        return linear_feature
    
class ResNet50_C1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model=resnet50()
        self.model.fc=nn.Linear(self.model.fc.in_features,2)
        self.softmax=nn.Softmax(1)
    def forward(self,x):
        return self.softmax(self.model(x))
    
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class SConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SConv, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch, 1, padding=1),
            nn.Conv2d(in_ch, out_ch, 1),
            
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)

class ResNet(nn.Module):

    # block = BasicBlock or Bottleneck
    # block_num为残差结构中conv2_x~conv5_x中残差块个数，是一个列表
    def __init__(self, block, blocks_num, num_classes= 2 , include_top=None):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
              
        self.layer1 = self._make_layer(block,  64, blocks_num[0])            # conv2_x 
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # conv5_x
        

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)            
        self.fc = nn.Sequential(    # 定义自己的分类层                
                nn.Linear(3904, 512),
                nn.ReLU(True),
                nn.Dropout(0.4),
                nn.Linear(512, 512)  ,
                nn.Sigmoid(),
                nn.Dropout(0.4),
                # nn.Linear(512, num_classes),
                )  
        self.classifier = nn.Linear(512, num_classes)

        self.rec = SConv(3904, 128)                             
        self.recons = nn.Sequential(               
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                DoubleConv(64, 64),
                nn.ConvTranspose2d(64, 32, 2, stride=2),
                DoubleConv(32, 32), 
                nn.Conv2d(32, 3, 1))


        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    # channel为残差结构中第一层卷积核个数
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

        # ResNet50/101/152的残差结构，
        block.expansion=4
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        
        x_s = self.maxpool(x1)
        x2 = self.layer1(x_s)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)


        # Multi-scale的上下采样结构
        f5 = nn.functional.interpolate(x5, scale_factor=8, mode='bilinear', align_corners=True)
        # print("f5 shape:",f5.shape)     #[4, 2048,56,56]      
        f4 = nn.functional.interpolate(x4, scale_factor=4, mode='bilinear', align_corners=True)
        # print("f4 shape:",f4.shape)       #[4, 1024,56,56]        
        f3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        # print("f3 shape:",f3.shape)      #[4, 512,56,56]          
        f2 = nn.functional.interpolate(x2, scale_factor=1, mode='bilinear', align_corners=True)
        # print("f2 shape:",f2.shape)       #[4, 256,56,56]        
        f1 = nn.functional.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=True)
        # print("f1 shape:",f1.shape)        #[4, 64,56,56]        
        
        f_all = torch.cat([f1,f2,f3,f4,f5],dim=1)  
        # print("f_all shape:",f_all.shape) #[4, 3904,56,56]
        
        #分类
        dense = self.avgpool(f_all)
        # print("dense shape: ", dense.shape)
        dense1 = dense.view(f_all.size(0), 3904)    
        # dense = nn.AdaptiveAvgPool2d(1)(f_all).view(f_all.size(0), 3904)        
        # print("dense shape:" ,dense.shape) #[4,3904]              
        fea = self.fc(dense1)
        cla_out = self.classifier(fea)# 自定义的分类部分             

             
        # 重建
        rec_fea = self.rec(f_all)          
        # print("rec_fea shape:",rec_fea.shape)  #[4,128,56,56]                       
        re_1 = self.recons[0](rec_fea)
        # print("re_1 shape: ", re_1.shape)  #[64,112,112]      
        c11 = self.recons[1](re_1)
        # print("c11 shape: ", c11.shape)       #[64,112,112] 
        re_2 = self.recons[2](c11)
        # print("re_2 shape: ", re_2.shape)        #[32,224,224]
        c12 = self.recons[3](re_2)
        # print("c12 shape: ", c12.shape)        #[32,224,224]
        c13 = self.recons[4](c12)
        # print("c13 shape: ", c13.shape)        #[1,224,224]
        rec_out = nn.Sigmoid()(c13)
        # print("rec_out shape: ", rec_out.shape) #[1,224,224]  //   [3,224,224]
        return cla_out, rec_out, fea

class Bottleneck(nn.Module):
    expansion = 4  # 残差结构中第三层卷积核个数是第一/二层卷积核个数的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # 捷径分支 short cut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet50_cls_rec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model=ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2, include_top=None)
    def forward(self,x):
        cla_out,rec_out,fea=self.model(x)
        cla_out2,rec_out2,fea2=self.model(rec_out)
        return cla_out,rec_out,fea,fea2