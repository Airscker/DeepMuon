'''
Author: airscker
Date: 2022-09-17 18:11:14
LastEditors: airscker
LastEditTime: 2022-12-03 23:40:04
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''

import numpy as np
from tqdm import tqdm
import pickle as pkl
import numba

import torch
from torch.utils.data import Dataset
torch.set_default_tensor_type(torch.DoubleTensor)

@numba.jit
def pattern_data_1T(event,shape=(10,10,40,3)):
    """## Convert the Original Hailing Data into Pattern Image with specified shape
    - Args:
        - event: Single Hailing original data
        - shape: The shape of the pattern data. Defaults to (10,10,40,3) for 1TeV data, or (10,10,50,3) for 10TeV data
    - Returns:
        - Converted Pattern Image with specified shape, dtype: nparray
    """
    pattern=np.zeros(shape)
    for i in range(len(event)):
        pattern[int(event[i][0])][int(event[i][1])][int(event[i][2])]=event[i][3:]
    return pattern

def Rotate90(image,label):
    """
    Rotate the image 90 degrees clockwise and return the rotated image and label.
    @param image - the image to rotate.
    @param label - the label to rotate.
    @returns the rotated image and label.
    """
    image=np.transpose(np.array(image),(1,0,2,3))
    image=image[::-1,...]
    label=np.array([-label[1],label[0],label[2]])
    return image,label

def Rotate180(image,label):
    """
    Rotate the image 180 degrees. Also rotate the label 180 degrees.
    @param image - the image to rotate.
    @param label - the label to rotate.
    @returns the rotated image and label.
    """
    image=np.array(image)
    image=image[::-1,::-1,:,:]
    label=np.array([-label[0],-label[1],label[2]])
    return image,label

def Flip(image,label):
    """
    Flip the image and label.
    @param image - the image to flip
    @param label - the label to flip
    @returns the flipped image and label
    """
    image=np.array(image)
    image=image[:,:,::-1,:]
    label=np.array([label[0],label[1],-label[2]])
    return image,label

class HailingDataset_Direct(Dataset):
    def __init__(self,datapath='./Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl',min_z=9):
        '''
        ## Dataset Built for Loading the Preprocessed Hailing 1TeV/10TeV Data
        - Args: 
            - datapath: The datapth of the preprocessed Hailing data, default to be './Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl'
        - Output:
            - Pattern Image, shape: [10,10,40/50,3], dtype: nparray -> torch.tensor
            - Position-Direction, shape: [3,], dtype: nparray -> torch.tensor, info: [px,py,pz]
        '''
        self.datapath=datapath
        self.origin_data=None
        self.pattern_imgs=[]
        self.pos_direction=[]
        self.min_z=min_z
        self.__Init()
    def __len__(self):
        return len(self.origin_data)
    def __getitem__(self, index):
        image=np.array(self.origin_data[index][0])
        array=np.nonzero(np.count_nonzero(image,axis=(0,1,3)))
        image=image[:,:,array[0][0]:(array[0][-1]+1),:]
        image=np.append(image,np.zeros((10,10,max(self.min_z-image.shape[2],0),3)),axis=2)
        image=torch.from_numpy(image)
        image=torch.permute(image,(3,0,1,2))
        image[1:,:,:,:]=0.0001*image[1:,:,:,:]
        label=torch.from_numpy(self.origin_data[index][1][3:])
        return image,label
    def __Init(self):
        with open(self.datapath,'rb')as f:
            self.origin_data=pkl.load(f)
        f.close()

class SSP_Dataset(Dataset):
    def __init__(self,datapath='./Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl'):
        '''
        ## Dataset Built for Loading the Preprocessed Hailing 1TeV/10TeV Data
        - Args: 
            - datapath: The datapth of the preprocessed Hailing data, default to be './Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl'
        - Output:
            - Pattern Image, shape: [10,10,40/50,3], dtype: nparray -> torch.tensor
            - Position-Direction, shape: [3,], dtype: nparray -> torch.tensor, info: [px,py,pz]
        '''
        self.datapath=datapath
        self.origin_data=None
        self.pattern_imgs=[]
        self.pos_direction=[]
        self.__Init()
        self.__sort()
    def __len__(self):
        return len(self.origin_data)*8
    def __getitem__(self, index):
        # image=np.array(self.origin_data[index][0])
        # array=np.nonzero(np.count_nonzero(image,axis=(0,1,3)))
        # image=image[:,:,array[0][0]:(array[0][-1]+1),:]
        # image=np.append(image,np.zeros((10,10,max(9-image.shape[2],0),3)),axis=2)
        image=torch.from_numpy(self.sortedata[index][0])
        image=torch.permute(image,(3,0,1,2))
        image[1:,:,:,:]=0.0001*image[1:,:,:,:]
        label=100*torch.from_numpy(self.sortedata[index][1][3:])
        return image,label
    def __sort(self):
        len_map={}
        for i in range(len(self.origin_data)):
            image=np.array(self.origin_data[i][0])
            # array=np.nonzero(np.count_nonzero(image,axis=(0,1,3)))
            # image=image[:,:,array[0][0]:(array[0][-1]+1),:]
            # image=np.append(image,np.zeros((10,10,max(9-image.shape[2],0),3)),axis=2)
            image=np.delete(image,np.where(np.count_nonzero(image,axis=(0,1,3))==0),axis=2)
            image=np.append(image,np.zeros((10,10,max(9-image.shape[2],0),3)),axis=2)
            length=image.shape[2]
            if length in len_map:
                len_map[length].append([image,self.origin_data[i][1]])
            else:
                len_map[length]=[[image,self.origin_data[i][1]]]
        len_keys=list(len_map.keys())
        len_keys=sorted(len_keys)
        self.sortedata=[]
        for i in range(len(len_keys)):
            items=len_map[len_keys[i]]
            for j in range(len(items)):
                self.sortedata.append(items[j])
    def __Init(self):
        with open(self.datapath,'rb')as f:
            self.origin_data=pkl.load(f)
        f.close()

class HailingDataset_Direct2(Dataset):
    def __init__(self,datapath='./Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl',augment=False):
        '''
        ## Dataset Built for Loading the Preprocessed Hailing 1TeV/10TeV Data
        - Args: 
            - datapath: The datapth of the preprocessed Hailing data, default to be './Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl'
        - Output:
            - Pattern Image, shape: [10,10,40/50,3], dtype: nparray -> torch.tensor
            - Position-Direction, shape: [3,], dtype: nparray -> torch.tensor, info: [px,py,pz]
        '''
        self.datapath=datapath
        self.origin_data=None
        self.pattern_imgs=[]
        self.pos_direction=[]
        self.augmentation={0:Rotate180,1:Rotate90,2:Flip}
        self.augment=augment
        self.__Init()
    def __len__(self):
        return len(self.origin_data)
    def __getitem__(self, index):
        image=np.array(self.origin_data[index][0])
        label=self.origin_data[index][1][3:]
        '''Data augmentation'''
        if self.augment:
            oper=np.unique(np.random.randint(0,1,np.random.randint(0,2)))#[-1,0]range,[0,1]random length
            for oper_i in range(len(oper)):
                image,label=self.augmentation[oper[oper_i]](image,label)
        image=torch.from_numpy(image.copy())
        image=torch.permute(image,(3,0,1,2))
        image[1:,:,:,:]=0.0001*image[1:,:,:,:]
        label=torch.from_numpy(label)
        return image,label
    def __Init(self):
        print(f'Loading dataset {self.datapath}')
        with open(self.datapath,'rb')as f:
            self.origin_data=pkl.load(f)
        f.close()
        print(f'Dataset {self.datapath} loaded')

class HailingDataset_Plane_Z(Dataset):
    def __init__(self,datapath='./Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl',min_z=9):
        self.datapath=datapath
        self.origin_data=None
        self.pattern_imgs=[]
        self.pos_direction=[]
        self.min_z=min_z
        self.__Init()
    def __len__(self):
        return len(self.origin_data)
    def __getitem__(self, index):
        image=np.array(self.origin_data[index][0])
        image=torch.from_numpy(image)
        image=torch.permute(image,(3,0,1,2))
        image[1:,:,:,:]=0.0001*image[1:,:,:,:]
        label=self.origin_data[index][1]
        label1=[]
        for i in range(40):
            t=np.float((i-label[2])/label[5])
            label1.append([np.array((label[3]*t+label[0]),(label[4]*t+label[1]),i)])
        label1.append(np.array(self.origin_data[index][1][3:]))
        label1=torch.from_numpy(label1)
        return image,label1
    def __Init(self):
        with open(self.datapath,'rb')as f:
            self.origin_data=pkl.load(f)
        f.close()



class HailingData_Init:
    def __init__(self,datapath='./Hailing-Muon/data/1TeV/validate_norm.pkl',output='Hailing_1TeV_val_data.pkl',shape=(10,10,40,3)):
        """## Convert the Original Hailing Data into a list of Pattern Images(dtype: nparray -> torch.tensor) as well as Position-Direction(dtype: nparray -> torch.tensor)
        - Args:
            - datapath: The data path of the original Hailing data. Defaults to './Hailing-Muon/data/1TeV/validate_norm.pkl'.
            - output: The output path of the converted Hailing pattern image data: [description]. Defaults to 'Hailing_1TeV_val_data.pkl'.
            - shape: The shape of the Hailing pattern image data. Defaults to (10,10,40,3) for 1TeV data and (10,10,50,3) for 10TeV data available.
        """
        self.datapath=datapath
        self.origin_data=None
        self.output=output
        self.shape=shape
        self.__Init()
    def __Init(self):
        '''Used to save pattern imgs, shape[10,10,40/50,3] and label[6,]'''
        with open(self.datapath,'rb')as f:
            self.origin_data=pkl.load(f)
        f.close()
        bar=tqdm(range(len(self.origin_data)),mininterval=1)
        bar.set_description('Loading dataset')
        newdata=[]
        for i in bar:
            if len(self.origin_data[i][0].shape)==2:
                label=np.array(self.origin_data[i][1])
                label[:3]=label[:3]/1000.0
                newdata.append([np.array(pattern_data_1T(self.origin_data[i][0],shape=self.shape)),label])
        with open(self.output,'wb')as f:
            pkl.dump(newdata,f)
        f.close()
        print(f'file saved as {self.output}')
# data=HailingData_Init(datapath='../Hailing-Muon/data/1TeV/validate_norm.pkl',output='../Hailing-Muon/data/1TeV/Hailing_1TeV_val_data_1k.pkl')
# data=HailingData_Init(datapath='../Hailing-Muon/data/1TeV/train_norm.pkl',output='../Hailing-Muon/data/1TeV/Hailing_1TeV_train_data_1k.pkl')
# HailingDataset_Direct()
