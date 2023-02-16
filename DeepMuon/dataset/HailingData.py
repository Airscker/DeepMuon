'''
Author: airscker
Date: 2022-09-17 18:11:14
LastEditors: airscker
LastEditTime: 2023-02-16 19:08:58
Description: Datasets Built for Hailing TRIDENT Project

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import numpy as np
from tqdm import tqdm
import pickle as pkl
import numba

import torch
from torch.utils.data import Dataset
torch.set_default_tensor_type(torch.DoubleTensor)


@numba.jit
def pattern_data_1T(event, shape=(10, 10, 40, 3)):
    """
    ## Convert the Original Hailing Data into Pattern Image with specified shape

    ### Args:
        - event: Single Hailing original data
        - shape: The shape of the pattern data. Defaults to (10,10,40,3) for 1TeV data, or (10,10,50,3) for 10TeV data

    ### Returns:
        - Converted Pattern Image with specified shape, dtype: nparray
    """
    pattern = np.zeros(shape)
    for i in range(len(event)):
        pattern[int(event[i][0])][int(event[i][1])
                                  ][int(event[i][2])] = event[i][3:]
    return pattern


def Rotate90(image, label):
    """
    ## Rotate the 3D image along Z axis by 90 degrees and return the rotated image and label.

    ### Args:
        - image: the image to rotate.
        - label: the label to rotate.

    ### Return:
        - the rotated image and label.
    """
    image = np.transpose(np.array(image), (1, 0, 2, 3))
    image = image[::-1, ...]
    label = np.array([-label[1], label[0], label[2]])
    return image, label


def Rotate180(image, label):
    """
    ## Rotate the 3D image along Z axis by 180 degrees and return the rotated image and label.

    ### Args:
        - image: the image to rotate.
        - label: the label to rotate.

    ### Return:
        - the rotated image and label.
    """
    image = np.array(image)
    image = image[::-1, ::-1, :, :]
    label = np.array([-label[0], -label[1], label[2]])
    return image, label


def Flip(image, label):
    """
    ## Flip the 3D image' Z axis and return the rotated image and label.

    ### Args:
        - image: the image to rotate.
        - label: the label to rotate.

    ### Return:
        - the flipped image and label.
    """
    image = np.array(image)
    image = image[:, :, ::-1, :]
    label = np.array([label[0], label[1], -label[2]])
    return image, label


def Same(image, label):
    return image, label


class HailingDataset_DirectV3(Dataset):
    '''
    ## Dataset Built for Loading the Preprocessed Hailing 1TeV/10TeV Data, origial data shape: [10,10,40/50,3]

    ### Args: 
        - datapath: The datapth of the preprocessed Hailing data, default to be './Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl'

    ### Output:
        - Pattern Image, shape: [3,40/50,10,10], dtype: nparray -> torch.tensor
        - Position-Direction, shape: [3,], dtype: nparray -> torch.tensor, info: [px,py,pz]
    '''

    def __init__(self, datapath='./Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl', augment=False):

        self.datapath = datapath
        self.origin_data = None
        self.pattern_imgs = []
        self.pos_direction = []
        self.augmentation = {0: Rotate180, 1: Rotate90, 2: Flip, 3: Same}
        self.augment = augment
        self.__Init()

    def __len__(self):
        return len(self.origin_data)

    def __getitem__(self, index):
        image = np.array(self.origin_data[index][0])
        label = self.origin_data[index][1][3:]
        '''Data augmentation'''
        if self.augment:
            # [0,3]range,[0,3]random length
            oper = np.unique(np.random.randint(0, 4, np.random.randint(0, 4)))
            for oper_i in range(len(oper)):
                image, label = self.augmentation[oper[oper_i]](image, label)
        image = torch.from_numpy(image.copy())
        # image = torch.permute(image, (3, 0, 1, 2))
        image = torch.permute(image, (3, 2, 0, 1))
        image[1:, :, :, :] = 0.0001*image[1:, :, :, :]
        label = torch.from_numpy(label)
        return image, label

    def __Init(self):
        print(f'Loading dataset {self.datapath}')
        with open(self.datapath, 'rb')as f:
            self.origin_data = pkl.load(f)
        f.close()
        print(f'Dataset {self.datapath} loaded')


class HailingDataset_Direct2(Dataset):
    '''
    ## Dataset Built for Loading the Preprocessed Hailing 1TeV/10TeV Data, origial data shape: [10,10,40/50,3]

    ### Args: 
        - datapath: The datapth of the preprocessed Hailing data, default to be './Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl'

    ### Output:
        - Pattern Image, shape: [3,10,10,40/50], dtype: nparray -> torch.tensor
        - Position-Direction, shape: [3,], dtype: nparray -> torch.tensor, info: [px,py,pz]
    '''

    def __init__(self, datapath='./Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl', augment=False):

        self.datapath = datapath
        self.origin_data = None
        self.pattern_imgs = []
        self.pos_direction = []
        self.augmentation = {0: Rotate180, 1: Rotate90, 2: Flip, 3: Same}
        self.augment = augment
        self.__Init()

    def __len__(self):
        return len(self.origin_data)

    def __getitem__(self, index):
        image = np.array(self.origin_data[index][0])
        label = self.origin_data[index][1][3:]
        '''Data augmentation'''
        if self.augment:
            # [0,3]range,[0,3]random length
            oper = np.unique(np.random.randint(0, 4, np.random.randint(0, 4)))
            for oper_i in range(len(oper)):
                image, label = self.augmentation[oper[oper_i]](image, label)
        image = torch.from_numpy(image.copy())
        image = torch.permute(image, (3, 0, 1, 2))
        # image = torch.permute(image, (3, 2, 0, 1))
        image[1:, :, :, :] = 0.0001*image[1:, :, :, :]
        label = torch.from_numpy(label)
        return image, label

    def __Init(self):
        print(f'Loading dataset {self.datapath}')
        with open(self.datapath, 'rb')as f:
            self.origin_data = pkl.load(f)
        f.close()
        print(f'Dataset {self.datapath} loaded')


class HailingData_Init:
    """
    ## Convert the Original Hailing Data into a list of Pattern Images(dtype: nparray -> torch.tensor) as well as Position-Direction(dtype: nparray -> torch.tensor)

    ### Args:
        - datapath: The data path of the original Hailing data. Defaults to './Hailing-Muon/data/1TeV/validate_norm.pkl'.
        - output: The output path of the converted Hailing pattern image data: [description]. Defaults to 'Hailing_1TeV_val_data.pkl'.
        - shape: The shape of the Hailing pattern image data. Defaults to (10,10,40,3) for 1TeV data and (10,10,50,3) for 10TeV data available.
    """

    def __init__(self, datapath='./Hailing-Muon/data/1TeV/validate_norm.pkl', output='Hailing_1TeV_val_data.pkl', shape=(10, 10, 40, 3)):

        self.datapath = datapath
        self.origin_data = None
        self.output = output
        self.shape = shape
        self.__Init()

    def __Init(self):
        '''Used to save pattern imgs, shape[10,10,40/50,3] and label[6,]'''
        with open(self.datapath, 'rb')as f:
            self.origin_data = pkl.load(f)
        f.close()
        bar = tqdm(range(len(self.origin_data)), mininterval=1)
        bar.set_description('Loading dataset')
        newdata = []
        for i in bar:
            if len(self.origin_data[i][0].shape) == 2:
                label = np.array(self.origin_data[i][1])
                label[:3] = label[:3]/1000.0
                newdata.append(
                    [np.array(pattern_data_1T(self.origin_data[i][0], shape=self.shape)), label])
        with open(self.output, 'wb')as f:
            pkl.dump(newdata, f)
        f.close()
        print(f'file saved as {self.output}')
