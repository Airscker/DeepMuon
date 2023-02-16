'''
Author: Airscker
Date: 2022-07-17 21:01:46
LastEditors: airscker
LastEditTime: 2023-02-16 17:42:59
Description: Dataset Built for Pandax4T-III Pattern Relocalization Project

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import pickle as pkl
torch.set_default_tensor_type(torch.DoubleTensor)


def PatternData2Img(data):
    '''
    ## Convert labeled data into two dimensional image data

    ### Input format:
        It is in a reduced format which is in an Array of size (N, 3), where N is the number of pixels that have information.
        Each pixel element has 3 information (2 coordinates + 1 features). They are in order:
        - pixel id of X direction
        - pixel id of Y direction
        - Charge fraction on this PMT
    '''
    data = np.array(data, np.float32)
    img = np.zeros((17, 17))
    for i in range(data.shape[0]):
        img[int(data[i][0])][int(data[i][1])] = data[i][2]
    return img


def Data2IMGXY(datapath='..\\intro\\NewMC_validate_image2d.pkl', output='.\\data\\IMG2D_XY.pkl'):
    '''
    ### Args:
        - datapath: the path of the MC_IMAGE2D data, in .pkl format
        - output: the path of the img-xy output, which stores images-xy data converted from original experiment data, in .pkl format

    ### Return: 
        patternIMG in the form [N*[img,[x,y]]]]
        - patternIMG[i][0]: the pattern image
        - patternIMG[i][1]: the coordinates' list of the source, [x,y]\n
    '''
    data = pkl.load(open(datapath, 'rb'))
    bar = tqdm(range(len(data)), mininterval=1)
    patternIMG = []
    for i in bar:
        bar.set_description('Converting Pattern Data->Img_XY')
        img = PatternData2Img(data[i][0])
        pos = data[i][1]
        patternIMG.append([img, pos])
    if output != '':
        print('---Saving data---')
        pkl.dump(patternIMG, open(output, 'wb'))
        print('---IMG-XY data saved as {}---'.format(output))
    return patternIMG


class PandaxDataset(Dataset):
    '''
    ## Dataset build for loading prerocessed Pandax-4T data
        - image shape: [1,17,17]
        - label shape: [2]

    ### Args:
        - IMG_XY_path: the path of preprocessed image data(saved in pickle file)
    '''

    def __init__(self, IMG_XY_path='././data/IMG2D_XY.pkl'):

        self.IMG_XY_path = IMG_XY_path
        self.IMGs = []
        self.labels = []
        self.__Init()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.IMGs[idx]
        image = np.reshape(image, (1, image.shape[0], image.shape[1]))
        label = self.labels[idx]/100  # Key action
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        return image, label

    def __Init(self):
        '''Load image data'''
        data = pkl.load(open(self.IMG_XY_path, 'rb'))
        img = []
        label = []
        for i in range(len(data)):
            img.append(data[i][0])
            label.append(data[i][1])
        self.IMGs = np.array(img)
        self.labels = np.array(label)
        return img, label
