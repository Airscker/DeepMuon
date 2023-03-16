'''
Author: airscker
Date: 2023-03-12 14:21:18
LastEditors: airscker
LastEditTime: 2023-03-17 00:30:58
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import numpy as np
import pickle as pkl
import cv2
from skimage import exposure

import torch
from torch.utils.data import Dataset

def add_random_number(img:np.ndarray,ratio=0.4):
    if np.random.rand()<ratio:
        noise=np.random.randn(img.shape[0],img.shape[1],img.shape[2])
    return img+noise

def flip(img:np.ndarray,ratio=0.4):
    if np.random.rand()<ratio:
        flip_axis=np.random.randint(0,2)
        for i in range(len(img)):
            img[i]=np.flip(img[i],flip_axis)
    return img

def rotate(img:np.ndarray,angle_range=180,ratio=0.4):
    if np.random.rand()<ratio:
        channel,cols,rows=img.shape
        angle=np.random.rand()*angle_range
        Matrix = cv2.getRotationMatrix2D((cols/2, rows/2), (2*np.random.rand()-1)*angle, 1)
        for i in range(len(img)):
            img[i] = cv2.warpAffine(img[i], Matrix, (rows, cols), borderValue=(255, 255, 255))
    return img

def bright(img:np.ndarray,light_range=(0.5,1.2),ratio=0.4):
    if np.random.rand()<ratio:
        ratio=np.random.rand(light_range[0],light_range[1])
        for i in range(len(img)):
            img[i] = exposure.adjust_gamma(img[i], ratio)
    return img

def exclude_key(dictionary: dict, del_key: str = 'type'):
    '''
    Delete key-value map from dictionary
    '''
    new_dict = {}
    for key in dictionary.keys():
        if key != del_key:
            new_dict[key] = dictionary[key]
    return new_dict

class EGFR_NPY(Dataset):
    def __init__(self,
                 img_dataset='',
                 augment=True,
                 augment_ratio=[0.3,0.6],
                 augment_pipeline=[dict(type='add_random_number'),
                                   dict(type='flip'),
                                   dict(type='rotate',angle_range=180),
                                   dict(type='bright',light_range=(0.8,1.1))]) -> None:
        assert img_dataset.endswith('.pkl'), f'PKL-format image dataset expected, however, {img_dataset} given.'
        super().__init__()
        with open(img_dataset,'rb')as f:
            '''list([imgs,label,pid]), imgs.shape=(224,224)'''
            self.dataset=pkl.load(f)
        self.augment=augment
        self.augment_ratio=augment_ratio
        self.augment_pipeline=augment_pipeline
    def clip_top_bottom(self,img:np.ndarray,prop=[0.001,0.001]):
        pixel_values=np.sort(img.flatten())
        top_clip=pixel_values[int(len(pixel_values)*(1-prop[1]))]
        bottom_clip=pixel_values[int(len(pixel_values)*prop[0])]
        img=np.clip(img,bottom_clip,top_clip)
        return img
    def norm_range(self,img:np.ndarray):
        return (img-np.min(img))/(np.max(img)-np.min(img))
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        img=np.array([self.clip_top_bottom(self.dataset[index][0])]*3)
        if self.augment:
            env=globals()
            for augment in self.augment_pipeline:
                try:
                    img = env[augment['type']](img, **exclude_key(augment),ratio=self.augment_ratio[self.dataset[index][1]])
                except:
                    pass
        img=self.norm_range(img)
        img=torch.from_numpy(img)
        label=torch.LongTensor([self.dataset[index][1]])
        return img,label