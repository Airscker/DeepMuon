'''
Author: airscker
Date: 2023-05-15 13:41:53
LastEditors: airscker
LastEditTime: 2023-05-15 13:46:58
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

class MinistDataset(Dataset):
    def __init__(self,train=True) -> None:
        super().__init__()
        self.dataset=datasets.FashionMNIST(
            root="data",
            train=train,
            download=True,
            transform=ToTensor()
        )
    def __getitem__(self, index):
        return self.dataset[index][0],self.dataset[index][1]
    def __len__(self):
        return len(self.dataset)
    