'''
Author: airscker
Date: 2023-04-30 15:40:28
LastEditors: airscker
LastEditTime: 2023-04-30 16:23:57
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle as pkl

def valence_map(elements:list,valences:list):
    """
    ## Given a list of elements and their corresponding valences, create a dictionary mapping each element to its valence.
    
    ### Args:
        - elements: a list of elements, such as [Element F, Element Pb].
        - valences: a list of valences corresponding to the elements, such as ['Pb2+', 'F-'].
    ### Return:
        A dictionary mapping each element to its valence, such as {'F': -1, 'Pb': 2}.
    """
    map={}
    for ele in elements:
        ele=str(ele)
        for val in valences:
            if ele in val:
                num=val.replace(ele,'')
                if num=='-':
                    num='1-'
                if num=='+':
                    num='1+'
                num=int(num[-1]+num[:-1])
                map[ele]=num
    return map

class ValenceDataset(Dataset):
    def __init__(self,annotation=''):
        super().__init__()
        with open(annotation,'r')as f:
            self.mp_list=f.readlines()
        self.dataset=[]
        self.unpack_data()
    def unpack_data(self):
        for i in range(len(self.mp_list)):
            self.mp_list[i]=self.mp_list[i].split('\n')[0]
            with open(self.mp_list[i],'rb')as f:
                info=pkl.load(f)
            valences=valence_map(info['elements'],info['valences'])
            spectrum=info['xanes']
            for sub_spec in spectrum.keys():
                element=sub_spec.split('-')[-2]
                self.dataset.append([np.array(spectrum[sub_spec]),valences[element]])
    def __getitem__(self, index):
        super().__getitem__(index)
        data,label=self.dataset[index]
        data=torch.from_numpy(data).type(torch.FloatTensor)
        label=torch.LongTensor([label])
        return data,label
    def __len__(self):
        return len(self.dataset)