"""
Author: airscker
Date: 2023-04-30 15:40:28
LastEditors: airscker
LastEditTime: 2023-05-02 13:41:38
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle as pkl


def valence_map(elements: list, valences: list):
    """
    ## Given a list of elements and their corresponding valences, create a dictionary mapping each element to its valence.

    ### Args:
        - elements: a list of elements, such as [Element F, Element Pb].
        - valences: a list of valences corresponding to the elements, such as ['Pb2+', 'F-'].
    ### Return:
        A dictionary mapping each element to its valence, such as {'F': -1, 'Pb': 2}.
    """
    map = {}
    for ele in elements:
        ele = str(ele)
        for val in valences:
            if ele in val:
                num = val.replace(ele, "")
                if not num[0].isalpha():
                    if num == "-":
                        num = "1-"
                    if num == "+":
                        num = "1+"
                    num = float(num[-1] + num[:-1])
                    map[ele] = num
    return map


class ValenceDataset(Dataset):
    """
    The ValenceDataset class is a PyTorch Dataset that loads and preprocesses X-ray absorption near edge structure (XANES) spectra data for machine learning tasks.
    It takes an annotation file as input, which contains the paths to the data files to be loaded. The class unpacks the data files,
    extracts the XANES spectra and corresponding valences of the elements in the spectra, and returns them as a tuple of data and label for each sample.

    ## Args:
        - annotation: the path of the annotation text file which contains the paths of data samples to be used to train/test the model.
    """

    def __init__(self, annotation="",xy_label=False,available_valences=[2,3]):
        super().__init__()
        with open(annotation, "r") as f:
            self.mp_list = f.readlines()
        self.dataset = []
        self.xy_label=xy_label
        self.available_valences=available_valences
        self.unpack_data()

    def unpack_data(self):
        for i in range(len(self.mp_list)):
            self.mp_list[i] = self.mp_list[i].split("\n")[0]
            with open(self.mp_list[i], "rb") as f:
                info = pkl.load(f)
            valences = valence_map(info["elements"], info["valences"])
            spectrum = info["xanes"]
            for sub_spec in spectrum.keys():
                element = sub_spec.split("-")[-2]
                # if element == "Fe" and valences[element].is_integer() and sub_spec.endswith('K'):
                if element == "Fe" and sub_spec.endswith('K'):
                    if int(valences[element]) not in self.available_valences:
                        continue
                    if self.xy_label:
                        spec=np.array(spectrum[sub_spec])
                    else:
                        spec=np.array(spectrum[sub_spec][1])
                    self.dataset.append(
                        [spec, int(valences[element])-min(self.available_valences)]
                    )

    def __getitem__(self, index):
        data, label = self.dataset[index]
        data = torch.from_numpy(data).type(torch.FloatTensor)
        label = torch.LongTensor([label])
        # label=torch.Tensor([float(label)]).type(torch.FloatTensor)
        return data, label

    def __len__(self):
        return len(self.dataset)
    

class ValenceDatasetV2(Dataset):
    """
    The ValenceDatasetV2 class is a PyTorch Dataset that loads and preprocesses X-ray absorption near edge structure (XANES) spectra data for machine learning tasks.
    It takes an PKL file which contains all data samples as input, extracts the XANES spectra and corresponding valences of the elements in the spectra, and returns them as a tuple of data and label for each sample.

    ## Args:
        - annotation: the path of the annotation text file which contains the paths of data samples to be used to train/test the model.
    """

    def __init__(self, annotation="",xy_label=True):
        super().__init__()
        self.xy_label=xy_label
        with open(annotation, "rb") as f:
            self.dataset=pkl.load(f)

    def __getitem__(self, index):
        label, data = self.dataset[index]
        if not self.xy_label:
            data=data[1]
        else:
            data=np.array(data)
            data[0]=data[0]/10000.0
        data = torch.from_numpy(data).type(torch.FloatTensor)
        label = torch.Tensor([float(label)]).type(torch.FloatTensor)
        return data, label

    def __len__(self):
        return len(self.dataset)
    
