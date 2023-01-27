'''
Author: airscker
Date: 2023-01-27 19:51:21
LastEditors: airscker
LastEditTime: 2023-01-27 20:08:51
Description: Dataset built for Video Swin-Transformer (VST) CMR Screening & Diagnose Model

Copyright (c) 2023 by airscker, All Rights Reserved. 
'''
import numpy as np
from tqdm import tqdm
import pickle as pkl
import numba

import torch
from torch.utils.data import Dataset
torch.set_default_tensor_type(torch.DoubleTensor)


class NNIDecodeV2(Dataset):
    def __init__(self, ann_file: str,
                 mask_ann_map: str,
                 fusion=False,
                 modalities: list = []):
        self.ann_file = ann_file
        self.mask_ann_map = mask_ann_map
        self.fusion = fusion

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

    def load_annotations(self):
        """Load annotation file to get nifti data information."""
        nifti_info_list = []
        with open(self.ann_file, 'r')as f_ann:
            ann_info = f_ann.readlines()
        for i in range(len(ann_info)):
            ann_info[i] = ann_info[i].split('\n')[0]
            data_info = {}
            if self.fusion:
                data_info['']
        with open(self.ann_file, 'r') as fin:
            if self.fusion:
                for line in fin:
                    line_split = line.strip().split()
                    video_info = {}
                    idx = 0
                    # idx for frame_dir
                    all_dir = []
                    for i in range(len(self.type)):
                        frame_dir = line_split[idx]
                        if self.data_prefix is not None:
                            frame_dir = osp.join(self.data_prefix, frame_dir)
                        all_dir.append(frame_dir)
                        idx += 1
                    video_info['frame_dir'] = all_dir
                    if self.with_offset:
                        # idx for offset and total_frames
                        video_info['offset'] = int(line_split[idx])
                        idx += 1

                    if 'sax' in self.type or '4ch' in self.type:
                        video_info['total_frames_cine'] = int(line_split[idx])
                        idx += 1
                    if 'lge' in self.type:
                        video_info['total_frames_lge'] = int(line_split[idx])
                        idx += 1

                    # idx for label[s]
                    label = [int(x) for x in line_split[idx:]]
                    assert label, f'missing label in line: {line}'
                    if self.multi_class:
                        assert self.num_classes is not None
                        video_info['label'] = label
                    else:
                        #                        assert len(label) == 1
                        video_info['label'] = label
                    video_infos.append(video_info)

            else:
                for line in fin:
                    line_split = line.strip().split()
                    video_info = {}
                    idx = 0
                    # idx for frame_dir
                    frame_dir = line_split[idx]
                    if self.data_prefix is not None:
                        frame_dir = osp.join(self.data_prefix, frame_dir)
                    video_info['frame_dir'] = frame_dir
                    idx += 1
                    if self.with_offset:
                        # idx for offset and total_frames
                        video_info['offset'] = int(line_split[idx])
                        video_info['total_frames'] = int(line_split[idx + 1])
                        idx += 2
                    else:
                        # idx for total_frames
                        video_info['total_frames'] = int(line_split[idx])
                        idx += 1
                    # idx for label[s]
                    label = [int(x) for x in line_split[idx:]]
                    assert label, f'missing label in line: {line}'
                    if self.multi_class:
                        assert self.num_classes is not None
                        video_info['label'] = label
                    else:
                        assert len(label) == 1
                        video_info['label'] = label[0]
                    video_infos.append(video_info)

        return video_infos
