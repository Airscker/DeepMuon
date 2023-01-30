'''
Author: airscker
Date: 2023-01-27 19:51:21
LastEditors: airscker
LastEditTime: 2023-01-30 22:17:29
Description:
    ## Dataset built for:
        - Video Swin-Transformer (VST) CMR Screening & Diagnose Model
        - CNNLSTM CMR Screening & Diagnose Model

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''
import os
import cv2
import random
import warnings
import numpy as np
import pickle as pkl
import SimpleITK as sitk
from PIL import Image
from skimage import exposure

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
torch.set_default_tensor_type(torch.DoubleTensor)


def HistEqual(frames: np.ndarray):
    for i in range(len(frames)):
        img = frames[i]
        for j in range(img.shape[-1]):
            img[:, :, j] = cv2.equalizeHist(np.uint8(img[:, :, j]))
        frames[i] = img
    return frames


def SingleNorm(frames: np.ndarray):
    return (frames-np.mean(frames))/np.std(frames)


def Random_rotate(frames: np.ndarray, range=30, ratio=0.3):
    if np.random.rand() < ratio:
        rows, cols = frames.shape[1:3]
        Matrix = cv2.getRotationMatrix2D(
            (cols/2, rows/2), (2*np.random.rand()-1)*range, 1)
        for i in range(len(frames)):
            frames[i] = cv2.warpAffine(
                frames[i], Matrix, (rows, cols), borderValue=(255, 255, 255))
    return frames


def Random_Gamma_Bright(frames: np.ndarray, ratio=0.5, low=0.4, high=1):
    if np.random.rand() < ratio:
        for i in range(len(frames)):
            frames[i] = exposure.adjust_gamma(
                frames[i], np.random.uniform(low, high))
    return frames


def Padding(frames: np.ndarray, size=(120, 120)):
    new_imgs = []
    for img in frames:
        x = img.shape[0]
        y = img.shape[1]
        pad_x1 = (size[0]-x)//2
        pad_x2 = size[0]-x-pad_x1
        pad_y1 = (size[1]-y)//2
        pad_y2 = size[1]-y-pad_y1
        new_img = np.zeros((size[0], size[1], img.shape[-1]))
        if pad_x1 < 0 or pad_x2 < 0:
            img = img[-pad_x1:x+pad_x2, :, :]
            pad_x1 = 0
            pad_x2 = 0
        if pad_y1 < 0 or pad_y2 < 0:
            img = img[:, -pad_y1:y+pad_y2, :]
            pad_y1 = 0
            pad_y2 = 0
        for i in range(img.shape[-1]):
            new_img[:, :, i] = np.pad(img[:, :, i], ((
                pad_x1, pad_x2), (pad_y1, pad_y2)), 'constant', constant_values=(0, 0))
            new_imgs.append(new_img)
    return np.array(new_imgs)


def Resize(frames: np.ndarray, size=(240, 240)):
    new_imgs = []
    for img in frames:
        new_imgs.append(cv2.resize(img, size))
    return np.array(new_imgs)


def transform_train(image):
    angle = transforms.RandomRotation.get_params([-90, 90])
    flip = False
    cont = False
    negative = False
    if random.random() > 0.5:
        flip = True
    if random.random() > 0.5:
        cont = True
    if random.random() > 0.5:
        negative = True
    delta = torch.randn(1)
    if negative:
        delta = -delta
    delta = delta / 10
    for i in range(image.shape[0]):
        img = Image.fromarray(
            np.uint8(np.transpose(image[i, :, :, :], (1, 2, 0))))
        if cont:
            img = tf.adjust_contrast(img, 2)
        if flip:
            img = tf.hflip(img)
        img = tf.rotate(img, angle)
        img = tf.to_tensor(img)
        img = tf.normalize(img, torch.mean(
            img, (1, 2)).tolist(), torch.std(img, (1, 2)).tolist())
        # img = tf.normalize(img, [0.27154714, 0.27154769, 0.27156396], [0.31989314, 0.31985731, 0.31968247])
        image[i, :, :, :] = img + delta
    return image


def transform_test(image):
    for i in range(image.shape[0]):
        img = Image.fromarray(
            np.uint8(np.transpose(image[i, :, :, :], (1, 2, 0))))
        img = tf.to_tensor(img)
        img = tf.normalize(img, torch.mean(
            img, (1, 2)).tolist(), torch.std(img, (1, 2)).tolist())
        # img = tf.normalize(img, [0.27089914, 0.27090737, 0.2709189], [0.32018263, 0.32015745, 0.31996976])
        image[i, :, :, :] = img
    return image


'''
Dataset for VST
'''


class VST_Loader(Dataset):
    """
    ## Load and decode Nifti dataset for Video Swin-Transformer CMR Diagnose/Screening Model
    No necessarity of giving the file paths of masks, only crop position supported, higher processing efficiency.

    Pipeline of data loading/preprocessing:
        - Load text annotation file
        - Load data <-> roi annotation hash map file
        - Decide augmentation pipelines
        - Data preprocession:
            - Read nifti format data
            - Crop ROI
            - Clip the maximum/minimum (defaultly 0.1%) voxel according their intensity values
            - Normalize data within [0,255] integer space(unsigned int8)
            - Augmentation
        - To Tensor

    Tips for short axis(SAX) cinema data:
        - Must have keywords: `mid`, `up`, `down` in every patients' sax cinema filenames, such as `114514_ZHANG_SAN/slice_up.nii.gz`, `114514_ZHANG_SAN/slice_mid.nii.gz`, `114514_ZHANG_SAN/slice_down.nii.gz`.
        - Slices' keyword should represent their physical position along the `z` axis, we recommand you to get it by `SimpleITK.Image.GetOrigin()[-1]`.
    Args:
        mask_ann: The path of the `nifti filepath <-> mask crop position(np.array([x_min,x_max,y_min,y_max]))` hash map, data structure: `dict()`, only support `.pkl` file.
    """

    def __init__(self, ann_file: str = None,
                 mask_ann: str = None,
                 fusion=False,
                 modalities: list = [],
                 augment_pipeline: list = [dict(type='HistEqual'),
                                           dict(type='SingleNorm'),
                                           dict(type='Padding',
                                                size=(120, 120)),
                                           dict(type='Resize', size=(240, 240))]):
        self.ann_file = ann_file
        self.mask_ann = mask_ann
        self.fusion = fusion
        # modalities: sax 4ch lge
        self.modalities = modalities
        self.nifti_info_list = self.__load_annotations()
        self.__load_mask_ann()
        self.augment_pipeline = augment_pipeline
        self.augment_methods = {
            'HistEqual': HistEqual,
            'SingleNorm': SingleNorm,
            'Padding': Padding,
            'Resize': Resize,
            'Random_rotate': Random_rotate,
            'Random_Gamma_Bright': Random_Gamma_Bright}

    def __load_mask_ann(self):
        if self.mask_ann is None:
            warnings.warn('Data will be loaded directly without cropping,\
                if Nifti data need to be cropped according to masks pls specify the annotation of masks')
        else:
            assert self.mask_ann.endswith(
                '.pkl'), f'Mask annotation file must contains the datapath-maskpath dictionary, and .pkl format file expected, but {self.mask_ann} given.'
            print(
                'Friendly reminding: please make sure the file path of modality SAX is the path of middle slice')
        with open(self.mask_ann, 'rb')as f:
            self.data_mask_map = pkl.load(f)
        f.close()
        files = list(self.data_mask_map.keys())
        for i in range(len(files)):
            if not os.path.exists(files[i]):
                self.data_mask_map.pop(files[i])
        print(
            f'{len(files)} mask_ann hash mapping given, {len(self.data_mask_map)} maps available')

    def __load_annotations(self):
        """Load annotation file to get nifti data information."""
        nifti_info_list = []
        with open(self.ann_file, 'r')as f:
            ann_info = f.readlines()
        f.close()
        for i in range(len(ann_info)):
            ann_info[i] = ann_info[i].split('\n')[0]
            data_info = {}
            if self.fusion:
                # Get file path of corresponding modality data
                for j in range(len(self.modalities)):
                    data_info[self.modalities[j]] = ann_info[i].split(' ')[j]
                    if not os.path.exists(data_info[self.modalities[j]]):
                        warnings.warn(
                            f"Line {j+1} at {self.ann_file} -> {data_info[self.modalities[j]]} doesn't exists")
                        continue
                data_info['label'] = int(ann_info[i].split(' ')[-1])
                nifti_info_list.append(data_info)
            else:
                data_info[self.modalities[0]] = ann_info[i].split(' ')[0]
                if not os.path.exists(data_info[self.modalities[0]]):
                    warnings.warn(
                        f"Line {j+1} at {self.ann_file} -> {data_info[self.modalities[0]]} doesn't exists")
                    continue
                data_info['label'] = int(ann_info[i].split(' ')[-1])
                nifti_info_list.append(data_info)
        return nifti_info_list

    def __crop(self, file_path: str, mod: str):
        if not file_path.endswith('.nii.gz'):
            file_path += '.nii.gz'
        if mod == 'sax':
            # mid_slice_num = int(file_path.split(
            #     '/')[-1].split('.nii.gz')[0].split('_')[-1])
            sax_mid = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
            sax_up = sitk.GetArrayFromImage(sitk.ReadImage(
                file_path.replace('mid', 'up')))
            sax_down = sitk.GetArrayFromImage(sitk.ReadImage(
                file_path.replace('mid', 'down')))

            if self.mask_ann is not None:
                # crop_pos = self.__get_crop_pos(file_path=file_path)
                crop_pos = self.data_mask_map[file_path]
                sax_mid = self.clip_top_bottom(
                    sax_mid[:, crop_pos[2]:crop_pos[3], crop_pos[0]:crop_pos[1]])
                sax_down = self.clip_top_bottom(
                    sax_down[:, crop_pos[2]:crop_pos[3], crop_pos[0]:crop_pos[1]])
                sax_up = self.clip_top_bottom(
                    sax_up[:, crop_pos[2]:crop_pos[3], crop_pos[0]:crop_pos[1]])
            try:
                sax_fusion = np.array([sax_up, sax_mid, sax_down])
            except:
                print(sax_up.shape, sax_down.shape, sax_mid.shape)
                print(file_path, '\n', file_path.replace('mid', 'up'),
                      '\n', file_path.replace('mid', 'down'))
                return 0
            sax_fusion = np.moveaxis(sax_fusion, 0, -1)
            return sax_fusion
        elif mod == '4ch' or mod == 'lge':
            data = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
            if self.mask_ann is not None:
                # crop_pos = self.__get_crop_pos(file_path=file_path)
                crop_pos = self.data_mask_map[file_path]
                data = self.clip_top_bottom(data[:, crop_pos[2]:crop_pos[3],
                                                 crop_pos[0]:crop_pos[1]])
            rgbdata = np.array([data]*3)
            rgbdata = np.moveaxis(rgbdata, 0, -1)
            return rgbdata

    def norm_range(self, data):
        return np.uint8(255.0*(data-np.min(data))/(np.max(data)-np.min(data)))

    def clip_top_bottom(self, data: np.ndarray, scale=0.001):
        arr = np.sort(data.flatten())
        size = len(arr)
        min_value = arr[int(scale * size)]
        max_value = arr[int((1 - scale) * size)]
        data[np.where(data < min_value)] = min_value
        data[np.where(data > max_value)] = max_value
        return self.norm_range(data=data)

    def __len__(self):
        return len(self.nifti_info_list)

    def __getitem__(self, index):
        results = {}
        if 'sax' in self.modalities:
            sax_fusion = self.__crop(
                file_path=self.nifti_info_list[index]['sax'], mod='sax')
            results['sax'] = sax_fusion
        if '4ch' in self.modalities:
            lax4ch_data = self.__crop(
                file_path=self.nifti_info_list[index]['4ch'], mod='4ch')
            results['4ch'] = lax4ch_data
        if 'lge' in self.modalities:
            lge_data = self.__crop(
                file_path=self.nifti_info_list[index]['lge'], mod='lge')
            results['lge'] = lge_data
        for mod in self.modalities:
            for augment in self.augment_pipeline:
                results[mod] = self.augment_methods[augment['type']](
                    results[mod], **augment.pop('type'))
            # NTHWC -> NCTHW
            results[mod] = torch.from_numpy(np.moveaxis(results[mod], -1, 1))
        label = torch.LongTensor([self.nifti_info_list[index]['label']])
        return results, label


'''
Dataset for CNNLSTM
'''


class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        if self.map:
            x = self.map(self.dataset[index][0])
        else:
            x = self.dataset[index][0]  # image
        y = self.dataset[index][1]  # label
        return x, y

    def __len__(self):
        return len(self.dataset)


class CNNLSTM_Loader(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
