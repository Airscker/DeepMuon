'''
Author: airscker
Date: 2023-01-27 19:51:21
LastEditors: airscker
LastEditTime: 2023-04-04 00:19:32
Description:
    ## Dataset built for:
        - Video Swin-Transformer (VST) CMR Screening & Diagnose Model
        - CNNLSTM CMR Screening & Diagnose Model

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''
import os
import cv2
import warnings
import numpy as np
import pickle as pkl
import SimpleITK as sitk
from skimage import exposure

import torch
from torch.utils.data import Dataset


def exclude_key(dictionary: dict, del_key: str = 'type'):
    '''
    Delete key-value map from dictionary
    '''
    new_dict = {}
    for key in dictionary.keys():
        if key != del_key:
            new_dict[key] = dictionary[key]
    return new_dict


def Batch_norm(frames: np.ndarray, mean, std):
    """
    ## Normalize nifti data by specified mean value and std value

    ### Args:
        - frames: the numpy array data of nifti file, input shape should be `THWC`
        - mean: the mean value
        - std: the standard derivate value

    ### Return:
        - the nifti data augmented
    """
    return (frames-mean)/std



def HistEqual(frames: np.ndarray):
    """
    ## Histogram equalization for multi-frames nifti data

    ### Args:
        - frames: the numpy array data of nifti file, input shape should be `THWC`

    ### Return:
        - the nifti data augmented
    """
    for i in range(len(frames)):
        img = frames[i]
        for j in range(img.shape[-1]):
            img[:, :, j] = cv2.equalizeHist(np.uint8(img[:, :, j]))
        img=img.astype(np.uint8)
        frames[i] = img
    return frames


def SingleNorm(frames: np.ndarray):
    """
    ## Normalize nifti data by `(data-mean(data))/std(data)`

    ### Args:
        - frames: the numpy array data of nifti file, input shape should be `THWC`

    ### Return:
        - the nifti data augmented
    """
    return (frames-np.mean(frames))/np.std(frames)


def Random_rotate(frames: np.ndarray, range=180, ratio=0.3):
    """
    ## Rotate given nifti data in XY dimension within epecified rotation range with specified probability

    ### Args:
        - frames: the numpy array data of nifti file, input shape should be `THWC`
        - range: the range where rotation is allowed
        - ratio: the probability of rotation

    ### Return:
        - the nifti data augmented
    """
    if np.random.rand() < ratio:
        rows, cols = frames.shape[1:3]
        Matrix = cv2.getRotationMatrix2D(
            (cols/2, rows/2), (2*np.random.rand()-1)*range, 1)
        for i in range(len(frames)):
            frames[i] = cv2.warpAffine(
                frames[i], Matrix, (rows, cols), borderValue=(255, 255, 255))
    return frames


def Random_Gamma_Bright(frames: np.ndarray, ratio=0.5, low=0.4, high=1):
    """
    ## Adjust the brightness of nifti data within specified range with spcified probability

    ### Args:
        - frames: the numpy array data of nifti file, input shape should be `THWC`
        - ratio: the probability of adjustment
        - low: the minimal allowed brightness ratio compared to the oringinal data
        - high: the maximal allowed brightness ratio compared to the oringinal data

    ### Return:
        - the nifti data augmented
    """
    if np.random.rand() < ratio:
        for i in range(len(frames)):
            frames[i] = exposure.adjust_gamma(
                frames[i], np.random.uniform(low, high))
    return frames


def AddRandomNumber(frames: np.ndarray, range=0.1):
    """
    ## Adjust the brightness of nifti data within specified range with spcified probability

    ### Args:
        - frames: the numpy array data of nifti file, input shape should be `THWC`
        - range: the range of the noise value

    ### Return:
        - the nifti data augmented
    """
    noise = (2*np.random.rand(frames.shape)-1)*range
    return frames+noise


def Padding(frames: np.ndarray, size=(120, 120)):
    """
    ## Pad nifti data ROI with zero value to specified XY dimension size

    ### Args:
        - frames: the numpy array data of nifti file, input shape should be `THWC`
        - size: the size of the padding target, missed voxel data will be supplemented by zero and the oversized part will be deleted

    ### Return:
        - the nifti data augmented
    """
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
    """
    ## Resize nifti data to specfied XY dimension size

    ### Args:
        - frames: the numpy array data of nifti file, input shape should be `THWC`
        - size: the XY dimension size of the ultimate data

    ### Return:
        - the nifti data augmented
    """
    new_imgs = []
    for img in frames:
        new_imgs.append(cv2.resize(img, size))
    return np.array(new_imgs)


class NIIDecodeV2(Dataset):
    """
    ## Load and decode Nifti dataset for Video Swin-Transformer/CNNLSTM CMR Diagnose/Screening Model
    No necessarity of giving the file paths of masks, only crop position supported, higher processing efficiency.

    ### Pipeline of data loading/preprocessing:
        - Load text annotation file
        - Load data <-> roi annotation hash map file
        - Decide augmentation pipelines
        - Data preprocession:
            - Read nifti format data
            - Crop ROI
            - Clip the maximum/minimum (defaultly 0.1%) voxel according their intensity values
            - Normalize data within [0,255] integer space(unsigned int8)
            - Augmentation
        - To Tensor(NTHWC -> NCTHW)

    ### Tips for short axis(SAX) cinema data:
        - Must have keywords: `mid`, `up`, `down` in every patients' sax cinema filenames, such as `114514_ZHANG_SAN/slice_up.nii.gz`, `114514_ZHANG_SAN/slice_mid.nii.gz`, `114514_ZHANG_SAN/slice_down.nii.gz`.
        - Slices' keyword should represent their physical position along the `z` axis, we recommand you to get it by `SimpleITK.Image.GetOrigin()[-1]`.

    ### Args:
        - mask_ann: The path of the `nifti filepath <-> mask crop position(np.array([x_min,x_max,y_min,y_max]))` hash map, data structure: `dict()`, only support `.pkl` file.
        - fusion: Whether to train fusion model
        - modalities: The list of modality of datasets, avilable modalitiesare `sax`,`4ch`,`lge`.
            - eg. `modalities=['sax','4ch']`
        - model: Whether build dataset for `LSTM`, if not, just ignore this parameter otherwise set it as `LSTM`
        - frame_interval: The interval of frames used to resample cinema nifti data
        - augment_pipeline: The list of augmentation function names as well as their parameters.
            - eg. `augment_pipeline = [dict(type='HistEqual'),dict(type='SingleNorm'),dict(type='Padding',size=(120, 120)),dict(type='Resize', size=(240, 240))]`
    """

    def __init__(self, ann_file: str = None,
                 mask_ann: str = None,
                 fusion=False,
                 modalities: list = ['4ch'],
                 model=None,
                 frame_interval=2,
                 augment_pipeline: list = [dict(type='HistEqual'),
                                           dict(type='SingleNorm'),
                                           dict(type='Padding',
                                                size=(120, 120)),
                                           dict(type='Resize', size=(240, 240))]):
        self.model = model
        self.frame_interval = frame_interval
        self.ann_file = ann_file
        self.mask_ann = mask_ann
        self.fusion = fusion
        # modalities: sax 4ch lge
        self.modalities = modalities
        self.nifti_info_list = self.__load_annotations()
        self.__load_mask_ann()
        self.augment_pipeline = augment_pipeline

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
        print(
            f'To improve the performance of {self.__class__.__name__}, we deprecated ROI & Datapath checking pipeline. Please make sure the content of Datapath-ROI hash maps is correct before you start training models')

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
                            f"Line {i+1} at {self.ann_file} -> {data_info[self.modalities[j]]} doesn't exists")
                        continue
                data_info['label'] = int(ann_info[i].split(' ')[-1])
                nifti_info_list.append(data_info)
            else:
                data_info[self.modalities[0]] = ann_info[i].split(' ')[0]
                if not os.path.exists(data_info[self.modalities[0]]):
                    warnings.warn(
                        f"Line {i+1} at {self.ann_file} -> {data_info[self.modalities[0]]} doesn't exists")
                    continue
                data_info['label'] = int(ann_info[i].split(' ')[-1])
                nifti_info_list.append(data_info)
        return nifti_info_list

    def __crop(self, file_path: str, mod: str):
        '''
        Pipeline:
            - Read nfiti data according to annotation file
            - Resample numpy array of nifti data by specified frame interval(just for cinema data)
            - Crop ROI(If `mask_ann` given)
            - Clip top/bottom 0.1% voxel values
            - Concatenate three layers for short axis cinema nifti data / Repeat single layer three times for four chamber cinema data or LGE data
        '''
        if not file_path.endswith('.nii.gz'):
            file_path += '.nii.gz'
        if mod == 'sax':
            sax_mid = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
            sax_up = sitk.GetArrayFromImage(sitk.ReadImage(
                file_path.replace('mid', 'up')))
            sax_down = sitk.GetArrayFromImage(sitk.ReadImage(
                file_path.replace('mid', 'down')))
            sax_mid = sax_mid[0:len(sax_mid):self.frame_interval]
            sax_up = sax_up[0:len(sax_up):self.frame_interval]
            sax_down = sax_down[0:len(sax_down):self.frame_interval]
            if self.mask_ann is not None:
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
            if mod != 'lge':
                data = data[0:len(data):self.frame_interval]
            if self.mask_ann is not None:
                crop_pos = self.data_mask_map[file_path]
                data = self.clip_top_bottom(data[:, crop_pos[2]:crop_pos[3],
                                                 crop_pos[0]:crop_pos[1]])
            rgbdata = np.array([data]*3)
            # CTHW -> THWC
            rgbdata = np.moveaxis(rgbdata, 0, -1)
            return rgbdata

    def norm_range(self, data):
        '''Normalize voxel data values within 0~255 integer range'''
        return np.uint8(255.0*(data-np.min(data))/(np.max(data)-np.min(data)))

    def clip_top_bottom(self, data: np.ndarray, scale=0.001):
        '''Clip top/bottom voxel values with specific proportion'''
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
        data = []
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
        env = globals()
        for i in range(len(self.modalities)):
            mod=self.modalities[i]
            for augment in self.augment_pipeline:
                try:
                    if augment['type']=='Batch_norm' and len(self.modalities)>1:
                        results[mod] = env['Batch_norm'](
                            results[mod], mean=augment['mean'][i],std=augment['std'][i])
                    else:
                        results[mod] = env[augment['type']](
                            results[mod], **exclude_key(augment))
                except:
                    pass
            if self.model != 'LSTM':
                # THWC -> CTHW
                results[mod] = torch.from_numpy(
                    np.moveaxis(results[mod], -1, 0)).type(torch.FloatTensor)
            elif self.model == 'LSTM':
                # THWC -> TCHW
                results[mod] = torch.from_numpy(
                    np.moveaxis(results[mod], -1, 1)).type(torch.FloatTensor)
            data.append(results[mod])
        label = torch.LongTensor([self.nifti_info_list[index]['label']])
        if self.model != 'LSTM':
            if self.fusion:
                for i in range(len(data)):
                    data[i] = data[i].unsqueeze(0)
                return torch.cat(data, dim=0), label
            else:
                return data[0], label
        elif self.model == 'LSTM':
            return data[0], label
