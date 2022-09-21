# Tutorial of DeepMuon

Here in the frame of **DeepMuon**, you can create your customized `Model `s, `Loss function`s, as well as any other customized `Dataset`s

To use the DeepMuon training frame, there are some regulations you need to follow:

> 1. DeepMuon support **Single GPU** training and **Distributed Data Parallel** training
> 2. All customized **Deep Learning Models** and **Layers** must be stored into file folder `models`
> 3. All customized **Loss Functions** must be stored in file `models.Airloss.py`
> 4. All customized **Dataset** classes must be stored into file folder `dataset`
> 5. All **Training Configuration** files must be stored into file folder `config`
> 6. **Training Command** must be typed under the path of file folder`DeepMuon`
> 7. **Original data and work_dirs** are not permitted to be stored into file folder `DeepMuon` 

## Single GPU Training - train.py

### Command

```bash
python train.py --config /home/dachuang2022/Yufeng/DeepMuon/config/Hailing/MLP3_3D.py
```

### Introduction

`train.py` supports Single GPU training, the log files, tensorboard event files and checkpoints will be saved in `work_dir` specified in configuration file.

## Distributed Data Parallel Training - dist_train.py

### Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 22921 dist_train.py --config /home/dachuang2022/Yufeng/DeepMuon/config/Hailing/MLP3_3D.py
```

### Introduction

`dist_train.py` supports Distributed Data Parallel training, the log files, tensorboard event files and checkpoints will be saved in `work_dir` specified in configuration file.

## Customize Dataset

To create your own customized dataset, you need to **STRICTLY** follow these regulations:

> 1. Dataset classes and other scripts are stored in python files.
>
> 1. `__init__(self,datapath):`
>
>    The `__init__()` **only support one parameter which is the datapath to be loaded**.
>
>    You may claim that it may be not sufficient, but I will tell you that the need of sufficiency comes from your uselessness, as for a stupid researcher, whatever quantities of parameters will satisfy him
>
> 2. Except for the Dataset class, all other preprocess functions or classes are recommended to save with Dataset in the same file, you know, researches are not only challenges to our intelligence but also to our taste of beauty and tidy.
>
> 3. **You must commit your changes to git log every time you finished your fix**
>
>    ```bash
>    git add .
>    git commit -m 'anything you want to say'
>    ```
>    
> 5. **DO NOT ADD** any executable console scripts into Dataset file, for instance: `print()`,`plt.show()`... all kinds of executable console scripts all forbidden in the file.
>
> 4. **You need to refresh the** `__init__.py` after you add your customized Dataset to make sure `import` works normal.

### Create Your Dataset

To create your Dataset, you need to create a single python file under the path of folder `dataset` such as `dataset/exampledata.py`

And all Dataset class are stored within your Dataset file `dataset/exampledata.py`, Here we give you an example about the customized Dataset class

```python
from torch.utils.data import Dataset
import torch
import numpy as np
class HailingDataset_Pos(Dataset):
    def __init__(self,datapath=''):
        '''
        ## Dataset Example
        - Args:
            - datapath: The datapth of the data, default to be ''
        - Output:
            - Image, shape: [], dtype: nparray -> torch.tensor, other informations
            - Label, shape: [], dtype: nparray -> torch.tensor, other informations
        '''
        self.datapath=datapath
        self.data=self.__Init()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        image=torch.from_numpy(self.data[index]['img'])
        label=torch.from_numpy(self.data[index]['label'])
        return image,label
    def __Init(self):
        with open(self.datapath,'r')as f:
            data=f.readlines()
        return data
```

### Other Preprocessing Scripts

For example, you can add other preprocess functions or classes in the same file (example from `HailingData.py`):

```python
import numba
import numpy as np
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
```

## Customize Models

Before you create your DL Models, you need to **STRICTLY** follow these regulations:

> 1. Models are saved in python files
>
> 2. You must obey the basic regulation given by **Pytorch** to write your customized models.
>
> 3. Customized Layers are also saved in the same file of models
>
> 4. **You must commit your changes to git log every time you finished your fix**
>
>    ```bash
>    git add .
>    git commit -m 'anything you want to say'
>    ```
>
> 5. **DO NOT ADD** any executable console scripts into Dataset file, for instance: `print()`,`plt.show()`... all kinds of executable console scripts all forbidden in the file.
>
> 6. **You need to refresh the** `__init__.py` after you add your customized Dataset to make sure `import` works normal.
>
> 7. **Customized Loss Functions are stored in** `models/Airloss.py`

### Create Your Model

To create your Model, you need to create a single python file under the path of folder `models` such as `models/examplemodel.py`

And all Dataset class are stored within your Dataset file `models/examplemodel.py`, Here we give you an example about the customized Dataset class from `models/Pandax4T.py`:

```python
import torch
from torch import nn
class CONV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(5,5),stride=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=10,out_channels=40,kernel_size=(5,5)),
            nn.BatchNorm2d(40),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=40,out_channels=80,kernel_size=(5,5)),
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):   
        x=self.conv(x)
        x=self.flatten(x)
        x = self.linear_relu_stack(x)
        return x
```

### Customized Layer

Customized layers are saved in the same file of your model, here is an example from `models/Hailing.py`:

```pytho
from torch import nn
import torch.nn.functional as F
class HailingDirectNorm(nn.Module):
    def __init__(self) -> None:
        '''
        ## Customized Layer, Normalize the Direction Vector of Hailing Data Derived from _Direct Models
        - Input: [N,3], info: [px,py,pz]
        - Output: [N,3], info: [px,py,pz](Normalized)

        N is the batch size, and the output direction vector is normalized to 1
        '''
        super().__init__()
    def forward(self,x):
        return F.normalize(x)
```

### Customized Loss Function

Customized Loss Functions must be saved in `models/Airloss.py`, here is an example:

```python
import torch
from torch import nn
from torch import Tensor
import numpy as np
class MSALoss(nn.Module):
    def __init__(self,angle_ratio=1):
        """## MSEloss(vec1,vec2)+Angle(vec1,vec2)
        - Args:
            - angle_ratio (int, optional): The ratio to consider the angle loss into total loss. Defaults to 1.
        """
        super().__init__()
        self.angle_ratio=angle_ratio
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # print(input)
        mseloss=(input-target)**2
        mseloss=torch.sum(mseloss)/(mseloss.shape[0]*mseloss.shape[1])
        # angloss=ang(input=input,target=target)``
        angloss=angn(input=input,target=target)
        return mseloss+self.angle_ratio*angloss
def angn(input,target):
    input=input.detach().cpu().numpy()
    target=target.detach().cpu().numpy()
    res=np.zeros(input.shape[0])
    for i in range(input.shape[0]):
        res[i]=np.dot(input[i],target[i])/(np.sqrt(np.sum(input[i]**2)*np.sum(target[i]**2)))
    res=np.mean(np.arccos(res))
    return torch.from_numpy(np.array(res))
```

## Configuration Files

Here we introduce another core component of **DeepMuon**: Configuration Files.

Configuration files are used to eliminate the modification of `train.py`,`dist_train.py`, which are core of training, we can make sure every researcher is smart enough to understand the logic of those core files, we want to provide a simple and direct experience of customizing training configurations, the DeepMuon training frame will automatically recognize the configurations specified within configuration files and apply them into training.

All configuration files are stored in the folder `config`, before you create your configuration files, you must **STRICTLY** follow these rules:

> 1. **One Project, One Folder**
>
>    Here we know that there are several different projects are waiting for us. And during exploration of one project we have to try different configurations several times, so we need to make sure our management of projects and configurations of one project is tidy, direct, clear and even beautiful. Just as what I said before, researches are not only challenges to our intelligence but also to our taste of beauty and tidy.
>
> 2. **Every time new configuration files added, file `__init__.py` under the project folder must be refreshed**
>
> 3. **The edition of configuration files must follow the regulations given in the section `Configuration File Regulations`**
>
> 4. **You must commit your changes to git log every time you finished your fix**
>
>    ```bash
>    git add .
>    git commit -m 'anything you want to say'
>    ```
>
> 5. **DO NOT ADD** any executable console scripts into Dataset file, for instance: `print()`,`plt.show()`... all kinds of executable console scripts all forbidden in the file.

### Configuration File Regulations

1. **All these keywords must be presented in a configuration file:**

   - model
   - train_dataset
   - test_dataset
   - work_config
   - checkpoint_config
   - loss_fn
   - hyperpara
   - lr_config
   - gpu_config

2. **Regulations of keywords:**

   - Specify the model to be trained: 
   
     `model=dict(backbone='MLP3_3D_Direc')`

     - backbone: specify the model name, model will be picked up in the folder `models`
   
   - Specify the training dataset to be used to load the data, all dataset are stored in `dataset`: 
   
       `train_dataset=dict(backbone='HailingDataset_Direct',datapath='/home/dachuang2022/Yufeng/Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl')`
   
       - backbone: specify the name of Dataset class, Dataset will be loaded from folder `dataset`
       - datapath: specify the path of data, absolute path needed
   
   - Specify the testing dataset to be used to load the data, all dataset are stored in `dataset`: 
   
       `test_dataset=dict(backbone='HailingDataset_Direct',datapath='/home/dachuang2022/Yufeng/Hailing-Muon/data/1TeV/Hailing_1TeV_val_data.pkl')`
       
       - backbone: specify the name of Dataset class, Dataset will be loaded from folder `dataset`
       - datapath: specify the path of data, **absolute path needed**
       
   - Specify the work_dir to save the training log and checkpoints
       `work_config=dict(work_dir='/home/dachuang2022/Yufeng/Hailing-Muon/work_dir/1TeV/MLP3_3D',logfile='log.log')`
       
       - work_dir: the folder used to store the training logfile, tensorboard event foler `LOG`, and checkpoints. **Absolute path needed**
       - logfile: the name of the training logfile
       
   - Specify the checkpoint configuration
       `checkpoint_config=dict(load_from='',resume_from='',save_inter=10)`
       
       - load_from: the path of the pretrained `.pth` file to be loaded, model will be trained from epoch 0
       - resume_from: the path of pretrained `.pth` file to be used to resume the model training
       - save_inter: specify the checkpoint saving frequency
       
   - Specify the customized loss function to be used, if no customized loss function specified, nn.MSELoss() will be used
     
       If `loss_fn=None` specified, `nn.MSELoss()` will be used to train the model, otherwise `loss_fn=dict(backbone='')`
       
       - backbone: the name of the loss function created in file `Airloss.py`
       
   - Specify the Hyperparameters to be used
       `hyperpara=dict(epochs=10,batch_size=400,inputshape=[1,10,10,40,3])`
       
       - epochs: the training epochs
       - batch_size: the training batch_size
       - inputshape: the shape of the model input data, first element is the batch_size(here is 1), and the left elements are actual data shape
       
   - Specify the lr as well as its config, the lr will be optimized using `torch.optim.lr_scheduler.ReduceLROnPlateau()`
       `lr_config=dict(init=0.0005,patience=100)`
       
       - init: the initial learning rate
       - patience: the patience epochs used to judge the learning rate dacay status
       
   - Specify the GPU config and DDP
       `gpu_config=dict(distributed=True,gpuid=0)`
       
       - distributed:
         - `True`: DDP will be used to train the model, at this time, you must use `dist_train.py` to start the experiment.
         - `False`: Single GPU Training will be used, at this time, you must use `train.py` to start the experiment
       - gpuid: this parameter only have effects in Single GPU Training, it specify the GPU to be used in the experiment.
