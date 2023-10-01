# Accelerating large dataset loading speed via `MultiThreadLoader`

When we are dealing with large dataset such as containing over 1M single samples, the IO bound is always an issue for data loading. In oder to solve this problem systematically, we implemented multi-thread data loading algorithm `MultiThreadLoader` to help you with that. The usage of `MultiThreadLoader` is very simple:

```python
datasamples=MultiThreadLoader(LoadList:list[Any],ThreadNum:int,LoadMethod:Callable)
```

- LoadList: the file list of data samples, such as `[img_0001.jpg, img_0002.jpg,...]`.
- ThreadNum: number of threads to be used for data loading, such as `10`.
- LoadMethod: method used to load single data sample, such as `cv2.imread`.

This function returns a list of data samples, in which every element corresponds to the data loaded by the corresponding path in `LoadList`.

Here is an example which shows how to use `MultiThreadLoader` in `Dataset`:

```python
import os
import cv2
from DeepMuon.tools import MultiThreadLoader
from torch.utils.data import Dataset
class mydataset(Dataset):
    def __init__(self,data_folder):
	filelist=os.listdir(data_folder)
	self.dataset=MultiThreadLoader(filelist,10,cv2.imread)
    def __getitem__(self,index):
	return self.dataset[index]
    def __len__(self):
	return len(self.dataset)
```
