# What to do when adding a customized `Dataset` into DeepMuon?

DeepMuon is built based on PyTorch, and its grammar is the same. Here we give out some small tips that need to be paid attention to when you are creating your `Dataset`:

- Path of `Dataset` saved:

  The path of `Dataset` can be the installation position of DeepMuon, or anywhere else. We recommend you keep it under the installation path of DeepMuon, within the folder `dataset`, be the way, you also need to add the name of `Dataset` into the file `__init__.py` under the folder `dataset`. 

  However, if you put it anywhere else, you need to specify its path in the configuration, otherwise, errors will occur when loading your configuration. To add the `filepath` parameter of your `Dataset` into the configuration, please refer to [Elements of `config.py`](https://airscker.github.io/DeepMuon/tutorials/index.html#/config/config).

- Grammar of `Dataset`:

  All `Dataset`s must be inherited from base class `torch.utils.data.Dataset`, and in the `Dataset`, we must specify the detailed methods `__len()__` and `__getitem()__`, let's see an example comes from [TRIDENT Neutrino Telescope (Hailing Plan) Tracing Task](https://airscker.github.io/DeepMuon/blogs/index.html#/trident/trident?id=trident-neutrino-telescope-hailing-plan-tracing-task):

  ```python
  class HailingDataset_Direct2(Dataset):
      '''
      ## Dataset Built for Loading the Preprocessed Hailing 1TeV/10TeV Data, origial data shape: [10,10,40/50,3]
  
      ### Args: 
          - datapath: The datapth of the preprocessed Hailing data, default to be './Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl'
  
      ### Output:
          - Pattern Image, shape: [3,10,10,40/50], dtype: nparray -> torch.tensor
          - Position-Direction, shape: [3,], dtype: nparray -> torch.tensor, info: [px,py,pz]
      '''
  
      def __init__(self, datapath='./Hailing-Muon/data/1TeV/Hailing_1TeV_train_data.pkl', augment=False):
  
          self.datapath = datapath
          self.origin_data = None
          self.pattern_imgs = []
          self.pos_direction = []
          self.augmentation = {0: Rotate180, 1: Rotate90, 2: Flip, 3: Same}
          self.augment = augment
          self.__Init()
  
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
          image = torch.from_numpy(image.copy()).type(torch.FloatTensor)
          image = torch.permute(image, (3, 0, 1, 2))
          image[0,:,:,:]=(image[0,:,:,:]-torch.min(image[0,:,:,:]))/(torch.max(image[0,:,:,:])-torch.min(image[0,:,:,:]))
          mat_range=torch.max(image[1,:,:,:])-torch.min(image[1,:,:,:])
          image[1,:,:,:]=(image[1,:,:,:]-torch.min(image[1,:,:,:]))/mat_range
          image[2,:,:,:]=image[2,:,:,:]/mat_range
          label = torch.from_numpy(label).type(torch.FloatTensor)
          return image, label
  
      def __Init(self):
          print(f'Loading dataset {self.datapath}')
          with open(self.datapath, 'rb')as f:
              self.origin_data = pkl.load(f)
          f.close()
          print(f'Dataset {self.datapath} loaded')
  
  ```

  Just as the code shown above, `__len()__` returns the length of our training/testing dataset, it represents the number of available data samples which to be used in the training/testing procedures, the hyperparameter `batch_size` and `sub_batch_size` will be calculated according to the number of data samples.

  And `__getitem()__` returns the data and its label (most of the time, you can make it returns other data too), and it must have a parameter `index`, we return data samples according to their indexes. In the function `__getitem()__` you can specify the data augmentation pipelines or data preprocessing pipelines, in the example given above, We normalized the three channels of the TRIDENT dataset with different methods and applied data augmentation (if allowed). You can also put the preprocessing/augmentation pipelines anywhere else, just like the function `__Init()`, which loads the whole dataset at one time to reduce the time consumption brought by repeated IO operations(if we load data one by one).

- No unpackaged scripts allowed

  That is to say, we strongly don't recommend the usage of unpackaged runnable scripts in the file, for example, `print('XXX')` located at the first line of the file. This will result in ugly console output every time you import DeepMuon or its sub-modules.