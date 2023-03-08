# Customize `Dataset`

The operation of adding a customized `Dataset` is the same as what we do in PyTorch. However, the target of DeepMuon is to make deep-learning tasks easier and simpler, especially for non-CS major researchers. In this section we want to help you get familiar with the basic pipeline of using DeepMuon, so we omit technical details about the `Dataset`. If you want to read more information about `Dataset`, please go to the section [How to add augmentations?](https://airscker.github.io/DeepMuon/tutorials/index.html#/dataset/augment) and [What to do when adding a customized `Dataset` into DeepMuon?](https://airscker.github.io/DeepMuon/tutorials/index.html#/dataset/dataset)

To do interdisciplinary deep-learning tasks, the first thing is to make your dataset fully prepared. There are many famous datasets available on the internet, some of them are:
- [ImageNet](https://www.image-net.org/)

    > ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. The project has been instrumental in advancing computer vision and deep learning research. The data is available for free to researchers for non-commercial use. [1](#ref1)
- [EMPIAR](https://www.ebi.ac.uk/empiar/)

    > EMPIAR, the Electron Microscopy Public Image Archive, is a public resource for raw images underpinning 3D cryo-EM maps and tomograms (themselves archived in EMDB). EMPIAR also accommodates 3D datasets obtained with volume EM techniques and soft and hard X-ray tomography. [2](#ref2)

- [UK Biobank](https://www.ukbiobank.ac.uk/)

    > UK Biobank is a large-scale biomedical database and research resource, containing in-depth genetic and health information from half a million UK participants. The database is regularly augmented with additional data and is globally accessible to approved researchers undertaking vital research into the most common and life-threatening diseases. It is a major contributor to the advancement of modern medicine and treatment and has enabled several scientific discoveries that improve human health. [3](#ref3)

- [HTRU2](https://archive.ics.uci.edu/ml/datasets/HTRU2)

    > HTRU2 is a data set which describes a sample of pulsar candidates collected during the High Time Resolution Universe Survey (South). [4](#ref4)

- [DrugBank](https://www.drugbank.com/datasets)

    > Drug data is rapidly increasing In size, complexity, and inaccuracy. This is slowing down vital research and leaving us to rely on outdated practices. It is the most comprehensive, up-to-date, & accurate drug database on the market. Combining the advantages of AI and human expertise, DrugBank is able to source, validate, structure, and update our data every day. [5](#ref5)

The datasets shown above cover many different science subjects. We can choose a suitable dataset from the internet or create it by ourselves. In the DeepMuon, to start your first experiment, we decide to guide you using [FashionMNIST Dataset](https://pytorch.org/vision/stable/datasets.html#fashion-mnist).

## Download Dataset
You can download it manually or through the official codes given by PyTorch:

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

In DeepMuon we require the `Dataset` should be un-instantiated `Dataset` objects, the dataset shown above is an instance of basic class `Dataset` already. So to solve this problem, we can rewrite the code like this:

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class MINIST(Dataset):
    def __init__(self,root='root',download=True,train=True) -> None:
        super().__init__()
        if not os.path.exists(root):
            download=True
        self.download=download
        self.dataset=datasets.FashionMNIST(root=root,train=train,download=download,transform=ToTensor())
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index]
```

In this way, if we specify the name of the `Dataset`, that is, **MINIST**, it's just an index of customized `Dataset` and we can use it by the `getattr()` method. The reason why this requirement needs to be obeyed will be explained in the section [Dynamic importing mechanism](https://airscker.github.io/DeepMuon/tutorials/index.html#/config/import).

Up to now, you have already made your dataset prepared and the work of customizing the `Dataset` is completed. The next step is to create your own suitable model, please refer to the following section [Customize model](https://airscker.github.io/DeepMuon/tutorials/index.html#/start_exp/cus_model) to continue the learning.

## Bibliography
<p id='ref1'>[1] https://www.image-net.org/</p>
<p id='ref2'>[2] https://www.ebi.ac.uk/empiar/</p>
<p id='ref3'>[3] https://www.ukbiobank.ac.uk/</p>
<p id='ref4'>[4] https://archive.ics.uci.edu/ml/datasets/HTRU2</p>
<p id='ref5'>[5] https://www.drugbank.com/datasets</p>