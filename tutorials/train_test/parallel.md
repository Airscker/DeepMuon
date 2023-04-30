# Data Distributed Parallel & Fully Sharded Distributed Parallel

## Single GPU Training (SGT)

At the start of all multi-GPU training tutorials, we need to know the basic mechanism of single-GPU training mechanism. In the following, I will introduce the differences between CPU and GPU, the concept of Tensor, and basic knowledge of Input/Output flow (IO flow).

### CPU / GPU?

#### CPU: Central Processing Unit 

> Constructed from millions of transistors, the CPU can have multiple processing cores and is commonly referred to as the brain of the computer. It is essential to all modern computing systems as it executes the commands and processes needed for your computer and operating system. The CPU is also important in determining how fast programs can run, from surfing the web to building spreadsheets. [1](#1)

#### GPU: Graphics Processing Unit

> The GPU is a processor that is made up of many smaller and more specialized cores. By working together, the cores deliver massive performance when a processing task can be divided up and processed across many cores. [1](#1)

#### Why do we need GPU to train the Deep Learning models?

> CPUs and GPUs have a lot in common. Both are critical computing engines. Both are silicon-based microprocessors. And both handle data. But CPUs and GPUs have different architectures and are built for different purposes. [1](#1)
>
> Architecturally, the CPU is composed of just a few cores with lots of cache memory that can handle a few software threads at a time. In contrast, a GPU is composed of hundreds of cores that can handle thousands of threads simultaneously. [2](#2)
>
> Today, GPUs run a growing number of workloads, such as deep learning and artificial intelligence (AI). A GPU or other accelerators are ideal for deep learning training with neural network layers or on massive sets of certain data, like 2D images. Deep learning algorithms were adapted to use a GPU-accelerated approach. With acceleration, these algorithms significantly boost performance and bring the training time of real-world problems to a feasible and viable range. [1](#1)

### Tensor

Mathematically, tensors are defined as:

> An $n$ th-rank tensor in $m$-dimensional space is a mathematical object that has $n$ indices and $m^n$ components and obeys certain transformation rules. Each index of a tensor ranges over the number of dimensions of space. However, the dimension of the space is largely irrelevant in most tensor equations (with the notable exception of the contracted Kronecker delta). Tensors are generalizations of scalars (that have no indices), vectors (that have exactly one index), and matrices (that have exactly two indices) to an arbitrary number of indices. [3](#3)

Here I don't need you to understand the tensor algebra, and I believe you have learned linear algebra, so let me introduce the concept of Tensor in deep learning: A Tensor in deep learning is a data structure that has the same algebra standards with mathematical tensor, that is to say, the operations on Tensors in deep learning is the same as what we do to matrics in linear algebra.

Just as we have said before, the GPU has much more independent processing cores in three-dimensional space, they can simultaneously process algebra operations and the structure of mathematical tensor is similar to the GPU cores' distribution, which means that we can process tensor operations quickly based on the multi-processing ability of GPU. And that is the reason why we define the basic data structure of deep learning as tensors.

### IO

Input/Output flow (IO) is a basic computer science terminology, which describes the information passing courses during the operations of computers. If you have learned C++ you will be aware of IO, there are many types of IO in C++: RAM IO, file IO, hardware IO and etc. Whatever they are, they are all Input/Output related information flows.

Typically in deep learning, the data are read from the disk into RAM, if we directly operate on them, the operations are done via CPU, and if we want to use GPU to accelerate the training procedures, we need to define these data as Tensors(still in RAM), and put Tensors on GPU, after that, all operations would be done via GPU. But the data transferring from CPU to GPU is not at the speed of light regarding their size, the more data you transfer, the more times you transfer, and the more time will be wasted. So you would say, to reduce the time wasted we can put the entire Tensor dataset on GPU at once, right? But we also need to pay attention that rather than allocating extra space in disks for overstocked RAM, the size of GPU-RAM is strictly limited, so it's impossible for us to do this most of the time. That's why we need to pay attention to the batch size when we are going to train the deep learning models.

## Data Distributed Parallel (DDP)

To solve the GPU-RAM shortage problem, namely the extremely small batch size, the Data Distributed Parallel (DDP) was prompted. DDP copy model parameters for every GPU and different GPU process the same number of different samples, then their gradients were shared via inter-GPU teleportation. That is to say, here we are faced with another IO: inter-GPU data IO. Fortunately, nowadays with the development of technology most of the time, this is not an issue. To get more details about DDP, you can see the tutorial published by PyTorch: [Getting Started with Distributed Data Parallel.](https://www.bing.com/ck/a?!&&p=b3ac246838f855dfJmltdHM9MTY4MjgxMjgwMCZpZ3VpZD0wMjdjZDcyMC02NDFmLTYzNWQtMGIzMS1jNTQwNjU3OTYyYjUmaW5zaWQ9NTE3MQ&ptn=3&hsh=3&fclid=027cd720-641f-635d-0b31-c540657962b5&psq=distributed+data+para&u=a1aHR0cHM6Ly9weXRvcmNoLm9yZy90dXRvcmlhbHMvaW50ZXJtZWRpYXRlL2RkcF90dXRvcmlhbC5odG1s&ntb=1)

## Fully Sharded Distributed Parallel (FSDP)

You may say what if the model parameter tensor cannot be stored in GPU either? In this way, the DDP can no longer copy model parameters for every GPU, right? Yes, to solve this problem, FAIR prompted FSDP algorithm, which split model parameters into different GPUs and every GPU only refreshes a part of the model parameters in one training epoch. But no matter how the model parameters are distributed, refreshing them needs gathering gradients into one place, which means, there are much more inter-GPU IOs than DDP, and unfortunately not all of us have enough budgets to equip fast enough deep learning servers. To get more details of FSDP, please refer to: [Getting Started with Fully Sharded Data Parallel (FSDP).](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

## SGT/DDP/FSDP in DeepMuon

DeepMuon supports all three kinds of model training architectures, but we recommend using DDP because it can greatly improve the training speed and reduce the time used. And its IO is affordable most time. Here is an example of a DDP training/testing command:

```bash
Dmuon_train -g 0 1 -p 20224 -c config.py -tr dist_train -ts checkpoint.pth -sr
```

Here is the explanation of command parameters:

- `Dmuon_train` indicates the training/testing mode.
-  `-g` or `--gpus` indicates the IDs of the GPUs to be used to train/test the model.
-  `-p` or `--port` indicates the port used for DDP/FSDP IO teleportation.
-  `-c` or `--config` indicates the path of the configuration file of the experiment.
- `-tr` or `--train` indicates the training/testing pipeline file to be used, you can add the `.py` suffix or omit it, the pipeline file must be placed at `~/DeepMuon/train/`.
- `-ts` or `--test` indicates the checkpoint to be used to test the model performance, it's the path of the checkpoint file saved during training. **If you omit this parameter, the experiment will enter training mode automatically.**
- `-sr` or `--search` indicates whether to enable the Neural Network Hyperparameter Searching (NNHS) system, omitting this will automatically disable NNHS, and NNHS won't take action when `-ts`/`--test` parameter is valid (that is testing mode is enabled). For more details on NNHS please refer to [Neural Network Hyperparameter Searching (NNHS).](https://airscker.github.io/DeepMuon/tutorials/index.html#/train_test/nnhs)

If you want to start SGT, you just need to specify one GPU ID. And to enable the FSDP, you need to set its configurations in the experiment's configuration file, also you need to make sure the edition of PyTorch installed supports FSDP (>=1.12). For more details on modifying FSDP please refer to [Elements of config.py.](https://airscker.github.io/DeepMuon/tutorials/index.html#/config/config?id=fsdp_parallel)



## Bibliography

<p id='#1'>[1]: https://www.intel.com/content/www/us/en/products/docs/processors/cpu-vs-gpu.html</p>
<p id='#2'>[2]: https://blogs.nvidia.com/blog/2009/12/16/whats-the-difference-between-a-cpu-and-a-gpu/</p>
<p id='#3'>[3]: https://mathworld.wolfram.com/Tensor.html</p>