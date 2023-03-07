# TRIDENT Neutrino Telescope (Hailing Plan) Tracing Task


> The tRopIcal DEep-sea Neutrino Telescope, **TRIDENT** for short, is an envisioned neutrino observatory at the South China Sea, which aims to discover multiple high-energy astrophysical neutrino sources and provide a significant boost to the measurement of cosmic neutrino events of all flavors. The telescope will have an uneven configuration with more than 1,000 strings and 20,000 hybrid digital optical modules covering a volume of about 8 cubic-kilometer. The optical properties of the seawater marine environment at the selected site were characterized by the TRIDENT pathfinder experiment in September 2021. Currently, the prototype line for TRIDENT is under development.[1](#ref1)

## Introduction

Cosmic rays from deep space constantly bombard the Earth’s atmosphere, producing copious amounts of GeV - TeV neutrinos via hadronic interactions. Similar processes yielding higher energy (TeV -PeV) neutrinos are expected when cosmic rays are accelerated and interact in violent astrophysical sources, such as in jets of active galactic nuclei (AGN)[2](#ref2). Ultra-high-energy cosmic rays (UHECRs) traversing the Universe and colliding with the cosmic microwave background photons are predicted to generate ‘cosmogenic’ neutrinos (beyond EeV)[3](#ref3). Detecting astrophysical neutrino sources will therefore be the key to deciphering the origin of the UHECRs.

However, neutrinos cannot be detected directly. These ‘ghostly’ particles are measured using extremely sensitive technologies, detecting the charged particles generated in neutrino-matter interactions. In a general detector setup, large areas of photon sensors continuously monitor a large body of target mass, e.g. pure water, liquid scintillator, liquid argon, to measure these rare and tiny energy depositions. Neutrino telescopes use large volumes of wild sea/lake water or glacial ice to observe the low rate of incident high-energy astrophysical neutrinos.[4](#ref4)

Based on the theoretical calculation and detailed design, the next-generation neutrino telescope - The tRopIcal DEep-sea Neutrino Telescope, TRIDENT for short, was built in the South China Sea, which aims to discover multiple high-energy astrophysical neutrino sources and provide a significant boost to the measurement of cosmic neutrino events of all flavors.[4](#ref4)

TRIDENT involves many basic tasks to localize and uncover the mystery of neutrinos, one of the basic steps is to uncover the tracks of neutrinos within the TRIDENT. Traditional methods include molecule dynamic simulation and linear regression. However, these methods are complex and expensive when considering computing resources and time. To enable the real-time end-to-end estimation of tracks of neutrinos, we proposed a deep-learning-based method to rebuild the tracks of neutrinos, in order to help analyze the physical principles and mysteries of cosmic rays. 

To have a complete vision of our contribution, we first introduced the basic data structure from the structure of TRIDENT to the collection of signals. Then we show the architecture of the deep-learning model used to trace the tracks. The detailed training technology and pipeline are also presented. Finally, we demonstrated the capability and interpretation of our model.

## Data Structure
### Structure of TRIDENT [1](#ref1), [4](#ref4)

<center>
    <table style="border-collapse:collapse;border-spacing:0" class="tg" frame=void rules=none>
    <thead>
        <tr>
            <th style="font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><img src='https://github.com/Airscker/DeepMuon/blob/site/blogs/trident/trident_design.png?raw=true' height='300px'></th>
            <th style="font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><img src='https://github.com/Airscker/DeepMuon/blob/site/blogs/trident/trident_string.jpg?raw=true' height='300px'></th>
            <th style="font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><img src='https://github.com/Airscker/DeepMuon/blob/site/blogs/trident/HDOM.jpg?raw=true' height='300px'></th>
    </thead>
    <tbody>
        <tr>
            <td style="font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">(a)</td>
            <td style="font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">(b)</td>
            <td style="font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">(c)</td>
        </tr>
    </tbody>
</table>
<p align='left'><b>(a). Geometry layout of the TRIDENT array.</b> The pattern follows a Penrose tiling distribution. Each black dot represents a string of length of ∼ 0.7 km, while the dashed lines mark the paths for underwater maintenance robots. <b>(b). Single string of TRIDENT.</b> The preliminary layout of the envisioned telescope follows a Penrose tiling distribution with twofold inter-string distances, 70 m and 110 m. The full detector is composed of 1211 strings, each containing 20 hDOMs separated vertically by 30 m. <b>(c). Hybrid Digital Optical Module (hDOM).</b> TRIDENT will employ a hybrid digital optical module (hDOM) with both PMTs and SiPMs. On the one hand, SiPMs can respond to photon hit within tens of picoseconds. On the other, the TDCs are capable of digitizing the sharp raising edge of a SiPM. Besides, the White Rabbit system can provide precise global time stamps. With the help of these state-of-the-art technologies, the arrival time of Cherenkov photons will be more precisely measured. Applying hDOM in the telescope will achieve ~40% improvement in angular resolution in comparison to traditional PMT only DOM, which would significantly boost the source searching capability.</p>
</center>

### Collected Dataset

As shown above, the physical structure of TRIDENT determines the data structure we collected from TRIDENT. By converting signals accepted by hDOMs into orthogonal coordinates, we can feed data into common deep-learning models to infer the incident angle of neutrinos. The converted data sample is shown as this:

<center><div style='width:600px'>
    <img src="https://github.com/Airscker/DeepMuon/blob/site/blogs/trident/example_1TeV.png?raw=true" alt="example_1TeV" width='600px'/>
    <p align='left'>
        <b>Converted data structure.</b> The strength of signal points is represented as the color of signal points, and the vector indicates the direction of the incident neutrino.
    </p>
    </div></center>

However, the real data we got contained three types of information at every signal point: 

- Hit number
- Mean arriving time
- Standard deviation of arrival time

What's more, not every non-zero (at least contained one kind of information) point has three types of information, and each data sample's available signal points are different from another:

<center><div style='width:1000px' id='dist'><img src='https://github.com/Airscker/DeepMuon/blob/site/blogs/trident/dist.jpg?raw=true'><p align='left'>
    <b>Distribution of the number of available Signal points of every data sample(NAS).</b>
    The maximal number of available signal points within a data sample if 356 and the minimal number is only 1. And the mean value of NAS is 214, the standard derivation is 42. Suppose the distribution obeys normal distribution then its 3-Sigma range is 340 ~ 87.
    </p></div></center>

To avoid the possible negative effects brought by out-of-physical data samples, we need to abandon points without three kinds of information and abandon samples whose NAS is lower than $\mu-3\sigma=87$.

## Experiment

### ResMax3 (Residual Unit Based Multi-modality Spatial Pyramid Max-pooling V3)

<center><div style='width:1000px'><img src='https://github.com/Airscker/DeepMuon/blob/site/blogs/trident/resmax3.png?raw=true'><p align='left'>
    <b>Architecture of ResMax3</b>
    </p></div></center>

The 

### Important Configurations

#### Loss Function
$$
MSALoss(X,Y)=\frac{1}{n}\sum_{i=1}^{n=Dim_{X,Y}}[\alpha Mean(X_i-Y_i)^2+\beta(Sin(Angle(X_i,Y_i)))]
$$

Here we take angle loss and vector direction loss together into consideration. The vector direction loss was defined as the distance between the predicted unified direction vector(UDV) and the real UDV, the angle loss was defined as the sine value of the angle between UDVs. To find out the effects of different losses on the final result, we apply coefficients $\alpha$ and $\beta$ to weigh the importance of different losses.

The value of weights $\alpha$ and $\beta$ are chosen by evaluating different experiments' performance with all the same configuration except for the loss weights. After comparing different experiments' convergence speed and ultimate result at epoch 1000, we concluded that the best choice of loss weights is that:

$$
\alpha=0, \beta=1
$$

This result can be interpreted that because the angle loss only requires lines represented by predicted UDV and actual UDV has the same slope, and vector direction loss additionally requires the direction of vectors to remain close, it's easier for the model to converge to the global minimum value by using pure angle loss.
#### Tensor Precision

In modern computer science, the data stored within a computer (Only *Von Neumann*-type computers are considered) have different lengths of bits. In this experiment, the available precision types of the dataset are 16-bit float, 32-bit float, and 64-float. If nothing is specified the default precision of the dataset is 32-bit float. Without any exception, the 16-bit precision data are used to accelerate the training procedure of our models, that is Mixed Precision Training. However, Just as what we have shown in the section [Collected Dataset](#dist), the maximum number of NAS is 356, when it comes to the number of all points within a data sample, this value is 4000, that is to say, the maximum proportion of NAS of data samples is no more than $100\%\times 356/4000=8.9\%$. What extreme sparseness of NAS brought is that if we use default 32-bit float type Tensors to train the model, we will meet ***nan*** (Not a number, for short) values with model parameters. To solve this issue, we have to use the 64-bit type data to train our models, which means that the model is much more likely to occur gradient annihilation or gradient explosion when the model is handling 32-bit type float Tensors. **This phenomenon reflects that the important information was contained in the additional 32 bits compared to the 32-bit type Tensor. This is an important and rare phenomenon we should pay attention to, especially when our data samples' sparseness is really strong.**

#### Multi-modality

As we have seen, every point of the data sample has three different types of information, the hit number reflects the number of neutrino signals received by the corresponding detector, the mean arriving time represents the mean time of Cherenkov photons arriving the detector, and the standard derivation of arrival time has the similar statistical meaning with mean arriving time. 

The information of every data point can be divided into two groups, one group is hit number and another consists of mean/std arriving time. First, because hit num and arriving time have different physical meanings, and second, the mean value of hit num is far smaller than the mean value of mean/std arriving time. Theoretically, combining different types of information to train the same branch of the model will result in bad results, because different information modalities are entangled together to train the model, however, in the physical world, this operation is impossible to happen. What's more, there are several experiments that proved that entangled modalities make models worse. 

#### Data Preprocessing
##### Normalization
Every data sample has unique maximum/minimum values of available information, but apparently, we don't want the model to learn the differences between different samples' max/min signals. So we normalized the hit number and mean arriving time values of every sample within the range [0,1], and the standard arriving time is divided by the proportion of mean arriving time changed.
##### Augmentation
The augmentation methods are expected to avoid the over-fitting of the model, in this experiment, available augmentation technics are:
- Rotating 90/180 degrees
- Flipping

### Training Pipeline

We set the batch size as 48 at the first 1000 epochs, the initial learning rate is 0.0002, we used AdamW optimizer, weight_decay is 0.01, learning rate scheduler is ReduceLROnPlateau. From epoch 1001, the batch size was 1024, we changed the optimizer at epoch 1418 to SGD with momentum 0.9, and at epoch 1829, the optimizer returned to AdamW. From epoch 2057 to epoch 2200, we freeze the residual convolution part of ResMax3. To avoid the gradient annihilation or gradient explosion we used the gradient clip method, setting the maximum gradient clipping value as 0.01. 

### Interpretation - Integrated Gradient




<p id='ref1'>[1]: https://trident.sjtu.edu.cn/en</p>

<p id='ref2'>[2]: K.Murase, F. W. Stecker, <b>High-Energy Neutrinos from Active Galactic Nuclei.</b> arXv:2202.03381.</p>

<p id='ref3'>[3]: K. Kotera, D. Allard, A. Olinto, <b>Cosmogenic neutrinos: parameter space and detectabilty from PeV to ZeV</b>, Journal of Cosmology and Astroparticle Physics 2010 (10) (2010) 013-013. doi:10.1088/1475-7516/2010/10/013.</p>

<p id='ref4'>[4]: Z. P. Ye, F. Hu, W. Tian, Q. C. Chang, Y. L. Chang, Z. S. Cheng, J. Gao, T. Ge, G. H. Gong, J. Guo, X. X. Guo, X. G. He, J. T. Huang, K. Jiang, P. K. Jiang, Y. P. Jing, H. L. Li, J. L. Li, L. Li, W. L. Li, Z. Li, N. Y. Liao, Q. Lin, F. Liu, J. L. Liu, X. H. Liu, P. Miao, C. Mo, I. Morton-Blake, T. Peng, Z. Y. Sun, J. N. Tang, Z. B. Tang, C. H. Tao, X. L. Tian, M. X. Wang, Y. Wang, Y. Wang, H. D. Wei, Z. Y. Wei, W. H. Wu, S. S. Xian, D. Xiang, D. L. Xu, Q. Xue, J. H. Yang, J. M. Yang, W. B. Yu, C. Zeng, F. Y. D. Zhang, T. Zhang, X. T. Zhang, Y. Y. Zhang, W. Zhi, Y. S. Zhong, M. Zhou, X. H. Zhu, G. J. Zhuang, <b>Proposal for a neutrino telescope in South China Sea.</b>arXiv:2207.04519</p>