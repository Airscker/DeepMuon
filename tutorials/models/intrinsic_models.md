# Intrinsically supported models

To help you have a more simple experience when planning your experiments, we provided some models intrinsically in the DeepMuon. However, they are impossible to cover all types of tasks, we offer these models to help you understand the mechanism of different models and you can learn how to create your own models.

Here we list out the models intrinsically supported in the DeepMuon:

- ResMax3 (Residual Unit Based Multi-modality Spatial Pyramid Max-pooling V3)

  > ResMax3 was designed to process the high-sparseness TRIDENT dataset, due to the sparseness of the TRIDENT dataset, traditional convolution layers will dilute the features when models go deeper. To solve this problem, we implemented the structure of spatial pyramid max-pooling (SPP). The key step of SPP is that we use the MaxPooling method to select the strongest feature, at the same time we can ignore the diluted information which comes with convolution layers. In this way, the robustness of ResMax3 is improved.
  >
  > Considering the multi-modality property of TRIDENT dataset, we created two branches of residual units, these branches process different modalities' data and extract features from them. To get more details on ResMax3, please refer to [TRIDENT Neutrino Telescope (Hailing Plan) Tracing Task](https://airscker.github.io/DeepMuon/blogs/index.html#/trident/trident?id=trident-neutrino-telescope-hailing-plan-tracing-task).

- MLP (Multilayer Perceptron)

  > MLPs are useful in research for their ability to solve problems stochastically, which often allows approximate solutions for extremely complex problems like fitness approximation. 
  >
  > MLPs are universal function approximators as shown by Cybenko's theorem, so they can be used to create mathematical models by regression analysis. As classification is a particular case of regression when the response variable is categorical, MLPs make good classifier algorithms. 
  >
  > MLPs were a popular machine learning solution in the 1980s, finding applications in diverse fields such as speech recognition, image recognition, and machine translation software, but thereafter faced strong competition from much simpler (and related) support vector machines. Interest in backpropagation networks returned due to the successes of deep learning. [1](#ref1)

- U-Net

  > U-Net is a convolutional neural network that was developed for biomedical image segmentation at the Computer Science Department of the University of Freiburg [2](#ref2). The network is based on the fully convolutional network and its architecture was modified and extended to work with fewer training images and to yield more precise segmentations.
  >
  > The network consists of a contracting path and an expansive path, which gives it the u-shaped architecture. The contracting path is a typical convolutional network that consists of repeated application of convolutions, each followed by a rectified linear unit (ReLU) and a max pooling operation. During the contraction, the spatial information is reduced while feature information is increased. The expansive pathway combines the feature and spatial information through a sequence of up-convolutions and concatenations with high-resolution features from the contracting path.
  >
  > There are many applications of U-Net in biomedical image segmentation, such as brain image segmentation and liver image segmentation as well as protein binding site prediction. Variations of the U-Net have also been applied for medical image reconstruction. Here are some variants and applications of U-Net as follows: 
  >
  > - Pixel-wise regression using U-Net and its application on pansharpening. [3](#ref3)
  > - 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. [4](#ref4)
  > - TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation. [5](#ref5)
  > - Image-to-image translation to estimate fluorescent stains. [6](#ref6)
  > - In binding site prediction of protein structure. [7](#ref7)

- ViT (Vision Transformer)

  > ViT performance depends on decisions including that of the optimizer, dataset-specific hyperparameters, and network depth. CNNs are much easier to optimize. A variation on a pure transformer is to marry a transformer to a CNN stem/front end. 
  >
  > A typical ViT stem uses a 16x16 convolution with a 16 stride. By contrast, a 3x3 convolution with stride 2, increases stability and also improves accuracy. [9](#ref9) 
  >
  > The CNN translates from the basic pixel level to a feature map. A tokenizer translates the feature map into a series of tokens that are then fed into the transformer, which applies the attention mechanism to produce a series of output tokens. Finally, a projector reconnects the output tokens to the feature map. The latter allows the analysis to exploit potentially significant pixel-level details. This drastically reduces the number of tokens that need to be analyzed, reducing costs accordingly. [10](#ref10) 
  >
  > The differences between CNNs and Vision Transformers are many and lie mainly in their architectural differences. In fact, CNNs achieve excellent results even with training based on data volumes that are not as large as those required by Vision Transformers. This different behavior seems to derive from the presence in the CNNs of some inductive biases that can be somehow exploited by these networks to grasp more quickly the particularities of the analyzed images even if, on the other hand, they end up limiting them making it more complex to grasp global relations.[11](#ref11),[12](#ref12) On the other hand, the Vision Transformers are free from these biases which leads them to be able to capture also global and wider range relations but at the cost of more onerous training in terms of data. Vision Transformers also proved to be much more robust to input image distortions such as adversarial patches or permutations. [13](#ref13) However, choosing one architecture over another is not always the wisest choice, and excellent results have been obtained in several Computer Vision tasks through hybrid architectures combining convolutional layers with Vision Transformers. [14](#ref14),[15](#ref15),[16](#ref16)

- VST (Video Swin-Transformer)

  > Previously for convolutional models, backbone architectures for video were adapted from those for images simply by extending the modeling through the temporal axis. For example, 3D convolution is a direct extension of 2D convolution for joint spatial and temporal modeling at the operator level. As joint spatiotemporal modeling is not economical or easy to optimize, factorization of the spatial and temporal domains was proposed to achieve a better speed-accuracy tradeoff. In the initial attempts at Transformer-based video recognition, a factorization approach is also employed, via a factorized encoder or factorized self-attention. This has been shown to greatly reduce model size without a substantial drop in performance. [17](#ref17)
  >
  > VST takes advantage of the inherent spatiotemporal locality of videos, in which pixels that are closer to each other in the spatiotemporal distance are more likely to be correlated. Because of this property, full spatiotemporal self-attention can be well-approximated by self-attention computed locally, at a significant saving in computation and model size. [17](#ref17)

More models will be added with the development of DeepMuon, however, the target of providing intrinsically supported models is not to provide solutions for you, but to inspire your creation and make much more progress standing on the shoulder of giants.

## Bibliography

<p id='ref1'>[1] https://en.wikipedia.org/wiki/Multilayer_perceptron</p>
<p id='ref2'>[2] Ronneberger O, Fischer P, Brox T (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation". arXiv:1505.04597.</p>
<p id='ref3'>[3]  Yao W, Zeng Z, Lian C, Tang H (2018-10-27). "Pixel-wise regression using U-Net and its application on pansharpening". Neurocomputing. 312: 364–371. doi:10.1016/j.neucom.2018.05.103. ISSN 0925-2312. S2CID 207119255.</p>
<p id='ref4'>[4] Çiçek Ö, Abdulkadir A, Lienkamp SS, Brox T, Ronneberger O (2016). "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation". arXiv:1606.06650.</p>
<p id='ref5'>[5] Iglovikov V, Shvets A (2018). "TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation". arXiv:1801.05746.</p>
<p id='ref6'>[6] Kandel ME, He YR, Lee YJ, Chen TH, Sullivan KM, Aydin O, et al. (December 2020). "Phase imaging with computational specificity (PICS) for measuring dry mass changes in sub-cellular compartments". Nature Communications. 11 (1): 6256. arXiv:2002.08361. doi:10.1038/s41467-020-20062-x. PMC 7721808. PMID 33288761.</p>
<p id='ref7'>[7] Nazem F, Ghasemi F, Fassihi A, Dehnavi AM (April 2021). "3D U-Net: A voxel-based method in binding site prediction of protein structure". Journal of Bioinformatics and Computational Biology. 19 (2): 2150006. doi:10.1142/S0219720021500062. PMID 33866960.</p>
<p id='ref9'>[9] Xiao, Tete; Singh, Mannat; Mintun, Eric; Darrell, Trevor; Dollár, Piotr; Girshick, Ross (2021-06-28). "Early Convolutions Help Transformers See Better". arXiv:2106.14881</p>
<p id='ref10'>[10] Synced (2020-06-12). "Facebook and UC Berkeley Boost CV Performance and Lower Compute Cost With Visual Transformers". Medium. Retrieved 2021-07-11.</p>
<p id='ref11'>[11] Raghu, Maithra; Unterthiner, Thomas; Kornblith, Simon; Zhang, Chiyuan; Dosovitskiy, Alexey (2021-08-19). "Do Vision Transformers See Like Convolutional Neural Networks?". arXiv:2108.08810</p>
<p id='ref12'>[12] Coccomini, Davide (2021-07-24). "Vision Transformers or Convolutional Neural Networks? Both!"</p>
<p id='ref13'>[13] Naseer, Muzammal; Ranasinghe, Kanchana; Khan, Salman; Hayat, Munawar; Khan, Fahad Shahbaz; Yang, Ming-Hsuan (2021-05-21). "Intriguing Properties of Vision Transformers". arXiv:2105.10497</p>
<p id='ref14'>[14] Dai, Zihang; Liu, Hanxiao; Le, Quoc V.; Tan, Mingxing (2021-06-09). "CoAtNet: Marrying Convolution and Attention for All Data Sizes". arXiv:2106.04803</p>
<p id='ref15'>[15] Wu, Haiping; Xiao, Bin; Codella, Noel; Liu, Mengchen; Dai, Xiyang; Yuan, Lu; Zhang, Lei (2021-03-29). "CvT: Introducing Convolutions to Vision Transformers". arXiv:2103.15808</p>
<p id='ref16'>[16] Coccomini, Davide; Messina, Nicola; Gennaro, Claudio; Falchi, Fabrizio (2022). "Combining Efficient Net and Vision Transformers for Video Deepfake Detection". Image Analysis and Processing – ICIAP 2022. Lecture Notes in Computer Science. Vol. 13233. pp. 219–229. arXiv:2107.02612. doi:10.1007/978-3-031-06433-3_19. ISBN 978-3-031-06432-6. S2CID 235742764</p>
<p id='ref17'>[17] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, Han Hu. "Video Swin Transformer" arXiv:2106.13230.</p>