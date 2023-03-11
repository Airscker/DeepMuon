# From zero to infinity

> The term machine learning was coined in 1959 by Arthur Samuel, an IBM employee and pioneer in the field of computer gaming and artificial intelligence. The synonym self-teaching computers were also used in this time period.
>
> By the early 1960s an experimental "learning machine" with punched tape memory, called CyberTron, had been developed by Raytheon Company to analyze sonar signals, electrocardiograms, and speech patterns using rudimentary reinforcement learning. It was repetitively "trained" by a human operator/teacher to recognize patterns and equipped with a "goof" button to cause it to re-evaluate incorrect decisions. A representative book on research into machine learning during the 1960s was Nilsson's book on Learning Machines, dealing mostly with machine learning for pattern classification. Interest related to pattern recognition continued into the 1970s, as described by Duda and Hart in 1973. In 1981 a report was given on using teaching strategies so that a neural network learns to recognize 40 characters (26 letters, 10 digits, and 4 special symbols) from a computer terminal.
>
> Tom M. Mitchell provided a widely quoted, more formal definition of the algorithms studied in the machine learning field: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E." This definition of the tasks in which machine learning is concerned offers a fundamentally operational definition rather than defining the field in cognitive terms. This follows Alan Turing's proposal in his paper "Computing Machinery and Intelligence", in which the question "Can machines think?" is replaced with the question "Can machines do what we (as thinking entities) can do?".
>
> Modern-day machine learning has two objectives, one is to classify data based on models which have been developed, and the other purpose is to make predictions for future outcomes based on these models. A hypothetical algorithm specific to classifying data may use computer vision of moles coupled with supervised learning in order to train it to classify the cancerous moles. A machine learning algorithm for stock trading may inform the trader of future potential predictions.[1](#ref1)

Nowadays, deep-learning models are created to solve many problems not only traditional computer vision tasks but also interdisciplinary problems. The procedure of solving problems is a trip from zero to infinity. That is, the creation of appropriate models is the key to our success and it's a hard job. So we need to find out the right way to represent the principle of the physical world or reveal the latent variables not found yet.

## Which model fits best?

- Choose a well-established, commonly used model architecture to get working first. It is always possible to build a custom model later. [2](#ref2)
- Model architectures typically have various hyperparameters that determine the model's size and other details (e.g. number of layers, layer width, type of activation function). [2](#ref2)
  
  Thus, choosing the architecture really means choosing a family of different models (one for each setting of the model hyperparameters). [2](#ref2)
- When possible, try to find a paper that tackles something as close as possible to the problem at hand and reproduce that model as a starting point. [2](#ref2)

- CNN models tend to learn the localized information and Transformers typically have larger attention areas.
- The larger model, the larger dataset, otherwise more augmentations.
- Keep trying, and keep making progress.

## Grammar

The grammar of models in DeepMuon is the same as PyTorch, just like the `Dataset`, there are still some tips you need to pay attention to:

- Position of the model file:

  You can put it anywhere and you just need to specify the path of it when you are preparing the configuration of training. Or you can put it under the installation position of DeepMuon, folder `models`, and add the name of it into the file `__init__.py`.

- Example:

  ```python
  class MLP3(nn.Module):
      def __init__(self):
          super().__init__()
          self.flatten = nn.Flatten()
          self.linear_relu_stack = nn.Sequential(
              nn.Linear(17*17, 512),
              nn.BatchNorm1d(512),
              nn.LeakyReLU(),
              nn.Linear(512, 128),
              nn.BatchNorm1d(128),
              nn.LeakyReLU(),
              nn.Linear(128, 2)
          )
  
      def forward(self, x):
          x = self.flatten(x)
          logits = self.linear_relu_stack(x)
          return logits
  ```

  This is the code of MLP, which was used in the project [PandaX-4T III Radioactive Source Localization](https://airscker.github.io/DeepMuon/blogs/index.html#/pandax/pandax?id=pandax-4t-iii-radioactive-source-localization), the model class must be inherited from `torch.nn.Module`, and method `forward()` must be created.

  Function `forward()` typically accepts the data as its parameter, and you can define the model architecture within `__init()__`, PyTorch will automatically add `backward()` method to your model. The function `forward()` must return the result given by your model, typically it should be `torch.Tensor` type data.

  What's more, to help understand how to freeze some stages of your model, we show an example from the project [Artificial Intelligence Enabled Cardiac Magnetic Resonance Image Interpretation](https://airscker.github.io/DeepMuon/blogs/index.html#/cmr_vst/cmr_vst?id=artificial-intelligence-enabled-cardiac-magnetic-resonance-image-interpretation):

  ```python
  class fusion_model(nn.Module):
      def __init__(self,
                   num_classes,
                   mlp_in_channels=1024,
                   mlp_dropout_ratio=0.5,
                   freeze_vst=True,
                   patch_size=(2, 4, 4),
                   embed_dim=128,
                   depths=[2, 2, 18, 2],
                   num_heads=[4, 8, 16, 32],
                   window_size=(8, 7, 7),
                   mlp_ratio=4.,
                   qkv_bias=True,
                   qk_scale=None,
                   drop_rate=0.,
                   attn_drop_rate=0.,
                   drop_path_rate=0.3,
                   patch_norm=True,
                   sax_weight=None,
                   lax_weight=None,
                   lge_weight=None
                   ):
          super().__init__()
          weights = []
          if sax_weight is not None:
              weights.append(sax_weight)
          if lax_weight is not None:
              weights.append(lax_weight)
          if lge_weight is not None:
              weights.append(lge_weight)
          mlp_num_mod = len(weights)
          self.vst_backbones = nn.ModuleList([SwinTransformer3D(patch_size=patch_size,
                                                                embed_dim=embed_dim,
                                                                depths=depths,
                                                                num_heads=num_heads,
                                                                window_size=window_size,
                                                                mlp_ratio=mlp_ratio,
                                                                qkv_bias=qkv_bias,
                                                                qk_scale=qk_scale,
                                                                drop_rate=drop_rate,
                                                                attn_drop_rate=attn_drop_rate,
                                                                drop_path_rate=drop_path_rate,
                                                                patch_norm=patch_norm) for _ in range(mlp_num_mod)])
          self.init_weights(weights=weights)
          self.freeze = freeze_vst
          self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
          self.dropout = nn.Dropout(mlp_dropout_ratio)
          self.linear = nn.Linear(mlp_in_channels*mlp_num_mod, num_classes)
          self.freeze_vst()
  
      def init_weights(self, weights):
          for i in range(len(weights)):
              try:
                  checkpoint = torch.load(weights[i], map_location='cpu')
                  self.vst_backbones[i].load_state_dict(checkpoint, strict=False)
                  print(f'{weights[i]} loaded successfully')
              except:
                  print(f'{weights[i]} loading fail')
  
      def forward(self, x: torch.Tensor):
          '''x: NMCTHW'''
          assert x.shape[1] == len(
              self.vst_backbones), f'Multi modality input data types does not match the number of vst backbones; {len(self.vst_backbones)} types of data expected however {x.shape[1]} given'
          x = torch.permute(x, (1, 0, 2, 3, 4, 5))
          features = []
          for i in range(x.shape[0]):
              features.append(self.avgpool(
                  self.vst_backbones[i](x[i])).unsqueeze(0))
          features = torch.cat(features, dim=0)
          x = torch.permute(features, (1, 0, 2, 3, 4, 5))
          x = self.dropout(x)
          x = x.view(x.shape[0], -1)
          x = self.linear(x)
          return x
  
      def freeze_vst(self):
          if self.freeze:
              self.vst_backbones.eval()
              for params in self.vst_backbones.parameters():
                  params.requires_grad = False
  
      def train(self, mode: bool = True):
          super().train(mode)
          self.freeze_vst()
  ```
  
The function `train()` is inherited from `torch.nn.Module`, and you can define the **freeze_stage** method to freeze some layers or neurons during the model training. In the function `freeze_vst()`, we let the Video Swin-Transformer (VST for short) part of the fusion model freeze and only the linear transform part is allowed to be trained. Just like the pipeline of freezing shown above, `module.eval()` first, then `model.parameters.requires_grad=False`. Freezing model parts can reduce the computing resources needed especially we are using pre-trained models.

## Bibliography

<p id='ref1'>[1] https://en.wikipedia.org/wiki/Machine_learning</p>
<p id='ref2'>[2] https://github.com/google-research/tuning_playbook</p>