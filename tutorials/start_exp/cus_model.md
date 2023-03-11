# Customize model

Just as we have mentioned in the last section, the customized models are all based on PyTorch, too. PyTorch is a widely used deep-learning framework, it has complete and excellent technical support, which enables it to be convenient and easy to build experiments. We believe that learning PyTorch will help you get familiar with the modern deep-learning frontiers easier. But as for non-CS major researchers, it's still hard for them to get the hang of PyTorch, so the target of DeepMuon is to minimize the learning cost. You just need to learn the basic grammar of PyTorch then you can fully control your interdisciplinary deep-learning research.

Now we are met with the second important step of a standard deep-learning experiment: **Creating a suitable neural network model.** With the development of machine learning, we have seen many classical and advanced models built for different tasks. Here we show some famous models to help you have an approximate view of the deep-learning.

- SVM: Support vector machines
    > In machine learning, support vector machines (SVMs, also support vector networks [1](#ref1)) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis. Developed at AT&T Bell Laboratories by Vladimir Vapnik with colleagues (Boser et al., 1992, Guyon et al., 1993, Cortes and Vapnik, 1995 [1](#ref1), Vapnik et al., 1997) SVMs are one of the most robust prediction methods, being based on statistical learning frameworks or VC theory proposed by Vapnik (1982, 1995) and Chervonenkis (1974). Given a set of training examples, each marked as belonging to one of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). SVM maps training examples to points in space so as to maximize the width of the gap between the two categories. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.
    >
    > In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.

- CNN: Convolutional neural networks

    > Convolutional neural networks are specialized types of artificial neural networks that use a mathematical operation called convolution in place of general matrix multiplication in at least one of their layers. They are specifically designed to process pixel data and are used in image recognition and processing.

- Transformer

    > A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing (NLP) [2](#ref2) and computer vision (CV). [3](#ref3)
    >
    > Like recurrent neural networks (RNNs), transformers are designed to process sequential input data, such as natural language, with applications for tasks such as translation and text summarization. However, unlike RNNs, transformers process the entire input all at once. The attention mechanism provides context for any position in the input sequence. For example, if the input data is a natural language sentence, the transformer does not have to process one word at a time. This allows for more parallelization than RNNs and therefore reduces training times. [2](#ref4)
    >
    > Transformers were introduced in 2017 by a team at Google Brain[2] and are increasingly the model of choice for NLP problems, [4](#ref4) replacing RNN models such as long short-term memory (LSTM). The additional training parallelization allows training on larger datasets. This led to the development of pre-trained systems such as BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer), which were trained with large language datasets, such as the Wikipedia Corpus and Common Crawl, and can be fine-tuned for specific tasks. [5](#ref5),[6](#ref6)

Here to avoid the complexness brought by model itself, we choose MLP as an example to introduce the model customizing method:

```python
from torch import nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

We recommend to save this model under the installation path of DeepMuon, within the folder `models`.

This is a simplest neural network, it has one hidden layer and three layers. The model is applied on the MINIST dataset to classify the flowers in the dataset.

The customization is over and the next step is decide your experiments' training/testing configuration, and this is the last step to prepare our experiments. Please read the section [Customize configuration](https://airscker.github.io/DeepMuon/tutorials/index.html#/start_exp/cus_config) to complete your first experiment.


## Bibliography
<p id='ref1'>[1] Cortes, Corinna; Vapnik, Vladimir (1995). "Support-vector networks". Machine Learning. 20 (3): 273–297. CiteSeerX 10.1.1.15.9362. doi:10.1007/BF00994018. S2CID 206787478.</p>
<p id='ref2'>[2] Vaswani, Ashish; Shazeer, Noam; Parmar, Niki; Uszkoreit, Jakob; Jones, Llion; Gomez, Aidan N.; Kaiser, Lukasz; Polosukhin, Illia (2017-06-12). "Attention Is All You Need". arXiv:1706.03762 [cs.CL].</p>
<p id='ref3'>[3] He, Cheng (31 December 2021). "Transformer in CV". Transformer in CV. Towards Data Science.</p>
<p id='ref4'>[4] Wolf, Thomas; Debut, Lysandre; Sanh, Victor; Chaumond, Julien; Delangue, Clement; Moi, Anthony; Cistac, Pierric; Rault, Tim; Louf, Remi; Funtowicz, Morgan; Davison, Joe; Shleifer, Sam; von Platen, Patrick; Ma, Clara; Jernite, Yacine; Plu, Julien; Xu, Canwen; Le Scao, Teven; Gugger, Sylvain; Drame, Mariama; Lhoest, Quentin; Rush, Alexander (2020). "Transformers: State-of-the-Art Natural Language Processing". Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. pp. 38–45. doi:10.18653/v1/2020.emnlp-demos.6. S2CID 208117506.</p>
<p id='ref5'>[5] "Open Sourcing BERT: State-of-the-Art Pre-training for Natural Language Processing". Google AI Blog. Retrieved 2019-08-25.</p>
<p id='ref6'>[6] "Better Language Models and Their Implications". OpenAI. 2019-02-14. Retrieved 2019-08-25.</p>
