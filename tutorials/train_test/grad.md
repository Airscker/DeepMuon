# Gradient clipping & Gradient accumulation

## Gradient Clipping/Scaling

Gradient clipping is a technique that prevents exploding gradients in deep neural networks. It's used to improve stability and convergence when training very deep learning models. We use gradient clipping because during back-propagation in deep neural networks, unstable gradients can be a problem for neural networks, especially in recurrent neural networks.

Gradient clipping works by rescaling gradients when they exceed a threshold. This ensures stable training. And gradient scaling works by assigning a numerical value to each feature in a dataset. The value is based on a numerical scale, or gradient, that is determined by the relative importance of the feature. That is, normalizing the gradient vectors instead of clipping them.

For more details we recommend you to read these papers:

- [Floaters No More: Radiance Field Gradient Scaling for Improved Near-Camera Training
  ](https://arxiv.org/abs/2305.02756)
- [Morphogen gradient scaling by recycling of intracellular Dpp](https://pubmed.ncbi.nlm.nih.gov/34937053/)
- [Reparameterization through Spatial Gradient Scaling
  ](https://arxiv.org/abs/2303.02733)
- [Contextual Gradient Scaling for Few-Shot Learning](https://arxiv.org/abs/2303.02733)

## Gradient Accumulation

Gradient accumulation is a technique that combines the gradients from multiple optimization steps into one. This allows the gradients to be applied at regular intervals.

Gradient accumulation can be used to overcome memory limitations when training large models or processing large batches of data. For example, it can simulate training with a larger batch size than would fit into the available device memory.

To perform gradient accumulation, you can:

- Split a mini-batch into several micro-batches.
- Compute the gradients of the micro-batches sequentially.
- Accumulate the gradients to reduce the memory footprint of activations.

Gradient accumulation is useful when working with large images/volumetric data, using low-end hardware, or training on multiple GPUs.
For example, if your RAM limits you to batch size 16, you can try batch size = 4 and gradient accumulation = 8, which will result in batch size 32 and faster speeds.

## How to use them in `DeepMuon`?

You just need to specify the value of gradient clipping/scaling/accumulation in `config`, more details about using method please refer to [Elements of `config.py`](https://airscker.github.io/DeepMuon/tutorials/index.html#/config/config?id=optimize_config)
