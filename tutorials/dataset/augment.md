# How to add augmentations?

Augmentation methods are applied to increase the dataset's diversity, which is really helpful for reducing the over-fitting of our models, but how can we add it within the DeepMuon?

Here we show a method that comes from the project [TRIDENT Neutrino Telescope (Hailing Plan) Tracing Task](https://airscker.github.io/DeepMuon/blogs/index.html#/trident/trident?id=trident-neutrino-telescope-hailing-plan-tracing-task):

```python
def Rotate90(image, label):
    """
    ## Rotate the 3D image along Z axis by 90 degrees and return the rotated image and label.

    ### Args:
        - image: the image to rotate.
        - label: the label to rotate.

    ### Return:
        - the rotated image and label.
    """
    image = np.transpose(np.array(image), (1, 0, 2, 3))
    image = image[::-1, ...]
    label = np.array([-label[1], label[0], label[2]])
    return image, label


def Rotate180(image, label):
    """
    ## Rotate the 3D image along Z axis by 180 degrees and return the rotated image and label.

    ### Args:
        - image: the image to rotate.
        - label: the label to rotate.

    ### Return:
        - the rotated image and label.
    """
    image = np.array(image)
    image = image[::-1, ::-1, :, :]
    label = np.array([-label[0], -label[1], label[2]])
    return image, label


def Flip(image, label):
    """
    ## Flip the 3D image' Z axis and return the rotated image and label.

    ### Args:
        - image: the image to rotate.
        - label: the label to rotate.

    ### Return:
        - the flipped image and label.
    """
    image = np.array(image)
    image = image[:, :, ::-1, :]
    label = np.array([label[0], label[1], -label[2]])
    return image, label
```

The functions shown above are methods we have used during the training of ResMax3, and if you are careful enough, you can find that in the last section's example we specified the random augmentation pipeline in this way:

```python
'''Data augmentation'''
if self.augment:
  # [0,3]range,[0,3]random length
  oper = np.unique(np.random.randint(0, 4, np.random.randint(0, 4)))
  for oper_i in range(len(oper)):
    image, label = self.augmentation[oper[oper_i]](image, label)
```

And in the `__init__()`, we specified available augmentation methods like this:

```python
self.augmentation = {0: Rotate180, 1: Rotate90, 2: Flip, 3: Same}
```

So in this way, we can randomly apply the augmentation pipelines by generating a sequence of random integers. These integers correspond to different augmentation methods.

Actually, there are many ways to add augmentation pipelines, the example given above aims to inspire your creation.