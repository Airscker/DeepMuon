# How to create your own loss function?

Modern deep-learning models' convergency is mainly based on the gradient descent algorithm. That is to say, we optimize our model's parameters according to the gradient and loss value directly or indirectly given by loss functions. So choosing a proper loss function is vital to our targets. Loss functions can restrict the condition of models and tell them what to learn. What's more, loss functions also reflect the performance of our models. Up to now, there are many kinds of loss functions available, such as MSELoss, CrossEntropyLoss, and KLDivLoss. However these loss functions are not always the right choices for our task, so we have to customize our own losses according to the target we want to reach.

All customized loss functions should be able to reflect the gradient of the model's parameters, that is to say, we'd better make use of the auto-grad mechanism provided by PyTorch so we don't have to write `backward()` function for every loss function. Here is an example of `MSALoss`:

```python
from torch import nn
import torch

class MSALoss(nn.Module):
    """
    ## MSEloss(vec1,vec2)+Angle(vec1,vec2)

    ### Args:
        - angle_ratio: The ratio to consider the angle loss into total loss. Defaults to 1.
        - len_ratio: The ratio to consider the distance loss into total loss. Defaults to 1.
    """

    def __init__(self, angle_ratio=1, len_ratio=0):
        super().__init__()
        self.angle_ratio = angle_ratio
        self.len_ratio = len_ratio

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mseloss=torch.mean((input-target)**2)
        angloss = torch.mean(torch.sin(torch.arccos(torch.sum(
            input*target, axis=1)/torch.sqrt(torch.sum(input**2, axis=1)*torch.sum(target**2, axis=1)))))
        return self.len_ratio*mseloss+self.angle_ratio*angloss
```

This loss function calculates the relative angle and relative distance between two vectors. In this way, we help the model to localize the proper vector by looking for the proper direction and vector length. The distance loss is given by MSELoss and the angle loss is given by the sine value of the relative angle. Choosing a sine value avoids the $2\pi$ jump and gradient annihilation near zero value (if we choose cosine, the gradient near zero is 0).

Just like `Dataset` and models which we have introduced before, the path of the customized loss function can be anywhere and we recommend you to put it under the installation path of DeepMuon, folder `loss_fn`, adding the name into `__init__.py` too. Otherwise, you need to specify the loss function file's path in the configuration.