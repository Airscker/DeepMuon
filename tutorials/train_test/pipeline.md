# Prediction Pipeline

Typically there are serveral types of deep learning tasks, each of them may have different kinds of output and input, these inputs and outputs may also have different precisions, how to specify them is never a trouble if we don't care about the tidiness and quality o four code. Actually, in very early editions of `DeepMuon`, if we want to train a new model, sometimes we need to modify the `dist_train.train/test` to give the right input to models. For example, the `MLP` model just need a one-dim tensor but `CNNLSTM` needs two initial inputs. Training these two models sequencially just requires us to modify `dist_train.train/test` once, but what if we need to train them at the same time? what if there are much more diffrent models requires different kinds/numbers of inputs/outputs?

To solve this problem, keeping our code tidy and high-quality, we added `Pipeline` mechanism to help you choose how to properly post/pre-process model inputs/outputs, how to change the precision of tensors, how to ... etc. To understand how to use `Pipeline`, let's see an example of regression tasks:

```python
class classify(Pipeline):
    '''
    Model prediction pipeline built for normal classfication tasks such as Swin-Transformer, ResNet, Video-Swin Transformer, Vision Transformer etc.

    The predcition pipeline will give out:
        - pred: `torch.FloatTensor`/`torch.DoubleTensor`, the predicted values of classification models.
        - label: `torch.LongTensor`, the label used to evaluate the model's results.
    '''
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
    def predict(self,input,label,device,precision):
        input=input.type(precision).to(device)
        label=label.reshape(-1).to(device)
        pred=self.model(input)
        return pred,label
```

First, all customized training/testing pipeline must be implemeted within `DeepMuon.train.pipeline.py` (Don't have to include them into `__init__.py`)

Then, to implement your own pipeline, your class must be inherited from base class `DeepMuon.train.Pipeline` (Shown as the first line).

After steps above, you must implement method `predict`, which defaultly have four parameters:

- input: the input data get from dataloader.
- label: the lable get from dataloader.
- device: the device used to train/test model.
- precision: the data/label precision used to train/test model.

You don't have to make use of every parameters within `predict`, just make sure the returned values are predicted values of model and labels pre-processed/untouched, such as `pred` and `label`.

As you see, by using `Pipeline` mechanism, we seperate the inputing/outputing part of model training/testing out of the `dist_train.py`. This operation enables us to create much more complex data inputs and post/pre-processions, without worrying about the quality and tidy of our codes. After implementing your pipeline, you just need to specify it in your `config.py`, more details about how to specify it please refer to [Elements of `config.py`](https://airscker.github.io/DeepMuon/tutorials/index.html#/config/config?id=model).
