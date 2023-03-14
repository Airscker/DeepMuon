# Evaluation based on scores & labels

To estimate the model's performance, we have several different metrics to use. Here we introduce how to add your customized evaluation metrics to the DeepMuon.

## File path

In the latest edition of DeepMuon, we require you to put your own metrics in the file `evaluation.py`, under the installation path of DeepMuon, folder `loss_fn`.

## Metrics template

We require all metrics' parameters should be like this:

```python
def metric(scores,label,*args,**kwargs):
    '''Customized algorithm'''
    return result
```

## After customizing

You should add the name of your metrics into the file `__init__.py`, please make sure your name has no conflicts with other metrics.