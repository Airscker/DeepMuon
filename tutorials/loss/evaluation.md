# Evaluation based on scores & labels

To estimate the model's performance, we have several different metrics to use. Here we introduce how to add your customized evaluation metrics to the DeepMuon.

## File path

In the latest edition of DeepMuon, we require you to put your own metrics in the file `evaluation.py`, under the installation path of DeepMuon, folder `loss_fn`.

## Metrics template

We require all metrics' parameters should be like this:

```python
from DeepMuon.tools import EnableVisualization
@EnableVisualization(Name="MetricX",NNHSReport=True,TRTensorBoard=True,TRCurve=True,TSPlotMethod=R2JointPlot)
def metric(scores,label,*args,**kwargs):
    '''Customized algorithm'''
    return result
```

ATTENTION: After version 1.23.91, the metric-evaluating methods stored in module `evaluation.py` should be decorated by VisualiaztionRegister `DeepMuon.tools.AirDecorators.EnableVisualiaztion` to be properly visualized. Otherwise some evaluation metrics may occur fetal errors when using default plotting methods. More details about `@EnableVisualization` please refer to section [Visualize evaluation metrics using `@EnableVisualization`](https://airscker.github.io/DeepMuon/tutorials/index.html#/loss/visualization).

## After customizing

You should add the name of your metrics into the file `__init__.py`, please make sure your name has no conflicts with other metrics.
