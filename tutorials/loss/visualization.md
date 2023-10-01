# Visualize evaluation metrics using `@EnableVisualization`

Some evaluation metrics such as `R2_Value`, we can use default curve plotting tools of `DeepMuon` to visualize the trend of metrics during training. But when it comes to testing mode, we want to get such as [Joint Distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution) between predicted and true labels. As you see, this kind request is not suitable for other tasks which don't need to plot this. So we need to find a way to help us dicide whether to visualize this metric, how to visualize this kind of metric, should this metric being included into `NNHS` report of `TensorBoard`?

Fortunately, we implemented a visualization register `@EnableVisualization` to help you easily solve prolems we mentioned above. You just need to use it decorate every evaluation metric you want with one line of code. Here we give you an example of evaluation metric `R2_Value` to help you understand how to use it.

```python
from sklearn.metrics import r2_score
from DeepMuon.tools import EnableVisualiaztion
from DeepMuon.tools import R2JointPlot
@EnableVisualiaztion(Name="R2 Score",NNHSReport=True,TRTensorBoard=True,TRCurve=True,TSPlotMethod=R2JointPlot)
def R2Value(scores, labels):
    scores=np.array(scores).reshape(-1)
    labels=np.array(labels).reshape(-1)
    return r2_score(labels, scores)
```

There are several parameters of visulization register as listed here:

- Name [str]: The name to be shown in the visualized report (Not for report's filename), if `None` is given, the `Name` will be the name of evaluation method.
- NNHSReport [bool]: Whether to show the result in NNHSReport, `False` defaultly. More details about NNHS report please refer to [NNHS Report](https://airscker.github.io/DeepMuon/tutorials/index.html#/log/nnhs_report)
- TRTensorBoard [bool]: Whether to show the result in TensorBoard during training, Tensorboard log information is recorded by `DeepMuon.tools.AirVisual.tensorboard_plot`, `False` defaultly. More details about Tensorboard please refer to [Usage of TensorBoard](https://airscker.github.io/DeepMuon/tutorials/index.html#/log/tensorboard)
- TRCurve [bool]: Whether to show the result in the training curve at the end of training, the curve is plotted by `DeepMuon.tools.AirVisual.plot_curve`, `False` defaultly. More details about TRCurve please refer to [Plotting training curves based on JSON log](https://airscker.github.io/DeepMuon/tutorials/index.html#/log/tr_curve)
- TSPlotMethod [Callable]: if specified, the tested result will be plotted by this method, otherwise, the result will not be plotted. And `TSPlotMethod` MUST accept the same parameters as the decorated evaluation method, what's more, another parameter `save_path` should be included to specify the directory to save the plot. More details about TSPlotMethod please refer to [Evaluation Metric Visualizing Methods](https://airscker.github.io/DeepMuon/tutorials/index.html#/vis_metric)

ATTENTION: After version 1.23.91, the metric-evaluating methods stored in module `evaluation.py` should be decorated by VisualiaztionRegister `DeepMuon.tools.AirDecorators.EnableVisualiaztion` to be properly visualized. Otherwise all visualization commands for the UN-decorated evaluation metric will be **disabled** to avoid some fetal errors when using default visualizing methods, the disabling action is executed within `dist_train.evaluation`.
