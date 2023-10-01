# Evaluation Metric Visualizing Methods

If you haven't read the post [Visualize evaluation metrics using `@EnableVisualization`](https://airscker.github.io/DeepMuon/tutorials/index.html#/loss/visualization), we recommend you to read it first before goning on.

To help you understand how to write your own metric visualizing methods within `DeepMuon.tools.AirVisual`, let's see an example first:

```python
def R2JointPlot(scores,labels,save_path:str='./',tag:str='TS'):
    chart=pd.DataFrame({'Pedicted':scores,'True':labels})
    plt.figure(figsize=(30,30))
    sns.pairplot(chart)
    plt.savefig(os.path.join(save_path,f'{tag}_R2Pair.jpg'),dpi=300)
    sns.set(font_scale=1.5)
    sns.jointplot(x='Pedicted',y='True',data=chart,kind='reg')
    plt.savefig(os.path.join(save_path,f'{tag}_R2Joint.jpg'),dpi=300)
```

This is a method used for plotting Joint Distribution between predicted and true labels. To make sure everything goes well when you are using `@EnableVisualization`, you MUST specify at least four parameters of visualizing method:

- scores: Result predicted by model. After model testing pipeline is finished, you will get `scores.npy` under your model's workdir, the scores will be read from this file.
- labels: True labels of testing dataset, similar to `scores`, `labels` will be loaded by reading `labels.npy`.
- save_path: The path to save the plotted figure, most of time, we don't need to specify that, and the figures will be automatically saved under your `workdir/Figure`, but that doesn't mean we can omit this when we are writing code, the main process need to know where workdir is and give it your visualizing method.
- tag: The prefix tag to be used to name your saved figures or other things, also you must keep it when you are writing your own codes.

The visulization methods are built for `@EnableVisualization` registers to determine how to visualize specific evaluation metrics. These methods are exevuted after finishing model testing pipeline, within function `dist_train.evaluation`.
