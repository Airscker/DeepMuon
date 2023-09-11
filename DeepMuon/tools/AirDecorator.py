'''
Author: airscker
Date: 2023-09-02 22:05:13
LastEditors: airscker
LastEditTime: 2023-09-11 18:22:31
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import warnings
from functools import wraps
from typing import Callable
from ..tools.AirVisual import plot_curve

class EnableVisualiaztion:
    """
    ## Register visualization permissions for evaluation methods.

    ### Args:
        - Name [str]: The name to be shown in the visualized report (Not for report's filename), 
            if `None` is given, the name will be the name of evaluation method.
        - NNHSReport [bool]: Whether to show the result in NNHSReport, `False` defaultly.
        - TRTensorBoard [bool]: Whether to show the result in TensorBoard during training, 
            Tensorboard log information is recorded by `DeepMuon.tools.AirVisual.tensorboard_plot`, `False` defaultly.
        - TRCurve [bool]: Whether to show the result in the training curve at the end of training, 
            the curve is plotted by `DeepMuon.tools.AirVisual.plot_curve`, `False` defaultly.
        - TSPlotMethod [Callable]: if specified, the tested result will be plotted by this method, 
            otherwise, the result will not be plotted. And `TSPlotMethod` MUST accept the same parameters as the decorated evaluation method, 
            what's more, another parameter `save_path` should be included to specify the directory to save the plot.

    ### Returns:
        - result: The result of the decorated evaluation method.
        - Five values of the decorator's parameters, which are `Name`, `NNHSReport`, `TRTensorBoard`, `TRCurve`, `TSPlotMethod`, to be used by further procession.
        - 'VisualizationRegistered': A string to indicate that the decorated method is registered for visualization.
    """
    def __init__(self,
                 Name:str=None,
                 NNHSReport:bool=False,
                 TRTensorBoard:bool=False,
                #  TRTensorBoardMethod:Callable=None,
                 TRCurve:bool=False,
                #  TRCurveMethod:Callable=None,
                #  TSPlot:bool=False,
                 TSPlotMethod:Callable=None):
        self.Name = Name
        self.NNHSReport = NNHSReport
        self.TRTensorBoard=TRTensorBoard
        self.TRCurve=TRCurve
        # if TRTensorBoard:
        #     if TRTensorBoardMethod is None:
        #         TRTensorBoardMethod=tensorboard_plot
        # else:
        #     TRTensorBoardMethod=None
        # if TRCurve:
        #     if TRCurveMethod is None:
        #         TRCurveMethod=plot_curve
        # else:
        #     TRCurveMethod=None
        # if TSPlot and TSPlotMethod is None:
        #     warnings.warn(f"Metric plotting in test mode enabled for {Name}, however no plotting method is provided.")
        self.TSPlotMethod = TSPlotMethod
        # self.TRCurveMethod = TRCurveMethod
        # self.TRTensorBoardMethod = TRTensorBoardMethod
    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # print(f'Call {self.func.__name__}()')
            if self.Name is None:
                self.Name=func.__name__
            result=func(*args, **kwargs)
            return result,self.Name,self.NNHSReport,self.TRTensorBoard,self.TRCurve,self.TSPlotMethod,'VisualizationRegistered'
        return wrapper