'''
Author: airscker
Date: 2023-02-15 19:56:03
LastEditors: airscker
LastEditTime: 2023-02-15 20:12:12
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved.
'''

import os
import warnings

import torch
from torch import nn


def model_profile(input: torch.Tensor, model: nn.Module, workdir: str):
    if not os.path.exists(workdir):
        os.makedirs(workdir)
        warnings.warn(
            f"{workdir} does not exist, please check the path of workdir again. To avoid errors, the workdir given is created automatically")
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=True) as prof_ana:
        pred = model(input)
    table = prof_ana.table()
    prof_ana.export_chrome_trace(os.path.join(workdir, 'model_profile.json'))
    return table
