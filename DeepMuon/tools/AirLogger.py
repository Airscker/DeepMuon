'''
Author: Airscker
Date: 2022-08-26 21:23:01
LastEditors: airscker
LastEditTime: 2022-11-20 01:03:08
Description: NULL

Copyright (c) 2022 by Airscker, All Rights Reserved. 
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
import time
import json
class LOGT(object):
    """
    ## Initialize the logger .

        Args:
            log_dir: The path to save the log file. Defaults to './'.
            logfile: The name of the log file. Defaults to be log.log'.
            new: Whether to override the existed log file
        Return:
            The path of the log file
    """
    def __init__(self, log_dir='./',logfile='log.log',new=False):
        if not os.path.exists(log_dir) and log_dir!='':
            print(f'{log_dir} Created')
            os.makedirs(log_dir)
        self.logfile = os.path.join(log_dir, logfile)
        if new==True and os.path.exists(self.logfile):
            os.remove(self.logfile)
    def log(self, message:str,show=True):
        """write a message to the logfile

        Args:
            message: the message to be saved
            show : show the message in the terminal. Defaults to be True.

        """
        if show==True:
            print(message)
        with open(self.logfile,'a+')as log:
            log.write(f'{message}\n')
        return self.logfile
# print(__file__)