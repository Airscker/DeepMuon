'''
Author: Airscker
Date: 2022-08-26 21:23:01
LastEditors: airscker
LastEditTime: 2023-02-16 18:02:43
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import os
import json
import numpy as np


class LOGT(object):
    """
    ## Initialize the logger .

    ### Args:
        - log_dir: The path to save the log file. Defaults to './'.
        - logfile: The name of the log file. Defaults to be log.log'.
        - new: Whether to override the existed log file
    """

    def __init__(self, log_dir='./', logfile='log.log', new=False):
        if not os.path.exists(log_dir) and log_dir != '':
            print(f'{log_dir} Created')
            os.makedirs(log_dir)
        self.logfile = os.path.join(log_dir, logfile)
        self.jsonfile = os.path.join(log_dir, f'{logfile}.json')
        if new == True and os.path.exists(self.logfile):
            os.remove(self.logfile)

    def log(self, message: str, show=True):
        """
        ## write a message to the logfile

        ### Args:
            - message: the message to be saved
            - show : show the message in the terminal. Defaults to be True.
        """
        if show == True:
            print(message)
        with open(self.logfile, 'a+')as log:
            log.write(f'{message}\n')
        log.close()


class LOGJ(object):
    """
    ## Initialize the json logger to record data by json files.

    ### Args:
        - log_dir: The path to save the log file. Defaults to './'.
        - logfile: The name of the log file. Defaults to be log.log'.
        - new: Whether to override the existed log file
    """

    def __init__(self, log_dir='./', logfile='log.log.json', new=False) -> None:
        assert logfile.endswith(
            '.json'), f"logfile must be json file. However, {logfile} given"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.logfile = os.path.join(log_dir, logfile)
        if new and os.path.exists(self.logfile):
            os.remove(self.logfile)

    def log(self, message=dict):
        """
        ## Write the data to the json logfile.

        ### Args:
            - message: the data to be written to the logfile. Must be dictionary format.
        """
        for key in message.keys():
            value = message[key]
            if isinstance(value, np.ndarray):
                message[key] = value.tolist()
        message_json = json.dumps(message)
        with open(self.logfile, 'a+')as f:
            f.write(f"{message_json}\n")
        f.close()
