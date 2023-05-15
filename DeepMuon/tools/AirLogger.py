'''
Author: Airscker
Date: 2022-08-26 21:23:01
LastEditors: airscker
LastEditTime: 2023-05-16 00:56:57
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import os
import json
import numpy as np
from typing import Union
from prettytable import PrettyTable

def convert_table(dictionary:dict=None):
    info=[list(dictionary.keys()),list(dictionary.values())]
    table=PrettyTable()
    table.add_column('INFO',info[0])
    table.add_column('VALUE',info[1])
    return str(table)

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
            # print(f'{log_dir} Created')
            os.makedirs(log_dir)
        self.logfile = os.path.join(log_dir, logfile)
        self.jsonfile = os.path.join(log_dir, f'{logfile}.json')
        if new == True:
            if os.path.exists(self.logfile):
                os.remove(self.logfile)
            if os.path.exists(self.jsonfile):
                os.remove(self.jsonfile)

    def log(self, message:Union[dict,str], show=True, json_log=False):
        """
        ## write a message to the logfile

        ### Args:
            - message: the message to be saved.
            - show: show the message in the terminal. Defaults to be True.
            - json_log: whether to record the data into json file.
        """
        if json_log:
            json_info={}
            for key in message.keys():
                value = message[key]
                if isinstance(value, np.ndarray):
                    json_info[key] = value.tolist()
                else:
                    json_info[key]=value
            message_json = json.dumps(json_info)
            with open(self.jsonfile, 'a+')as f:
                f.write(f"{message_json}\n")
            f.close()
        if show == True:
            if isinstance(message,dict):
                message=convert_table(message)
            print(message)
        with open(self.logfile, 'a+')as f:
            f.write(f'{message}\n')
        f.close()