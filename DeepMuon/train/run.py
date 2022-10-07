'''
Author: airscker
Date: 2022-10-07 21:35:54
LastEditors: airscker
LastEditTime: 2022-10-07 23:39:30
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist',action='store_true')
    parser.add_argument("--gpus",nargs='+',default=0)
    parser.add_argument('-p','--port',default=22918)
    parser.add_argument('-c','--config',default='/home/dachuang2022/Yufeng/DeepMuon/config/Hailing/Vit.py')
    args = parser.parse_args()
    msg=os.path.relpath(__file__,os.getcwd()).split('DeepMuon')[0]+'LICENSE.txt'
    if args.dist:
        assert len(args.gpus)>1,f'More than 1 GPUs expected, but only one GPU: {args.gpus} given'
        env=''
        for i in range(len(args.gpus)):
            env+=f'{args.gpus[i]}'
            if i+1 <len(args.gpus):
                env+=','
        file=os.path.relpath(__file__,os.getcwd()).replace(os.path.basename(__file__),'dist_train.py')
        command=f'CUDA_VISIBLE_DEVICES={env} torchrun --nproc_per_node={len(args.gpus)} --master_port {args.port} {file} --config {args.config} --msg {msg}'
    else:
        file=os.path.relpath(__file__,os.getcwd()).replace(os.path.basename(__file__),'train.py')
        command=f'python {file} --config {args.config} --msg {msg}'
    os.system(command)

if __name__=='__main__':
    main()