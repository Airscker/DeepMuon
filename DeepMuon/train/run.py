'''
Author: airscker
Date: 2022-10-07 21:35:54
LastEditors: airscker
LastEditTime: 2023-02-08 17:24:08
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import argparse
import DeepMuon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dist', action='store_true')
    parser.add_argument('-g', '--gpus', nargs='+', default=0)
    parser.add_argument('-p', '--port', default=22911)
    parser.add_argument('-c', '--config', default='')
    parser.add_argument('-t', '--train', default='dist_train.py')
    args = parser.parse_args()
    pkg_path = DeepMuon.__path__[0]
    msg = os.path.join(pkg_path.split('DeepMuon')[0], 'LICENSE.txt')
    if args.dist:
        assert len(
            args.gpus) > 1, f'More than 1 GPUs expected, but only one GPU: {args.gpus} given'
        env = ''
        for i in range(len(args.gpus)):
            env += f'{args.gpus[i]}'
            if i+1 < len(args.gpus):
                env += ','
        if not args.train.endswith('.py'):
            args.train += '.py'
        file = os.path.join(pkg_path, 'train', args.train)
        command = f'CUDA_VISIBLE_DEVICES={env} torchrun --nproc_per_node={len(args.gpus)} --master_port {args.port} {file} --config {args.config} --msg {msg}'
    else:
        file = os.path.join(pkg_path, 'train', 'train.py')
        command = f'python {file} --config {args.config} --msg {msg}'
    os.system(command)
    # print(command)


if __name__ == '__main__':
    main()
