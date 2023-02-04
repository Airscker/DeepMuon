from nni import Experiment
import os


def model_optim():
    search_space = {
        'weight_decay': {'_type': 'choice', '_value': [0.01, 0.05, 0.1, 0.2]},
        'lr': {'_type': 'choice', '_value': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03]},
        'dropout': {'_type': 'choice', '_value': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]},
    }
    exp_name = 'CNNLSTM'
    print(f'Current PID: {os.getpid()}')
    experiment = Experiment('local')
    experiment.config.trial_concurrency = 1
    experiment.config.experiment_name = exp_name
    experiment.config.trial_command = f'CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29521 --use_env dist_train_lstm.py --config /data/JoyceW/VST_fusion_dataset/CNNLSTM/DeepMuon/config/CMR_VST/CNNLSTM.py --msg /data/JoyceW/VST_fusion_dataset/CNNLSTM/LICENSE.txt'
    # experiment.config.trial_code_directory='/data/Airscker/UBK_VST_Airscker/CNNSLTM/4CH_train'
    experiment.config.trial_code_directory = '/data/JoyceW/VST_fusion_dataset/CNNLSTM/DeepMuon/train'
    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_trial_number = 100
    # experiment.config.max_trial_duration='20s'
    # experiment.config.experiment_working_directory='/data/Airscker/UBK_VST_Airscker/CNNSLTM/nni_experiments'
    experiment.config.experiment_working_directory = '/data/JoyceW/VST_fusion_dataset/CNNLSTM/workdir/nni_experiments'
    print(experiment.config)
    experiment.run(14001)
    experiment.stop()
    return 0


model_optim()
