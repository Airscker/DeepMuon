cd /data/JoyceW/VST_fusion_dataset/new_framework
pip install -v -e ./ -i https://pypi.tuna.tsinghua.edu.cn/simple
cd /data/JoyceW/VST_fusion_dataset/DM_workdir/Attr_sax_0.994_11cls
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 22106 --use_env /data/JoyceW/VST_fusion_dataset/new_framework/DeepMuon/train/dist_train_vst.py --config config.py --test /data/JoyceW/VST_fusion_dataset/workdir/4ch_cine_11cls/spacing_0.994/TEST004/best_f1_score_epoch_166_deepmuon.pth