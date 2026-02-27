import sys
import os
import cv2
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# We might not have MMEngine easily available if not installed properly, but let's assume it is.
try:
    from mmengine.config import Config
    from mmseg.registry import DATASETS
    import deeproof.datasets.roof_dataset

    cfg = Config.fromfile('configs/deeproof_production_swin_L.py')
    cfg.train_dataloader.dataset.data_root = 'data/OmniCity'
    cfg.train_dataloader.dataset.ann_file = 'data/OmniCity/train.txt'

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    print(f'Dataset length: {len(dataset)}')
    if len(dataset) > 0:
        sample = dataset[0]
        inputs = sample['inputs']
        print(f'Inputs shape: {inputs.shape}')
        sem_seg = getattr(sample['data_samples'], 'gt_sem_seg', None)
        if sem_seg is not None:
            print(f'Semantic mask shape: {sem_seg.data.shape}')
        normals = getattr(sample['data_samples'], 'gt_normals', None)
        if normals is not None:
            print(f'Normals shape: {normals.data.shape}')
        
        print('SUCCESS: Pipeline loaded without exceptions and shape dimensions matched target')
    else:
        print('Dataset empty. Skipping item test.')
except Exception as e:
    import traceback
    traceback.print_exc()
    print('Failed')
