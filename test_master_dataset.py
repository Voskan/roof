import sys
import os
import torch
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

try:
    from mmengine.config import Config
    from mmseg.registry import DATASETS
    import deeproof.datasets.roof_dataset

    cfg = Config.fromfile('configs/deeproof_production_swin_L.py')
    
    # We enforce MasterRoofDataset explicitly just in case
    cfg.train_dataloader.dataset.data_root = 'data/MassiveMasterDataset/'
    cfg.train_dataloader.dataset.ann_file = 'data/MassiveMasterDataset/train.txt'

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    print(f'Dataset built successfully. Total valid images: {len(dataset)}')
    
    if len(dataset) > 0:
        sample = dataset[0]
        inputs = sample['inputs']
        print(f'Inputs shape: {inputs.shape}')
        
        sem_seg = sample['data_samples'].gt_sem_seg.data
        print(f'Semantic mask shape: {sem_seg.shape}')
        print(f'Unique semantic IDs found in first batch: {torch.unique(sem_seg)}')
        
        inst_data = sample['data_samples'].gt_instances
        if inst_data.masks.numel() > 0:
            print(f'Instances extracted: {len(inst_data.labels)} with labels {inst_data.labels.tolist()}')
        else:
            print('No instances found in this sample.')
            
        print('SUCCESS: Pytorch Data pipeline gracefully loaded the Master Dataset!')
    else:
        print('Dataset empty. Skipping item test.')
except Exception as e:
    import traceback
    traceback.print_exc()
    print('Failed')
