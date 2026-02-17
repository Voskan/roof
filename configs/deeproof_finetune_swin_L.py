# DeepRoof-2016: Fine-tuning Configuration
_base_ = [
    './deeproof_production_swin_L.py',
]

# 1. Training Parameters
# ----------------------
max_iters = 20000        # Total training steps (~30-50 epochs depending on dataset size)
val_interval = 2000     # Evaluate every 2000 steps
batch_size = 4          # Images per GPU (Total 16 if 4 GPUs)
input_res = (1024, 1024) # High-quality resolution for roof detail

# 2. Load Pre-trained Weights
# Pre-trained on ImageNet-22K then building-specific datasets
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-l-in22k-384x384_8x2_50e_ade20k/mask2former_swin-l-in22k-384x384_8x2_50e_ade20k_20221204_194411-d00d9841.pth'

# 3. Adjust Learning Rate for Fine-tuning
optimizer = dict(
    type='AdamW',
    lr=0.00005, # Stable LR for fine-tuning
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999)
)

optim_wrapper = dict(
    _delete_=True, # Overwrite production wrapper
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2)
)

# 3. Training Schedule Adjustment
# Fewer iterations for fine-tuning
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='PolyLR',
        power=0.9,
        begin=500,
        end=20000,
        eta_min=1e-7,
        by_epoch=False,
    )
]

# 4. Data Pipeline: Increased Augmentation for Fine-tuning
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='GoogleMapsAugmentation', prob=0.8), # More aggressive probe
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline
    )
)
