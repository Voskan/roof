
# DeepRoof-2016: Production Configuration for A100 Cluster (4 GPUs)
_base_ = [
    './swin/swin_large.py', # Inherit Backbone & Neck from Swin-L
]

# 1. Model Configuration
model = dict(
    type='DeepRoofMask2Former', # Our Multi-Task Model
    
    # Custom Geometry Head
    geometry_head=dict(
        type='GeometryHead',
        embed_dims=256,
        num_layers=3,
        hidden_dims=256
    ),
    
    # Multi-Task Loss Weights (A100 Optimized)
    # High priority on Geometry and Segmentation
    geometry_loss_weight=5.0,
    
    decode_head=dict(
        type='Mask2FormerHead',
        # Standard Mask2Former params
        # ... (inherited/default logic)
        loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=5.0),
        loss_dice=dict(type='DiceLoss', loss_weight=5.0),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0)
    )
)

# 2. Data Pipeline & Dataloader
dataset_type = 'DeepRoofDataset'
data_root = 'data/OmniCity/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='GoogleMapsAugmentation', prob=0.5), # Corrected Vector Rotation Fix
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4, # Samples per GPU (4 GPUs = Total Batch Size 16)
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        pipeline=train_pipeline
    )
)

# 3. Optimizer & Scheduler (A100 Best Practice)
optimizer = dict(
    type='AdamW', 
    lr=0.0001, 
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999)
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2) # Stable gradients for geometry
)

param_scheduler = [
    # Linear Warmup for 1500 iterations
    dict(
        type='LinearLR', 
        start_factor=0.001, 
        by_epoch=False, 
        begin=0, 
        end=1500
    ),
    # Poly Decay for the rest of 100k iters
    dict(
        type='PolyLR',
        power=0.9,
        begin=1500,
        end=100000,
        min_lr=1e-6,
        by_epoch=False,
    )
]

# 4. Runtime Config
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=5000)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, max_keep_ckpts=3)
)
