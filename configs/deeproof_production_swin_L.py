
# DeepRoof-2016: Production Configuration for A100 Cluster (4 GPUs)
_base_ = [
    './swin/swin_large.py', # Inherit Backbone & Neck from Swin-L
]

# 0. Custom Imports
custom_imports = dict(
    imports=[
        'mmdet.models',
        'deeproof.models.backbones.swin_v2_compat',
        'deeproof.datasets.roof_dataset',
        'deeproof.models.heads.mask2former_head',
        'deeproof.models.deeproof_model',
        'deeproof.models.heads.geometry_head'
    ],
    allow_failed_imports=False)

# 1. Shared Model Settings
num_classes = 3
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

# 1. Model Configuration
model = dict(
    type='DeepRoofMask2Former', # Our Multi-Task Model
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode='whole'),
    
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
        type='DeepRoofMask2FormerHead',
        in_channels=[192, 384, 768, 1536],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(type='mmdet.DiceCost', weight=5.0, pred_act=True, eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler')))
)

# 2. Data Pipeline & Dataloader
dataset_type = 'DeepRoofDataset'
data_root = 'data/OmniCity/'

# DeepRoofDataset handles loading and augmentation in its custom __getitem__.
train_pipeline = []

train_dataloader = dict(
    batch_size=4, # Samples per GPU (4 GPUs = Total Batch Size 16)
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        normal_suffix='.npy',
        pipeline=train_pipeline
    )
)

val_pipeline = []

# Test/inference pipeline: standard mmseg pipeline for external images.
# Resize to training resolution (512×512 = native OmniCity size) to keep
# the feature distribution consistent with what the model learned.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='PackSegInputs'),
]

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        normal_suffix='.npy',
        pipeline=val_pipeline
    )
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_dataloader = val_dataloader
test_evaluator = val_evaluator

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
        eta_min=1e-6,
        by_epoch=False,
    )
]

# 4. Runtime Config
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=5000)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, max_keep_ckpts=3)
)
