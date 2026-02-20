
# DeepRoof-2026: Production Configuration for A100 Cluster (4 GPUs)
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

    # Absolute Ideal: Use PhysicallyWeightedNormalLoss
    geometry_loss=dict(
        type='PhysicallyWeightedNormalLoss',
        loss_weight=2.0,
        azimuth_weight=1.5  # Heavy focus on slope-aware azimuth
    ),

    decode_head=dict(
        type='DeepRoofMask2FormerHead',
        in_channels=[192, 384, 768, 1536],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=300,
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
                        num_levels=3,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(
                num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            num_feats=128, normalize=True),
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
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type='DeepRoofCrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            # bg=1, flat=1, sloped=10, no_obj=0.1
            # FIX: Increased sloped weight from 3->10 to combat extreme class imbalance.
            class_weight=[1.0, 1.0, 10.0, 0.1]),
        loss_mask=dict(
            type='DeepRoofDiceLoss',
            loss_weight=5.0,
            eps=1e-6,
            reduction='mean'),
        loss_dice=dict(
            type='DeepRoofDiceLoss',
            loss_weight=5.0,
            eps=1e-6,
            reduction='mean'),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    # ClassificationCost weight=2.0 gives cls matching higher priority
                    # vs mask/dice costs. Per-class imbalance is handled by loss_cls.class_weight.
                    # Note: mmdet.ClassificationCost does not accept a 'class_weight' kwarg.
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
        image_size=(1024, 1024),
        pipeline=train_pipeline
    )
)

val_pipeline = []

# Test/inference pipeline: standard mmseg pipeline for external images.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='PackSegInputs'),
]

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in [0.75, 1.0, 1.25]
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='PackSegInputs')]
        ])
]

tta_model = dict(type='SegTTAModel')

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
        image_size=(1024, 1024),
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
    # FIX Bug #1 (production): max_norm=1.0 is the standard for Mask2Former.
    # 0.01 was 100x too aggressive — clipped geometry gradients to near-zero.
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0)
        }
    )
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
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=3,
        save_best='mIoU',
    )
)
