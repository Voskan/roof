# DeepRoof-2026: Master Scratch Training Configuration
_base_ = [
    './deeproof_production_swin_L.py',
]

# 1. Training Parameters (SCRATCH PROFILE)
# ----------------------------------------
max_iters = 160000        # Extended training for full convergence from scratch
val_interval = 8000      # Evaluate every 8k steps
batch_size = 4           # Samples per GPU (Total 16 on 4x A100)
input_res = (1024, 1024)

# 2. Force Scratch Training
# -------------------------
load_from = None
resume_from = None

# 3. Optimized Optimizer & Scale
# ------------------------------
optimizer = dict(
    type='AdamW',
    lr=0.0001,           # Base LR for scratch training (higher than fine-tune)
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999)
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2)
)

# 4. Long-Term Schedule
# ---------------------
train_cfg = dict(
    type='IterBasedTrainLoop', 
    max_iters=160000, 
    val_interval=8000
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=False,
        begin=0,
        end=1500           # Longer warmup for scratch stability
    ),
    dict(
        type='PolyLR',
        power=0.9,
        begin=1500,
        end=160000,
        min_lr=1e-7,
        by_epoch=False
    )
]

# 5. Advanced Checkpointing (Best + Last + Intermediaries)
# --------------------------------------------------------
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10000,      # Save periodic snapshots
        max_keep_ckpts=5,    # Keep last 5 snapshots
        save_best='mIoU',    # ALWAYS keep the best performing model
        published_keys=['meta', 'state_dict']
    ),
    logger=dict(type='LoggerHook', interval=100)
)

# 6. High-Intensity Data Augmentation
# -----------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='GoogleMapsAugmentation', prob=0.8), # Maximum intensity for scratch
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline
    )
)
