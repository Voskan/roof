# DeepRoof-2026: Master Scratch Training Configuration
_base_ = [
    './deeproof_production_swin_L.py',
]

# 1. Training Parameters (SCRATCH PROFILE)
# ----------------------------------------
max_iters = 160000       # Extended training for full convergence from scratch
val_interval = 8000      # Evaluate every 8k steps
batch_size = 4           # Samples per GPU (total 16 on 4x A100)
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

# 4. Long-Term Schedule (ITER BASED)
# ----------------------------------
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=val_interval
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        # Prevent ultra-low LR plateau at start; warm up over first 1k iters.
        start_factor=0.01,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='PolyLR',
        power=0.9,
        begin=1000,
        end=max_iters,
        eta_min=1e-6,
        by_epoch=False
    )
]

# Keep LR correct when training on fewer than 4 GPUs.
auto_scale_lr = dict(enable=True, base_batch_size=16)

# 5. Advanced Checkpointing (Best + Last + Intermediaries)
# --------------------------------------------------------
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=val_interval,
        max_keep_ckpts=5,    # Keep last 5 snapshots
        save_best='mIoU',    # ALWAYS keep the best performing model
        published_keys=['meta', 'state_dict']
    ),
    logger=dict(type='LoggerHook', interval=50) # Print stats every 50 batches
)

# 6. Data Pipeline
# ----------------
# DeepRoofDataset performs augmentation internally in __getitem__.
train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(pipeline=[])
)
