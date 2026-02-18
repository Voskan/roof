# DeepRoof-2026: Master Scratch Training Configuration
_base_ = [
    './deeproof_production_swin_L.py',
]

# 1. Training Parameters (SCRATCH PROFILE)
# ----------------------------------------
max_iters = 80000        # Shorter: fine-tuning from iter_40000 weights
val_interval = 5000      # Evaluate every 5k steps
batch_size = 8           # Doubled from 4—8 to use more VRAM (~60-70%)
input_res = (512, 512)     # Native OmniCity image size

# 2. Resume from iter_40000 checkpoint (load weights only, reset optimizer)
# -------------------------
load_from = 'work_dirs/swin_l_scratch_v1/iter_40000.pth'
resume = False  # Load weights only, fresh optimizer for new loss weights

# 3. Optimized Optimizer & Scale
# ------------------------------
optimizer = dict(
    type='AdamW',
    lr=0.00005,          # Lower LR for fine-tuning (half of scratch)
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
        start_factor=0.1,   # Gentle warmup (already pretrained)
        by_epoch=False,
        begin=0,
        end=500             # Short warmup
    ),
    dict(
        type='PolyLR',
        power=0.9,
        begin=500,
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
        max_keep_ckpts=5,
        save_best='mIoU',
        published_keys=['meta', 'state_dict']
    ),
    logger=dict(type='LoggerHook', interval=50)
)

# 6. Data Pipeline
# ----------------
# DeepRoofDataset performs augmentation internally in __getitem__.
train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(pipeline=[])
)
