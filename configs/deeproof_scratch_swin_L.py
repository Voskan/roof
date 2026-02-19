# DeepRoof-2026: Fine-Tune Configuration (loaded from swin_l_scratch_v1/iter_40000)
_base_ = [
    './deeproof_production_swin_L.py',
]

# 1. Training Parameters (FINE-TUNE PROFILE)
# ----------------------------------------
max_iters = 80000        # Fine-tuning from iter_40000 weights
val_interval = 5000      # Evaluate every 5k steps
batch_size = 16          # A100 80GB ~70% utilization

# 2. Resume from iter_40000 checkpoint (load weights only, reset optimizer)
# -------------------------
load_from = '/workspace/roof/work_dirs/swin_l_scratch_v1/iter_40000.pth'
resume = False  # Load weights only, fresh optimizer for new loss weights

# 3. Optimized Optimizer & Scale
# ------------------------------
optimizer = dict(
    type='AdamW',
    lr=0.00005,          # Lower LR for fine-tuning (half of scratch 0.0001)
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999)
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    # FIX Bug #1: max_norm=0.01 was 100x too small — completely killed geometry gradients.
    # Standard Mask2Former / mmdet value is 1.0.
    # At 0.01 the effective LR for geometry was ~0.0000001, causing full class collapse.
    clip_grad=dict(max_norm=1.0, norm_type=2)
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

# FIX Bug #2: param_scheduler MUST be completely replaced (not merged) here.
# Without _delete_=True the base config's PolyLR(end=100000) remains active,
# creating two simultaneous PolyLR schedules that multiply together.
# The combined decay caused LR to drop ~2x faster than intended, collapsing
# training after iter 5000.  Adding _delete_=True removes the base schedulers.
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,    # Gentle warmup (already pretrained weights)
        by_epoch=False,
        begin=0,
        end=500             # Short warmup: 500 iters
    ),
    dict(
        type='PolyLR',
        power=0.9,
        begin=500,
        end=max_iters,      # Aligned with actual training span (80000)
        eta_min=1e-6,
        by_epoch=False
    )
]

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

# Keep LR correct when training on fewer than 4 GPUs.
# base_batch_size=16 matches actual total batch size so scale factor = 1.0 always.
auto_scale_lr = dict(enable=True, base_batch_size=16)
