# DeepRoof-2026: Swin Transformer V2-Large Backbone Configuration

# Pre-trained Weights Source:
# Primary: SatMAE (Satellite Masked Autoencoder) - Best for Domain Adaptation
# Fallback: ImageNet-22k - Standard High-Performance Initialization

# Checkpoint URL for Swin-V2-Large (ImageNet-22k pre-trained, 384x384 input)
# We will fine-tune this on 1024x1024 input.
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'

# If SatMAE weights are available locally, uncomment the following line:
# checkpoint_file = 'pretrain/satmae_swin_large_vit.pth'

model = dict(
    backbone=dict(
        type='SwinTransformerV2',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        # Keep window_size aligned with the official pretrained checkpoint
        # (swin_large_patch4_window12_384_22k) to avoid state_dict shape mismatches.
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,   # Set True if GPU memory is tight (Gradient Checkpointing)
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        
        # SwinV2 Specifics for High Res
        pretrained_window_sizes=[12, 12, 12, 6] # Mismatch with window_size=24 is handled by interpolation in V2
    ),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536], # Swin-Large channels
        out_channels=256,
        num_outs=4),
)
