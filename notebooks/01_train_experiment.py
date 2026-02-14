
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

# Ensure custom modules are registered
import deeproof.models.deeproof_model
import deeproof.models.heads.geometry_head
import deeproof.models.losses

def run_experiment():
    print("Starting DeepRoof-2026 Training Experiment...")
    
    # Register modules
    register_all_modules(init_default_scope=False)
    
    # 1. Setup Configuration for Model
    # We define a minimal config dictionary for testing
    model_cfg = dict(
        type='DeepRoofMask2Former',
        data_preprocessor=dict(
            type='SegDataPreProcessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_val=0,
            seg_pad_val=255),
        backbone=dict(
            type='SwinTransformerV2',
            embed_dims=96, # Reduced for speed in test
            depths=[2, 2, 6, 2], # Reduced depth
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False),
        decode_head=dict(
            type='Mask2FormerHead',
            in_channels=[96, 192, 384, 768], # Matched to reduced swin above
            strides=[4, 8, 16, 32],
            feat_channels=256,
            out_channels=256,
            num_classes=2, # Roof vs Background
            num_queries=100,
            num_transformer_feat_level=3,
            pixel_decoder=dict(
                type='MSDeformAttnPixelDecoder',
                num_outs=3,
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU'),
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=3,
                            num_points=4),
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=256,
                            feedforward_channels=1024,
                            num_fcs=2,
                            act_cfg=dict(type='ReLU', inplace=True),
                        ),
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                positional_encoding=dict(
                    type='SinePositionalEncoding', num_feats=128, normalize=True)),
            enforce_decoder_input_project=False,
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            transformer_decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=9,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        attn_drop=0.0,
                        proj_drop=0.0,
                        dropout_layer=None,
                        batch_first=False),
                    ffn_cfgs=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        act_cfg=dict(type='ReLU', inplace=True),
                        ffn_drop=0.0,
                        dropout_layer=None,
                        add_identity=True),
                    feedforward_channels=2048,
                    operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                     'ffn', 'norm'))),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=2.0,
                reduction='mean',
                class_weight=[1.0] * 2 + [0.1]),
            loss_mask=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=5.0),
            loss_dice=dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                reduction='mean',
                naive_dice=True,
                eps=1.0,
                loss_weight=5.0)),
        geometry_head=dict(
            type='GeometryHead',
            embed_dims=256,
            num_layers=3,
            hidden_dims=256),
        geometry_loss_weight=1.0,
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='ClassificationMatchCost', weight=2.0),
                    dict(
                        type='CrossEntropyLossMatchCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
                ]),
            sampler=dict(type='MaskPseudoSampler')),
        test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))
    )

    # 2. Build Model
    print("Building DeepRoofMask2Former...")
    model = MODELS.build(model_cfg)
    if torch.cuda.is_available():
        model = model.cuda()
    print("Model built successfully.")

    # 3. Create Synthetic Data (Single Batch)
    batch_size = 2
    H, W = 128, 128 # Small size for speed
    
    # Inputs: (B, C, H, W)
    inputs = torch.randn(batch_size, 3, H, W)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        
    # Data Samples: List of SegDataSample
    data_samples = []
    for _ in range(batch_size):
        data_sample = SegDataSample()
        data_sample.set_metainfo(dict(
            img_shape=(H, W),
            ori_shape=(H, W),
            pad_shape=(H, W)
        ))
        
        # GT Sem Seg (Labels: 0, 1)
        gt_sem_seg = torch.randint(0, 2, (1, H, W)).long()
        data_sample.gt_sem_seg = PixelData(data=gt_sem_seg)
        
        # GT Normals (3, H, W) - Random unit vectors (approx)
        gt_normals = torch.randn(3, H, W)
        gt_normals = gt_normals / gt_normals.norm(dim=0, keepdim=True)
        # Note: Mask2Former doesn't strictly use PixelData for normals in base, but our customized head does.
        # But for alignment, we usually attach it to SegDataSample directly or subclass.
        # Here we attach as property for our custom model to read.
        data_sample.gt_normals = PixelData(data=gt_normals)

        if torch.cuda.is_available():
            data_sample = data_sample.cuda()
            
        data_samples.append(data_sample)

    # 4. Training Loop (Single Step)
    print("\nRunning Forward Pass...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    # Forward -> Loss
    losses = model.loss(inputs, data_samples)
    
    print("\nLoss Components:")
    total_loss = 0
    for name, value in losses.items():
        if 'loss' in name:
             # value might be a list or tensor
             if isinstance(value, list):
                 loss_val = sum(value)
             else:
                 loss_val = value
             print(f"  {name}: {loss_val.item():.4f}")
             total_loss += loss_val
             
    print(f"Total Loss: {total_loss.item():.4f}")
    
    # 5. Backward Pass
    print("\nRunning Backward Pass...")
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print("Backward Pass Successful.")
    
    # 6. Check Gradients
    print("\nVerifying Gradients:")
    
    # Check Geometry Head Gradients
    geo_grad_norm = 0.0
    for p in model.geometry_head.parameters():
        if p.grad is not None:
             geo_grad_norm += p.grad.norm().item()
    print(f"  Geometry Head Gradient Norm: {geo_grad_norm:.6f}")
    
    # Check Backbone Gradients
    backbone_grad_norm = 0.0
    for p in model.backbone.parameters():
        if p.grad is not None:
            backbone_grad_norm += p.grad.norm().item()
    print(f"  Backbone Gradient Norm: {backbone_grad_norm:.6f}")

    if geo_grad_norm > 0:
        print("\nSUCCESS: Gradients are flowing to the Geometry Head!")
    else:
        print("\nWARNING: No gradients in Geometry Head. Check loss connection.")

if __name__ == '__main__':
    run_experiment()
