import pytest
import torch


def _build_tiny_model_cfg():
    num_classes = 3
    embed_dims = 64
    return dict(
        type='DeepRoofMask2Former',
        data_preprocessor=dict(
            type='SegDataPreProcessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            size_divisor=32,
            pad_val=0,
            seg_pad_val=255),
        backbone=dict(
            type='SwinTransformerV2',
            embed_dims=32,
            depths=[1, 1, 1, 1],
            num_heads=[2, 4, 8, 16],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            init_cfg=None,
            pretrained_window_sizes=[0, 0, 0, 0]),
        geometry_head=dict(
            type='GeometryHead',
            embed_dims=embed_dims,
            num_layers=2,
            hidden_dims=embed_dims),
        geometry_loss=dict(
            type='CosineSimilarityLoss',
            loss_weight=1.0),
        decode_head=dict(
            type='DeepRoofMask2FormerHead',
            in_channels=[32, 64, 128, 256],
            strides=[4, 8, 16, 32],
            feat_channels=embed_dims,
            out_channels=embed_dims,
            num_classes=num_classes,
            num_queries=20,
            num_transformer_feat_level=3,
            align_corners=False,
            pixel_decoder=dict(
                type='mmdet.MSDeformAttnPixelDecoder',
                num_outs=3,
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU'),
                encoder=dict(
                    num_layers=1,
                    layer_cfg=dict(
                        self_attn_cfg=dict(
                            embed_dims=embed_dims,
                            num_levels=3,
                            batch_first=True),
                        ffn_cfg=dict(
                            embed_dims=embed_dims,
                            feedforward_channels=256,
                            num_fcs=2,
                            ffn_drop=0.0,
                            act_cfg=dict(type='ReLU', inplace=True))),
                    init_cfg=None),
                positional_encoding=dict(num_feats=32, normalize=True),
                init_cfg=None),
            enforce_decoder_input_project=False,
            positional_encoding=dict(num_feats=32, normalize=True),
            transformer_decoder=dict(
                return_intermediate=True,
                num_layers=2,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=embed_dims,
                        num_heads=8,
                        attn_drop=0.0,
                        proj_drop=0.0,
                        dropout_layer=None,
                        batch_first=True),
                    cross_attn_cfg=dict(
                        embed_dims=embed_dims,
                        num_heads=8,
                        attn_drop=0.0,
                        proj_drop=0.0,
                        dropout_layer=None,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=embed_dims,
                        feedforward_channels=256,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            loss_cls=dict(
                type='DeepRoofCrossEntropyLoss',
                use_sigmoid=False,
                reduction='mean',
                loss_weight=2.0,
                class_weight=[1.0, 1.0, 2.0, 0.1]),
            loss_mask=dict(
                type='DeepRoofCrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=2.0),
            loss_dice=dict(
                type='DeepRoofDiceLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=2.0),
            train_cfg=dict(
                num_points=256,
                oversample_ratio=3.0,
                importance_sample_ratio=0.75,
                assigner=dict(
                    type='mmdet.HungarianAssigner',
                    match_costs=[
                        dict(type='mmdet.ClassificationCost', weight=2.0),
                        dict(type='mmdet.CrossEntropyLossCost', weight=2.0, use_sigmoid=True),
                        dict(type='mmdet.DiceCost', weight=2.0, pred_act=True, eps=1.0),
                    ]),
                sampler=dict(type='mmdet.MaskPseudoSampler'))),
        test_cfg=dict(mode='whole'),
    )


@pytest.mark.integration
def test_single_train_step_integration():
    pytest.importorskip('mmcv')
    pytest.importorskip('mmdet')
    pytest.importorskip('mmseg')

    from mmseg.registry import MODELS
    from mmseg.structures import SegDataSample
    from mmseg.utils import register_all_modules
    from mmengine.structures import InstanceData, PixelData

    import deeproof.models.backbones.swin_v2_compat
    import deeproof.models.deeproof_model
    import deeproof.models.heads.geometry_head
    import deeproof.models.heads.mask2former_head
    import deeproof.models.losses

    register_all_modules(init_default_scope=False)

    cfg = _build_tiny_model_cfg()
    model = MODELS.build(cfg)
    model.train()

    h, w = 64, 64
    inputs = torch.rand((1, 3, h, w), dtype=torch.float32)

    sample = SegDataSample()
    sample.set_metainfo(dict(img_shape=(h, w), ori_shape=(h, w), pad_shape=(h, w)))

    sem = torch.zeros((1, h, w), dtype=torch.long)
    sem[:, 16:48, 16:48] = 1
    sample.gt_sem_seg = PixelData(data=sem)

    dense_normals = torch.zeros((3, h, w), dtype=torch.float32)
    dense_normals[2, :, :] = 1.0
    sample.gt_normals = PixelData(data=dense_normals)

    inst = InstanceData()
    mask = torch.zeros((1, h, w), dtype=torch.bool)
    mask[:, 16:48, 16:48] = True
    inst.masks = mask
    inst.labels = torch.tensor([1], dtype=torch.long)
    inst.normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    sample.gt_instances = inst

    losses = model.loss(inputs, [sample])
    loss_terms = [v for k, v in losses.items() if 'loss' in k and torch.is_tensor(v)]
    assert loss_terms, 'No loss terms produced by model.loss()'
    total_loss = torch.stack(loss_terms).sum()
    assert torch.isfinite(total_loss).item(), 'Total loss is not finite'

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    has_backbone_grad = any(p.grad is not None for p in model.backbone.parameters())
    has_decode_grad = any(p.grad is not None for p in model.decode_head.parameters())
    has_geo_grad = any(p.grad is not None for p in model.geometry_head.parameters())

    assert has_backbone_grad, 'No gradients on backbone'
    assert has_decode_grad, 'No gradients on decode_head'
    assert has_geo_grad, 'No gradients on geometry_head'
