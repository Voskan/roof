_base_ = ['./deeproof_production_swin_L.py']

custom_imports = dict(
    imports=[
        'mmdet.models',
        'deeproof.models.backbones.swin_v2_compat',
        'deeproof.datasets.roof_dataset',
        'deeproof.datasets.universal_roof_dataset',
        'deeproof.models.heads.mask2former_head',
        'deeproof.models.heads.dense_normal_head',
        'deeproof.models.heads.edge_head',
        'deeproof.models.deeproof_model',
        'deeproof.models.heads.geometry_head',
        'deeproof.models.losses',
        'deeproof.evaluation.metrics',
    ],
    allow_failed_imports=False)

dataset_type = 'UniversalRoofDataset'
train_pipeline = []

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        oversample=True,
        balance_seed=42,
        slope_threshold_deg=2.0,
        sr_dual_prob=0.25,
        sr_scale=2.0,
        image_size=(1024, 1024),
        pipeline=train_pipeline,
        sources=[
            dict(
                name='omnicity',
                data_root='data/OmniCity',
                ann_file='train.txt',
                img_suffix='.jpg',
                seg_map_suffix='.png',
                normal_suffix='.npy',
                images_dir='images',
                masks_dir='masks',
                normals_dir='normals',
                sam_masks_dir='sam_masks',
                allow_missing_normals=False,
            ),
            dict(
                name='roofn3d',
                data_root='data/RoofN3D',
                ann_file='train.txt',
                img_suffix='.jpg',
                seg_map_suffix='.png',
                normal_suffix='.npy',
                images_dir='images',
                masks_dir='masks',
                normals_dir='normals',
                sam_masks_dir='sam_masks',
                allow_missing_normals=True,
            ),
            dict(
                name='urbanscene3d',
                data_root='data/UrbanScene3D',
                ann_file='train.txt',
                img_suffix='.jpg',
                seg_map_suffix='.png',
                normal_suffix='.npy',
                images_dir='images',
                masks_dir='masks',
                normals_dir='normals',
                sam_masks_dir='sam_masks',
                allow_missing_normals=True,
            ),
        ],
    ),
)
