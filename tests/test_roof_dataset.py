import torch
import numpy as np
import os
import cv2
import sys
import types
import tempfile

# Mock minimal OpenMMLab surface needed by the dataset module.
mmseg_mod = types.ModuleType('mmseg')
mmseg_registry_mod = types.ModuleType('mmseg.registry')
mmseg_datasets_mod = types.ModuleType('mmseg.datasets')
mmseg_structures_mod = types.ModuleType('mmseg.structures')
mmengine_mod = types.ModuleType('mmengine')
mmengine_structures_mod = types.ModuleType('mmengine.structures')


class DummyRegistry:
    def register_module(self, **kwargs):
        def decorator(cls):
            return cls
        return decorator


class DummyBaseSegDataset:
    def get_data_info(self, idx):
        return self.data_list[idx]


class DummySegDataSample:
    def __init__(self):
        self.metainfo = {}

    def set_metainfo(self, info):
        self.metainfo = info


class DummyPixelData:
    def __init__(self, data=None):
        self.data = data


class DummyInstanceData:
    pass


mmseg_registry_mod.DATASETS = DummyRegistry()
mmseg_datasets_mod.BaseSegDataset = DummyBaseSegDataset
mmseg_structures_mod.SegDataSample = DummySegDataSample
mmengine_structures_mod.PixelData = DummyPixelData
mmengine_structures_mod.InstanceData = DummyInstanceData

sys.modules['mmseg'] = mmseg_mod
sys.modules['mmseg.registry'] = mmseg_registry_mod
sys.modules['mmseg.datasets'] = mmseg_datasets_mod
sys.modules['mmseg.structures'] = mmseg_structures_mod
sys.modules['mmengine'] = mmengine_mod
sys.modules['mmengine.structures'] = mmengine_structures_mod

# Import the dataset once environment is mocked or if real deps available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need a real-ish test for the augmentation logic
from deeproof.datasets.roof_dataset import DeepRoofDataset


def test_transform_vectors_supports_torch_tensor():
    from deeproof.datasets.pipelines.augmentations import GeometricAugmentation

    aug = GeometricAugmentation([])

    # Horizontal flip should negate nx.
    normals = torch.zeros((8, 8, 3), dtype=torch.float32)
    normals[..., 0] = 1.0
    replay_h = dict(
        applied=True,
        __class_fullname__='albumentations.augmentations.geometric.transforms.HorizontalFlip',
        params={}
    )
    aug._transform_vectors(replay_h, normals)
    assert torch.allclose(normals[..., 0], torch.full((8, 8), -1.0))

    # 90 deg CCW should map north (0, 1, 0) to west (-1, 0, 0).
    normals = torch.zeros((8, 8, 3), dtype=torch.float32)
    normals[..., 1] = 1.0
    replay_r90 = dict(
        applied=True,
        __class_fullname__='albumentations.augmentations.geometric.transforms.RandomRotate90',
        params=dict(factor=1)
    )
    aug._transform_vectors(replay_r90, normals)
    assert torch.allclose(normals[..., 0], torch.full((8, 8), -1.0), atol=1e-5)
    assert torch.allclose(normals[..., 1], torch.zeros((8, 8)), atol=1e-5)

def test_augmentation_consistency():
    print("Testing DeepRoofDataset augmentation consistency...")
    
    # Create dummy data files
    data_root = tempfile.mkdtemp(prefix='deeproof_test_data_')
    os.makedirs(f'{data_root}/images', exist_ok=True)
    os.makedirs(f'{data_root}/masks', exist_ok=True)
    os.makedirs(f'{data_root}/normals', exist_ok=True)
    
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.imwrite(f'{data_root}/images/test.png', img)
    
    # Instance mask with a specific building
    instance_mask = np.zeros((256, 256), dtype=np.uint16)
    instance_mask[50:150, 50:150] = 1 
    cv2.imwrite(f'{data_root}/masks/test.png', instance_mask)
    
    # Normals: East facing (1, 0, 0)
    normals = np.zeros((256, 256, 3), dtype=np.float32)
    normals[50:150, 50:150, 0] = 1.0  # East
    np.save(f'{data_root}/normals/test.npy', normals)
    
    with open(f'{data_root}/train.txt', 'w') as f:
        f.write('test\n')
    
    # Initialize dataset without calling BaseSegDataset.__init__
    dataset = object.__new__(DeepRoofDataset)
    dataset.ann_file = f'{data_root}/train.txt'
    dataset.data_root = data_root
    dataset.test_mode = False
    dataset.img_suffix = '.png'
    dataset.seg_map_suffix = '.png'
    dataset.normal_suffix = '.npy'
    
    # Manually set data_info for test
    dataset.data_list = [{
        'img_path': f'{data_root}/images/test.png',
        'seg_map_path': f'{data_root}/masks/test.png',
        'normal_path': f'{data_root}/normals/test.npy',
        'img_id': 'test'
    }]

    # 1. Test Horizontal Flip
    from deeproof.datasets.pipelines.augmentations import GeometricAugmentation
    import albumentations as A
    
    # Configure augmentor for deterministic horizontal flip
    dataset.augmentor = GeometricAugmentation([
        A.HorizontalFlip(p=1.0),
    ])

    item = dataset[0]
    
    # New format: dict(inputs=Tensor, data_samples=SegDataSample)
    assert 'inputs' in item, "Missing 'inputs' key in dataset output"
    assert 'data_samples' in item, "Missing 'data_samples' key in dataset output"
    
    ds = item['data_samples']
    
    # Check instance masks via data_samples
    assert hasattr(ds, 'gt_instances'), "data_samples missing gt_instances"
    gt_inst = ds.gt_instances
    assert hasattr(gt_inst, 'masks'), "gt_instances missing masks"
    masks = gt_inst.masks
    assert torch.is_tensor(masks), "gt_instances.masks must be tensor"
    
    if masks.numel() > 0:
        # Instance mask after horizontal flip
        # Original building was at cols 50:150 (width=256).
        # After horizontal flip: 256-150=106, 256-50=206 -> cols 106:206
        mask_np = masks[0].numpy() if masks.ndim == 3 else masks.numpy()
        assert np.any(mask_np[:, 106:206]), "Instance mask horizontal flip failed"
        assert not np.any(mask_np[:, 50:100]), "Instance mask old position not cleared after flip"
    
    # Check normals via data_samples
    assert hasattr(ds, 'gt_normals'), "data_samples missing gt_normals"
    normals_data = ds.gt_normals.data  # CHW format [3, H, W]
    normal_hwc = normals_data.permute(1, 2, 0).numpy()
    
    # Original: East (1, 0, 0) -> Horiz flip -> West (-1, 0, 0)
    # Check in the flipped region (was 50:150 cols -> 106:206 cols after flip)
    val_nx = normal_hwc[100, 150, 0]
    print(f"Augmented nx at (100, 150): {val_nx}")
    assert np.isclose(val_nx, -1.0, atol=1e-5), f"Normal vector horizontal flip failed: got {val_nx}"

    # Check inputs tensor shape (CHW)
    inp = item['inputs']
    assert inp.ndim == 3, f"inputs should be 3D (CHW), got {inp.ndim}D"
    assert inp.shape[0] == 3, f"inputs channel dim should be 3, got {inp.shape[0]}"

    print("SUCCESS: Augmentation consistency verified.")
    
    # Cleanup
    import shutil
    shutil.rmtree(data_root, ignore_errors=True)


def test_load_data_list_hard_example_oversampling():
    data_root = tempfile.mkdtemp(prefix='deeproof_test_hard_')
    os.makedirs(f'{data_root}/images', exist_ok=True)
    os.makedirs(f'{data_root}/masks', exist_ok=True)
    os.makedirs(f'{data_root}/normals', exist_ok=True)

    with open(f'{data_root}/train.txt', 'w') as f:
        f.write('a\n')
        f.write('b\n')

    with open(f'{data_root}/hard_examples.txt', 'w') as f:
        f.write('b\n')

    dataset = object.__new__(DeepRoofDataset)
    dataset.ann_file = f'{data_root}/train.txt'
    dataset.data_root = data_root
    dataset.img_suffix = '.jpg'
    dataset.seg_map_suffix = '.png'
    dataset.normal_suffix = '.npy'
    dataset.hard_examples_file = f'{data_root}/hard_examples.txt'
    dataset.hard_example_repeat = 3

    data_list = DeepRoofDataset.load_data_list(dataset)
    ids = [d['img_id'] for d in data_list]
    assert ids.count('a') == 1
    assert ids.count('b') == 3

    import shutil
    shutil.rmtree(data_root, ignore_errors=True)

if __name__ == "__main__":
    test_augmentation_consistency()
