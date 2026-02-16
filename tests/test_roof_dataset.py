import torch
import numpy as np
import os
import cv2
import sys
from unittest.mock import MagicMock, patch

# Mock OpenMMLab and Albumentations if needed
sys.modules['mmseg'] = MagicMock()
sys.modules['mmseg.registry'] = MagicMock()
sys.modules['mmseg.datasets'] = MagicMock()
sys.modules['mmengine'] = MagicMock()
sys.modules['mmengine.structures'] = MagicMock()

# Import the dataset once environment is mocked or if real deps available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need a real-ish test for the augmentation logic
from deeproof.datasets.roof_dataset import DeepRoofDataset

def test_augmentation_consistency():
    print("Testing DeepRoofDataset augmentation consistency...")
    
    # Create dummy data files
    data_root = 'tmp_test_data'
    os.makedirs(f'{data_root}/images', exist_ok=True)
    os.makedirs(f'{data_root}/masks', exist_ok=True)
    os.makedirs(f'{data_root}/normals', exist_ok=True)
    
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.imwrite(f'{data_root}/images/test.png', img)
    
    # Instance mask with a specific building
    instance_mask = np.zeros((256, 256), dtype=np.uint16)
    instance_mask[50:150, 50:150] = 1 
    cv2.imwrite(f'{data_root}/masks/test.png', instance_mask)
    
    # Normals: North facing (0, 1, 0)
    normals = np.zeros((256, 256, 3), dtype=np.float32)
    normals[50:150, 50:150, 1] = 1.0 # North
    np.save(f'{data_root}/normals/test.npy', normals)
    
    with open(f'{data_root}/train.txt', 'w') as f:
        f.write('test\n')
    
    # Initialize dataset
    # We mock the BaseSegDataset properties to avoid loading metadata etc.
    with patch('mmseg.datasets.BaseSegDataset.__init__', return_value=None):
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
    # We force a horizontal flip to check vector rotation
    from deeproof.datasets.pipelines.augmentations import GoogleMapsAugmentation
    import albumentations as A
    
    # Configure augmentor for deterministic horizontal flip
    dataset.augmentor = GoogleMapsAugmentation(prob=1.0)
    dataset.augmentor.transforms = A.ReplayCompose([
        A.HorizontalFlip(p=1.0)
    ], additional_targets={'normals': 'mask'})

    item = dataset[0]
    
    # Check instance mask flip
    # Original building was at 50:150 (width). Image size 256.
    # After horizontal flip, width indices should be flipped: (256-150):(256-50) -> 106:206
    mask_aug = item['gt_instance_seg'].numpy()
    assert np.any(mask_aug[:, 106:206] == 1), "Instance mask horizontal flip failed"
    assert not np.any(mask_aug[:, 50:100] == 1), "Instance mask old position not cleared after flip"
    
    # Check normal vector rotation
    # Original: (0, 1, 0) [North]. Horiz flip (mirror across Y-axis) -> nx' = -nx
    # Since nx was 0, it stays (0, 1, 0).
    # Let's try Diagonal / East: (1, 0, 0) -> (-1, 0, 0) [West]
    normals[50:150, 50:150, 0] = 1.0
    normals[50:150, 50:150, 1] = 0.0
    np.save(f'{data_root}/normals/test.npy', normals)
    
    item = dataset[0]
    normal_aug = item['gt_normals'].permute(1, 2, 0).numpy()
    
    # In the flipped region, nx should be -1.0
    val_nx = normal_aug[100, 150, 0]
    print(f"Augmented nx at (100, 150): {val_nx}")
    assert np.isclose(val_nx, -1.0, atol=1e-5), f"Normal vector horizontal flip failed: got {val_nx}"

    print("SUCCESS: Augmentation consistency verified.")
    
    # Cleanup
    import shutil
    shutil.rmtree(data_root)

if __name__ == "__main__":
    test_augmentation_consistency()
