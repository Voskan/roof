import os.path as osp
import numpy as np
import cv2
from typing import Dict, List, Optional, Callable, Union
import torch
from torch.utils.data import Dataset
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from deeproof.datasets.pipelines.augmentations import GoogleMapsAugmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

@DATASETS.register_module()
class DeepRoofDataset(BaseSegDataset):
    """
    Custom Dataset for DeepRoof-2026.
    Combines OmniCity and Building3D data.
    
    Returns:
        - img: Image tensor (3, H, W)
        - gt_semantic_seg: Semantic labels (Flat=1, Sloped=2, Background=0)
        - gt_instance_seg: Instance masks (0=Background, 1..N=Instances)
        - gt_normals: Dense normal map (3, H, W)
    """
    METAINFO = dict(
        classes=('background', 'flat_roof', 'sloped_roof'),
        palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
    )

    def __init__(self,
                 ann_file: str,
                 img_suffix: str = '.png',
                 seg_map_suffix: str = '.png',
                 normal_suffix: str = '.npy',
                 data_root: Optional[str] = None,
                 test_mode: bool = False,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = False,
                 **kwargs):
        
        self.normal_suffix = normal_suffix
        self.augmentor = GoogleMapsAugmentation()
        
        # BaseSegDataset handles basic file list loading if ann_file follows standard MM format
        # or we can override load_data_list
        super().__init__(
            ann_file=ann_file,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            data_root=data_root,
            test_mode=test_mode,
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """
        Load data_list from ann_file.
        Expected format of ann_file (txt):
        image_path mask_path
        """
        # For simplicity, we assume ann_file is a list of ids or paths
        # If data_root provided, paths are relative
        data_list = []
        if self.ann_file and osp.isfile(self.ann_file):
            with open(self.ann_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                # Custom parsing logic: assume "img_name" only, rest derived
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(self.data_root, 'images', img_name + self.img_suffix),
                    seg_map_path=osp.join(self.data_root, 'masks', img_name + self.seg_map_suffix),
                    normal_path=osp.join(self.data_root, 'normals', img_name + self.normal_suffix),
                    img_id=img_name
                )
                data_list.append(data_info)
        return data_list

    def __getitem__(self, idx: int) -> Dict:
        data_info = self.get_data_info(idx)
        
        # 1. Load Image
        img = cv2.imread(data_info['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Load Instance Mask (uint16)
        instance_mask = cv2.imread(data_info['seg_map_path'], cv2.IMREAD_UNCHANGED)
        
        # 3. Load Normal Map
        # Normals saved as .npy (float, -1 to 1) or .png (uint8, 0 to 255)
        # Based on process scripts, we saved .npy for precise training
        try:
            normals = np.load(data_info['normal_path'])
        except:
            # Fallback if png for Building3D
            normals_vis = cv2.imread(data_info['normal_path'].replace('.npy', '.png'))
            if normals_vis is None:
                 # Create dummy normals if missing
                 normals = np.zeros_like(img, dtype=np.float32)
                 normals[:,:,2] = 1.0 # UP
            else:
                normals = (normals_vis.astype(np.float32) / 255.0) * 2.0 - 1.0
                normals = normals[:, :, ::-1] # BGR to RGB (XYZ)

        # 4. Generate Semantic Labels on-the-fly
        # Flat if slope < 5 degrees. Slope = arccos(nz)
        # nz is normals[:, :, 2]
        # Avoid numerical errors
        nz = np.clip(normals[:, :, 2], -1.0, 1.0)
        slope_rad = np.arccos(nz)
        slope_deg = np.degrees(slope_rad)
        
        # Semantic Label: 0=Background, 1=Flat, 2=Sloped
        semantic_mask = np.zeros_like(instance_mask, dtype=np.uint8)
        
        # Filter background (instance_mask == 0) -> 0
        is_roof = instance_mask > 0
        
        is_flat = (slope_deg < 5.0) & is_roof
        is_sloped = (slope_deg >= 5.0) & is_roof
        
        semantic_mask[is_flat] = 1
        semantic_mask[is_sloped] = 2
        
        # 5. Apply Augmentations
        if not self.test_mode:
            # We pass image, mask (semantic), and normals to the augmentor.
            # GoogleMapsAugmentation is a GeometricAugmentation wrapper that
            # handles the replay logic for vector rotations of 'normals'.
            # We also need to handle 'instance_mask'.
            
            # Temporary hack: Treat instance_mask as another 'mask' by passing it 
            # as an additional target if not already configured.
            # But the most robust way is to use the augmentor as intended.
            
            augmented = self.augmentor(
                image=img, 
                mask=semantic_mask,
                normals=normals
            )
            
            # NOTE: GoogleMapsAugmentation currently only has 'mask' and 'normals' 
            # as additional targets. We need to ensure instance_mask is also transformed.
            # Applying the same spatial transform to instance_mask:
            instance_mask_aug = A.ReplayCompose.replay(augmented['replay'], image=instance_mask)['image']
            
            img_tensor = augmented['image']          # already ToTensorV2'd
            sem_tensor = augmented['mask'].long()
            normal_tensor = augmented['normals'].permute(2, 0, 1).float()
            instance_tensor = torch.from_numpy(instance_mask_aug).long()
            
        else:
            # Test mode: just normalization and tensor conversion
            # Using a basic pipeline for consistency
            eval_transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            transformed = eval_transform(image=img)
            img_tensor = transformed['image']
            sem_tensor = torch.from_numpy(semantic_mask).long()
            instance_tensor = torch.from_numpy(instance_mask).long()
            normal_tensor = torch.from_numpy(normals).permute(2, 0, 1).float()

        return dict(
            img=img_tensor,
            gt_semantic_seg=sem_tensor,
            gt_instance_seg=instance_tensor,
            gt_normals=normal_tensor,
            img_metas=data_info
        )
