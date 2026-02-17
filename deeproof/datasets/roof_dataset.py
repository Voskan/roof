import os.path as osp
import numpy as np
import cv2
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from deeproof.datasets.pipelines.augmentations import GoogleMapsAugmentation


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
        # Keep BGR here and let SegDataPreProcessor handle bgr_to_rgb conversion.
        # This keeps train/inference color pipeline consistent with mmseg defaults.
        
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
            # Instance mask is also passed as an additional target.
            
            augmented = self.augmentor(
                image=img, 
                mask=semantic_mask,
                normals=normals,
                instance_mask=instance_mask
            )
            instance_mask_aug = augmented['instance_mask']
            
            img_aug = augmented['image']
            sem_aug = augmented['mask']
            normals_aug = augmented['normals']

            if isinstance(img_aug, torch.Tensor):
                img_tensor = img_aug.float()
            else:
                img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float()

            if isinstance(sem_aug, torch.Tensor):
                sem_tensor = sem_aug.long()
            else:
                sem_tensor = torch.from_numpy(sem_aug).long()

            if isinstance(normals_aug, torch.Tensor):
                # Accept both HWC and CHW tensor layouts.
                if normals_aug.ndim == 3 and normals_aug.shape[0] == 3:
                    normal_tensor = normals_aug.float()
                else:
                    normal_tensor = normals_aug.permute(2, 0, 1).float()
            else:
                normal_tensor = torch.from_numpy(normals_aug).permute(2, 0, 1).float()

            if isinstance(instance_mask_aug, torch.Tensor):
                instance_tensor = instance_mask_aug.long()
            else:
                instance_tensor = torch.from_numpy(instance_mask_aug).long()
            
        else:
            # Test mode: keep raw scale and let SegDataPreProcessor apply
            # normalization once for consistent train/infer behavior.
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            sem_tensor = torch.from_numpy(semantic_mask).long()
            instance_tensor = torch.from_numpy(instance_mask).long()
            normal_tensor = torch.from_numpy(normals).permute(2, 0, 1).float()

        out = dict(
            img=img_tensor,
            gt_semantic_seg=sem_tensor,
            gt_instance_seg=instance_tensor,
            gt_normals=normal_tensor,
            img_metas=data_info
        )

        # Build MMEngine-style sample package expected by SegDataPreProcessor
        # Keep legacy keys above for backward compatibility with existing tests/utilities.
        try:
            from mmseg.structures import SegDataSample
            from mmengine.structures import PixelData, InstanceData

            H, W = int(img_tensor.shape[-2]), int(img_tensor.shape[-1])
            sem_map = sem_tensor.squeeze(0) if sem_tensor.ndim == 3 else sem_tensor
            inst_map = instance_tensor.squeeze(0) if instance_tensor.ndim == 3 else instance_tensor

            data_sample = SegDataSample()
            data_sample.set_metainfo(
                dict(
                    img_shape=(H, W),
                    ori_shape=(H, W),
                    pad_shape=(H, W),
                    img_id=data_info.get('img_id', '')
                ))

            data_sample.gt_sem_seg = PixelData(data=sem_map.unsqueeze(0).long())
            data_sample.gt_normals = PixelData(data=normal_tensor.float())

            gt_instances = InstanceData()
            inst_ids = torch.unique(inst_map)
            inst_ids = inst_ids[inst_ids > 0]

            if inst_ids.numel() > 0:
                masks = torch.stack([(inst_map == i) for i in inst_ids], dim=0).bool()
                labels = []
                normals_per_inst = []
                for m in masks:
                    sem_vals = sem_map[m]
                    if sem_vals.numel() == 0:
                        label = 1
                    else:
                        cls_ids, counts = torch.unique(sem_vals, return_counts=True)
                        fg = cls_ids > 0
                        if fg.any():
                            cls_ids = cls_ids[fg]
                            counts = counts[fg]
                        label = int(cls_ids[counts.argmax()].item()) if cls_ids.numel() > 0 else 1
                    labels.append(label)

                    avg_n = normal_tensor[:, m].mean(dim=1)
                    avg_n = F.normalize(avg_n, p=2, dim=0)
                    normals_per_inst.append(avg_n)

                labels = torch.tensor(labels, dtype=torch.long)
                normals_inst = torch.stack(normals_per_inst, dim=0).float()
            else:
                masks = torch.zeros((0, H, W), dtype=torch.bool)
                labels = torch.zeros((0,), dtype=torch.long)
                normals_inst = torch.zeros((0, 3), dtype=torch.float32)

            # mmdet Mask2Former head expects tensor-like gt masks and directly
            # applies tensor ops (e.g. unsqueeze). Using BitmapMasks here breaks
            # that path on recent mmdet versions.
            gt_instances.masks = masks

            gt_instances.labels = labels
            gt_instances.normals = normals_inst
            data_sample.gt_instances = gt_instances

            out['inputs'] = img_tensor
            out['data_samples'] = data_sample
        except Exception:
            # Fallback for lightweight test environments with mocked mmseg/mmengine.
            pass

        return out
