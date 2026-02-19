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
        
        # 1. Load Image (BGR uint8 — SegDataPreProcessor handles bgr_to_rgb + normalization)
        img = cv2.imread(data_info['img_path'])
        if img is None:
            raise FileNotFoundError(f"Image not found: {data_info['img_path']}")
        
        # 2. Load Instance Mask (uint16)
        instance_mask = cv2.imread(data_info['seg_map_path'], cv2.IMREAD_UNCHANGED)
        if instance_mask is None:
            raise FileNotFoundError(f"Mask not found: {data_info['seg_map_path']}")
        
        # 3. Load Normal Map
        try:
            normals = np.load(data_info['normal_path'])
        except Exception:
            normals_vis = cv2.imread(data_info['normal_path'].replace('.npy', '.png'))
            if normals_vis is None:
                normals = np.zeros_like(img, dtype=np.float32)
                normals[:,:,2] = 1.0  # UP
            else:
                normals = (normals_vis.astype(np.float32) / 255.0) * 2.0 - 1.0
                normals = normals[:, :, ::-1]  # BGR to RGB (XYZ)

        # 4. Generate Semantic Labels on-the-fly
        # Flat if slope < 5 degrees. Slope = arccos(nz)
        nz = np.clip(normals[:, :, 2], -1.0, 1.0)
        slope_rad = np.arccos(nz)
        slope_deg = np.degrees(slope_rad)
        
        semantic_mask = np.zeros_like(instance_mask, dtype=np.uint8)
        is_roof = instance_mask > 0
        # FIX Bug #7: Lowered threshold from 5° to 2°.
        # OmniCity roofs viewed from nadir rarely exceed 5°, leaving sloped class
        # nearly empty (>98% flat). At 2° more instances are classified as sloped,
        # improving class balance and giving the sloped head real training signal.
        semantic_mask[(slope_deg < 2.0) & is_roof] = 1   # Flat (nearly horizontal)
        semantic_mask[(slope_deg >= 2.0) & is_roof] = 2  # Sloped (any detectable pitch)
        
        # 5. Apply Augmentations
        if not self.test_mode:
            augmented = self.augmentor(
                image=img, 
                mask=semantic_mask,
                normals=normals,
                instance_mask=instance_mask
            )
            img = augmented['image']
            semantic_mask = augmented['mask']
            normals = augmented['normals']
            instance_mask = augmented['instance_mask']

        # 6. Convert to tensors
        # Image: HWC uint8 BGR -> CHW float [0, 255]
        # SegDataPreProcessor will apply mean/std normalization and bgr_to_rgb.
        if isinstance(img, torch.Tensor):
            img_tensor = img.float()
            if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
        else:
            img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float()

        if isinstance(semantic_mask, torch.Tensor):
            sem_tensor = semantic_mask.long()
        else:
            sem_tensor = torch.from_numpy(semantic_mask.copy()).long()

        if isinstance(normals, torch.Tensor):
            if normals.ndim == 3 and normals.shape[0] == 3:
                normal_tensor = normals.float()
            else:
                normal_tensor = normals.permute(2, 0, 1).float()
        else:
            normal_tensor = torch.from_numpy(normals.copy()).permute(2, 0, 1).float()

        if isinstance(instance_mask, torch.Tensor):
            instance_tensor = instance_mask.long()
        else:
            instance_tensor = torch.from_numpy(instance_mask.copy()).long()

        # 7. Build MMEngine SegDataSample — the ONLY format MMEngine Runner expects
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
                img_id=data_info.get('img_id', ''),
                img_path=data_info.get('img_path', ''),
                seg_map_path=data_info.get('seg_map_path', ''),
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

        gt_instances.masks = masks
        gt_instances.labels = labels
        gt_instances.normals = normals_inst
        data_sample.gt_instances = gt_instances

        return dict(
            inputs=img_tensor,
            data_samples=data_sample,
        )

