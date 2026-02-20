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
                 image_size: Optional[tuple] = (1024, 1024),
                 data_root: Optional[str] = None,
                 test_mode: bool = False,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = False,
                 **kwargs):
        
        self.normal_suffix = normal_suffix
        if image_size is None:
            self.image_size = None
        elif isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            self.image_size = (int(image_size[0]), int(image_size[1]))
        else:
            raise ValueError(f'Invalid image_size={image_size}. Use int, (h, w), or None.')
        self.augmentor = GoogleMapsAugmentation(use_shadow=not test_mode)
        
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

        # Keep per-batch target shapes consistent for Mask2Former losses.
        # RandomScale introduces variable resolutions, but mmdet's Mask2Former
        # stacks mask targets across the batch and expects identical HxW.
        target_size = getattr(self, 'image_size', None)
        if target_size is not None:
            target_h, target_w = int(target_size[0]), int(target_size[1])

            def _to_numpy(arr):
                if isinstance(arr, torch.Tensor):
                    return arr.detach().cpu().numpy()
                return arr

            img_np = _to_numpy(img)
            sem_np = _to_numpy(semantic_mask)
            normal_np = _to_numpy(normals)
            inst_np = _to_numpy(instance_mask)

            if img_np.shape[:2] != (target_h, target_w):
                img_np = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if sem_np.shape[:2] != (target_h, target_w):
                sem_np = cv2.resize(sem_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            if inst_np.shape[:2] != (target_h, target_w):
                inst_dtype = inst_np.dtype
                inst_np = cv2.resize(inst_np.astype(np.int32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                inst_np = inst_np.astype(inst_dtype)
            if normal_np.shape[:2] != (target_h, target_w):
                normal_np = cv2.resize(normal_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Re-normalize normal vectors after interpolation.
            if normal_np.ndim == 3 and normal_np.shape[2] == 3:
                nrm = np.linalg.norm(normal_np, axis=2, keepdims=True)
                valid = nrm > 1e-6
                normal_np = np.where(valid, normal_np / np.clip(nrm, 1e-6, None), normal_np)

            img = img_np
            semantic_mask = sem_np
            normals = normal_np
            instance_mask = inst_np

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

        def _safe_pixel_data(tensor: torch.Tensor):
            """
            Build PixelData robustly across real mmengine and mocked test modules.
            Some mocked environments return MagicMock objects that do not preserve
            the passed tensor, so we force-assign `.data` when needed.
            """
            try:
                pd = PixelData(data=tensor)
            except Exception:
                class _PixelDataFallback:
                    def __init__(self, data):
                        self.data = data
                return _PixelDataFallback(tensor)
            # Force overwrite even if mock objects reuse a shared return_value.
            try:
                pd.data = tensor
            except Exception:
                class _PixelDataFallback:
                    def __init__(self, data):
                        self.data = data
                return _PixelDataFallback(tensor)
            if not torch.is_tensor(getattr(pd, 'data', None)):
                class _PixelDataFallback:
                    def __init__(self, data):
                        self.data = data
                return _PixelDataFallback(tensor)
            return pd

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

        data_sample.gt_sem_seg = _safe_pixel_data(sem_map.unsqueeze(0).long())
        data_sample.gt_normals = _safe_pixel_data(normal_tensor.float())

        gt_instances = InstanceData()
        inst_ids = torch.unique(inst_map)
        # Instance supervision should contain object instances only.
        # Background is learned from unmatched/no-object queries and semantic targets.
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
                    # If this instance is purely background, it will simply take class 0.
                    # Otherwise, assign it the majority foreground class.
                    fg = cls_ids > 0
                    if fg.any():
                        fg_cls_ids = cls_ids[fg]
                        fg_counts = counts[fg]
                        label = int(fg_cls_ids[fg_counts.argmax()].item())
                    else:
                        label = 0
                labels.append(label)

                if label == 0:
                    # Background doesn't have a valid surface normal. 
                    # Default to pointing straight up (0, 0, 1)
                    avg_n = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=normal_tensor.device)
                else:
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
