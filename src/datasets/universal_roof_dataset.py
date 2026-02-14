import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from typing import List, Optional, Dict
from deeproof.datasets.pipelines.augmentations import GoogleMapsAugmentation

class UniversalRoofDataset(Dataset):
    """
    Robust PyTorch Dataset for DeepRoof-2026.
    Unifies OmniCity, RoofN3D, and UrbanScene3D into a single training stream.
    """
    def __init__(
        self, 
        data_roots: List[str], 
        transform: Optional[GoogleMapsAugmentation] = None,
        oversample: bool = True,
        image_size: int = 1024
    ):
        self.data_roots = [Path(root) for root in data_roots]
        self.transform = transform
        self.oversample = oversample
        self.image_size = image_size
        
        # 1. Indexing: Recursive scan for valid samples
        self.samples = []
        self.dataset_indices = {} # Mapping root_path -> list of indices
        
        total_indexed = 0
        for root in self.data_roots:
            root_samples = []
            # Convention: {id}_rgb.jpg
            for img_path in tqdm(list(root.rglob("*_rgb.jpg")), desc=f"Indexing {root.name}"):
                base_id = img_path.name.replace("_rgb.jpg", "")
                
                # Verify mask exists
                mask_path = img_path.parent / f"{base_id}_mask.png"
                if not mask_path.exists():
                    # Fallback for some datasets that might use different extensions or naming
                    mask_path = img_path.parent / f"{base_id}_mask.jpg"
                
                if mask_path.exists():
                    # Normal map is optional
                    normal_path = img_path.parent / f"{base_id}_normal.npy"
                    
                    root_samples.append({
                        'id': base_id,
                        'image': img_path,
                        'mask': mask_path,
                        'normal': normal_path if normal_path.exists() else None,
                        'root': root
                    })
            
            self.dataset_indices[str(root)] = list(range(total_indexed, total_indexed + len(root_samples)))
            self.samples.extend(root_samples)
            total_indexed += len(root_samples)
            
        # 2. Balancing Strategy
        if self.oversample and len(self.data_roots) > 1:
            self._apply_balancing()
            
        print(f"UniversalRoofDataset Initialized: {len(self.samples)} samples across {len(self.data_roots)} sources.")

    def _apply_balancing(self):
        """
        Replicate samples from smaller datasets to match the size of the largest dataset.
        """
        counts = {root: len(indices) for root, indices in self.dataset_indices.items()}
        max_size = max(counts.values())
        
        new_samples = []
        for root, indices in self.dataset_indices.items():
            if counts[root] == 0: continue
            
            # Calculate replication factor
            reps = max_size // counts[root]
            remainder = max_size % counts[root]
            
            # Add full replications
            root_orig_samples = [self.samples[i] for i in indices]
            for _ in range(reps):
                new_samples.extend(root_orig_samples)
            
            # Add random remainder
            if remainder > 0:
                rem_indices = np.random.choice(indices, remainder, replace=False)
                new_samples.extend([self.samples[i] for i in rem_indices])
                
        self.samples = new_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_meta = self.samples[idx]
        
        # 1. Load Image (RGB)
        image = cv2.imread(str(sample_meta['image']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Load Mask
        mask = cv2.imread(str(sample_meta['mask']), cv2.IMREAD_GRAYSCALE)
        
        # 3. Load/Handle Normal Map
        valid_normal = 1.0
        if sample_meta['normal']:
            try:
                normal = np.load(str(sample_meta['normal']))
            except Exception:
                normal = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
                valid_normal = 0.0
        else:
            normal = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
            valid_normal = 0.0

        # Apply transformations (Geometric + Photometric)
        if self.transform:
            # Note: transform expects a dict with 'image', 'mask', and 'normals'
            augmented = self.transform(image=image, mask=mask, normals=normal)
            image = augmented['image']
            mask = augmented['mask']
            normal = augmented['normals']
        else:
            # Fallback basic normalization if no transform provided
            image = (image.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).long()
            normal = torch.from_numpy(normal).permute(2, 0, 1)

        return {
            'image': image,
            'mask': mask,
            'normal': normal,
            'valid_normal': torch.tensor(valid_normal, dtype=torch.float32),
            'id': sample_meta['id']
        }

from tqdm import tqdm
