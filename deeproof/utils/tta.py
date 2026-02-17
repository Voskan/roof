
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from mmengine.structures import InstanceData
from mmseg.structures import SegDataSample
from torchvision.ops import nms


def _instances_from_semantic(sem_map: torch.Tensor, min_area: int = 64) -> InstanceData:
    """Fallback conversion from semantic map to instance-style predictions."""
    if sem_map.ndim == 3:
        sem_map = sem_map.squeeze(0)
    sem_map = sem_map.long()
    device = sem_map.device
    H, W = int(sem_map.shape[-2]), int(sem_map.shape[-1])

    masks, labels, scores = [], [], []
    for cls_id in torch.unique(sem_map).tolist():
        cls_id = int(cls_id)
        if cls_id <= 0 or cls_id == 255:
            continue
        cls_mask = (sem_map == cls_id)
        if int(cls_mask.sum().item()) < min_area:
            continue
        comp_map = cls_mask.detach().cpu().numpy().astype(np.uint8)
        num_comp, comp_labels = cv2.connectedComponents(comp_map, connectivity=8)
        for comp_idx in range(1, int(num_comp)):
            comp = (comp_labels == comp_idx)
            if int(comp.sum()) < min_area:
                continue
            masks.append(torch.from_numpy(comp).to(device=device))
            labels.append(cls_id)
            scores.append(1.0)

    out = InstanceData()
    if masks:
        out.masks = torch.stack([m.bool() for m in masks], dim=0)
        out.labels = torch.tensor(labels, dtype=torch.long, device=device)
        out.scores = torch.tensor(scores, dtype=torch.float32, device=device)
    else:
        out.masks = torch.zeros((0, H, W), dtype=torch.bool, device=device)
        out.labels = torch.zeros((0,), dtype=torch.long, device=device)
        out.scores = torch.zeros((0,), dtype=torch.float32, device=device)
    return out


def apply_tta(model, 
              image: np.ndarray, 
              device='cuda') -> dict:
    """
    Apply Test Time Augmentation (TTA) on a single tile.
    
    Augmentations:
    1. Identity (0 deg)
    2. Rotate 90 deg (CCW)
    3. Rotate 180 deg
    4. Rotate 270 deg
    5. Horizontal Flip
    
    Args:
        model: Loaded DeepRoofMask2Former model.
        image (np.ndarray): Input tile (H, W, 3).
        device (str): Device to run inference on.
        
    Returns:
        dict: Merged instance predictions.
    """
    model.eval()
    H, W, C = image.shape
    
    # 1. Prepare Augmentations
    # Store tuples of (image_tensor, transform_type)
    # transform_type: 'id', 'rot90', 'rot180', 'rot270', 'hflip'
    
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() # (3, H, W)
    
    # Create batch of 5 images
    batch_imgs = []
    transforms = ['id', 'rot90', 'rot180', 'rot270', 'hflip']
    
    # Identity
    batch_imgs.append(img_tensor)
    
    # Rot 90 (k=1)
    batch_imgs.append(torch.rot90(img_tensor, k=1, dims=[1, 2]))
    
    # Rot 180 (k=2)
    batch_imgs.append(torch.rot90(img_tensor, k=2, dims=[1, 2]))
    
    # Rot 270 (k=3)
    batch_imgs.append(torch.rot90(img_tensor, k=3, dims=[1, 2]))
    
    # HFlip
    batch_imgs.append(torch.flip(img_tensor, dims=[2])) # dim 2 is Width
    
    # Stack
    batch_stack = torch.stack(batch_imgs)
    
    # 2. Run Inference
    # Create dummy samples
    batch_samples = [SegDataSample(metainfo=dict(img_shape=(H, W), ori_shape=(H, W))) for _ in range(5)]
    
    print(f"Running TTA Inference on batch of {len(batch_stack)}...")
    with torch.no_grad():
        # Use standard test_step so model data_preprocessor runs exactly as in
        # normal mmseg inference (normalization, padding, device transfer).
        batch_data = dict(
            inputs=[img for img in batch_stack],
            data_samples=batch_samples,
        )
        results = model.test_step(batch_data)
        
    all_pred_instances = []
    
    # 3. Inverse Transformations & Collect
    for i, res in enumerate(results):
        transform = transforms[i]
        preds = getattr(res, 'pred_instances', None)
        if preds is None or len(preds) == 0:
            sem = getattr(res, 'pred_sem_seg', None)
            sem_data = getattr(sem, 'data', None) if sem is not None else None
            if torch.is_tensor(sem_data):
                preds = _instances_from_semantic(sem_data)
            else:
                continue
        
        if len(preds) == 0:
            continue
            
        masks = preds.masks # (N, H, W)
        scores = preds.scores
        labels = preds.labels
        has_normals = hasattr(preds, 'normals') # (N, 3)
        
        # Inverse Transform Masks
        if transform == 'id':
            inv_masks = masks
        elif transform == 'rot90':
            # Inverse of Rot90 is Rot270 (k=3)
            inv_masks = torch.rot90(masks, k=3, dims=[1, 2])
        elif transform == 'rot180':
            # Inverse of Rot180 is Rot180 (k=2)
            inv_masks = torch.rot90(masks, k=2, dims=[1, 2])
        elif transform == 'rot270':
            # Inverse of Rot270 is Rot90 (k=1)
            inv_masks = torch.rot90(masks, k=1, dims=[1, 2])
        elif transform == 'hflip':
            inv_masks = torch.flip(masks, dims=[2])
            
        for k in range(len(scores)):
            mask = inv_masks[k].cpu().numpy()
            score = scores[k].cpu().item()
            label = labels[k].cpu().item()
            
            # Recalculate BBox from inverted mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols): continue
            
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            bbox = [xmin, ymin, xmax + 1, ymax + 1]
            
            inst = {
                'bbox': bbox,
                'mask': mask, # Full mask (or RLE)
                'score': score,
                'label': label
            }
            
            # Inverse Transform Normals
            # Normal vector n = (nx, ny, nz)
            # Coordinates: x (width), y (height), z (up)
            if has_normals:
                n = preds.normals[k].cpu().clone() # (3,)
                nx, ny, nz = n[0], n[1], n[2]
                
                # Careful with coordinate systems. 
                # Torch rot90 on (H, W):
                # k=1 (90 deg CCW): (x, y) -> (y, -x) ? No.
                # Image coords: (0,0) top-left. y down, x right.
                # Rot90 CCW: x' = y, y' = W-1-x.
                # Vector rotation is simpler, centered at origin.
                # Vector (nx, ny) rotation +90 deg:
                # x' = -y
                # y' = x
                # But we need INVERSE transform of the vector.
                # If we rotated image +90, the vector detected is in rotated frame.
                # We need to rotate it back -90 (or +270).
                # Inverse Matrix for +90 (CCW) is -90 (CW).
                # Rot -90: x' = y, y' = -x.
                
                if transform == 'id':
                    pass
                elif transform == 'rot90':
                    # Image was Rot90. Vector is in Rot90 frame.
                    # Undo Rot90 -> Rot -90 (270).
                    # (nx, ny) -> (ny, -nx)
                    inst_n = torch.stack([ny, -nx, nz])
                elif transform == 'rot180':
                    # Undo Rot180 -> Rot 180.
                    # (nx, ny) -> (-nx, -ny)
                    inst_n = torch.stack([-nx, -ny, nz])
                elif transform == 'rot270':
                    # Image was Rot270 (-90). Vector is in that frame.
                    # Undo Rot270 -> Rot +90.
                    # (nx, ny) -> (-ny, nx)
                    inst_n = torch.stack([-ny, nx, nz])
                elif transform == 'hflip':
                    # Horizontal flip (flip x).
                    # Undo HFlip is same HFlip.
                    # (nx, ny) -> (-nx, ny)
                    inst_n = torch.stack([-nx, ny, nz])
                    
                inst['normal'] = inst_n.numpy()
            
            # Filter low confidence before NMS
            if score > 0.1: # Lower threshold to allow aggregation
                all_pred_instances.append(inst)

    # 4. Aggregation (NMS)
    if len(all_pred_instances) == 0:
        return {'instances': []}
        
    print(f"Aggregating {len(all_pred_instances)} instances from TTA...")
    
    boxes = torch.tensor([inst['bbox'] for inst in all_pred_instances], dtype=torch.float32)
    scores = torch.tensor([inst['score'] for inst in all_pred_instances], dtype=torch.float32)
    
    # NMS
    # TTA usually generates highly overlapping boxes for the same object.
    # Standard NMS picks the highest score.
    # 'Soft' aggregation or averaging scores of matched boxes is hard with just NMS.
    # WBF (Weighted Box Fusion) is better but NMS is standard in simple TTA pipelines.
    # We will just pick the best one.
    
    keep_indices = nms(boxes, scores, iou_threshold=0.6)
    
    final_instances = [all_pred_instances[i] for i in keep_indices]
    
    return {
        'instances': final_instances,
        'count': len(final_instances)
    }
