
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


def _rotate_normal_vector(nx, ny, nz, degrees_ccw: float):
    """
    Rotate normal vector (nx, ny) in the image plane by degrees_ccw counter-clockwise.
    nz is invariant (nadir view, rotation is in XY plane).

    This applies the standard 2D rotation matrix:
        nx' =  nx * cos(theta) - ny * sin(theta)
        ny' =  nx * sin(theta) + ny * cos(theta)
    """
    theta = np.deg2rad(degrees_ccw)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    nx_new = nx * cos_t - ny * sin_t
    ny_new = nx * sin_t + ny * cos_t
    return nx_new, ny_new, nz


def apply_tta(model,
              image: np.ndarray,
              device='cuda') -> dict:
    """
    Apply Test Time Augmentation (TTA) on a single tile.

    Augmentations applied to image:
    1. Identity (0 deg)
    2. Rotate 90 deg CCW (k=1)
    3. Rotate 180 deg (k=2)
    4. Rotate 270 deg CCW / 90 CW (k=3)
    5. Horizontal Flip (flip x-axis)

    For each augmented prediction, we apply the INVERSE geometric transform to
    bring masks and normal vectors back to the original image coordinate system
    before aggregating with NMS.

    Normal vector inverse transforms (2D rotation, nz invariant):
        rot90  (applied +90):  inverse is -90  => (nx,ny) -> ( ny/1, -nx)  [rot(-90)]
                               Wait: torch.rot90 k=1 rotates 90 CCW, which means
                               a point (x,y) goes to (y, W-1-x) in image coords.
                               For vectors (centered at origin): (vx,vy) -> (-vy, vx).
                               Inverse of that is (vx,vy) -> (vy, -vx).
        rot180 (applied +180): inverse is -180 => (nx,ny) -> (-nx, -ny)
        rot270 (applied -90):  inverse is +90  => (nx,ny) -> (-ny, nx)
        hflip  (flip x):       inverse is hflip => (nx,ny) -> (-nx, ny)
    """
    model.eval()
    H, W, C = image.shape

    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # (3, H, W)

    # 1. Prepare batch
    transforms = ['id', 'rot90', 'rot180', 'rot270', 'hflip']
    batch_imgs = [
        img_tensor,                                      # id
        torch.rot90(img_tensor, k=1, dims=[1, 2]),       # rot90 CCW
        torch.rot90(img_tensor, k=2, dims=[1, 2]),       # rot180
        torch.rot90(img_tensor, k=3, dims=[1, 2]),       # rot270 CCW
        torch.flip(img_tensor, dims=[2]),                # hflip
    ]
    batch_stack = torch.stack(batch_imgs)

    # 2. Run Inference
    batch_samples = [
        SegDataSample(metainfo=dict(img_shape=(H, W), ori_shape=(H, W)))
        for _ in range(len(transforms))
    ]

    print(f"Running TTA on batch of {len(batch_stack)}...")
    with torch.no_grad():
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

        masks = preds.masks   # (N, H, W)
        scores = preds.scores
        labels = preds.labels
        has_normals = hasattr(preds, 'normals')

        # Inverse-transform masks back to original coordinate frame
        if transform == 'id':
            inv_masks = masks
        elif transform == 'rot90':
            # Applied rot90 CCW → inverse is rot90 CW (k=3)
            inv_masks = torch.rot90(masks, k=3, dims=[1, 2])
        elif transform == 'rot180':
            # Inverse of rot180 is rot180
            inv_masks = torch.rot90(masks, k=2, dims=[1, 2])
        elif transform == 'rot270':
            # Applied rot90 CW → inverse is rot90 CCW (k=1)
            inv_masks = torch.rot90(masks, k=1, dims=[1, 2])
        elif transform == 'hflip':
            inv_masks = torch.flip(masks, dims=[2])
        else:
            inv_masks = masks

        for k in range(len(scores)):
            mask = inv_masks[k].cpu().numpy()
            score = scores[k].cpu().item()
            label = labels[k].cpu().item()

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue

            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            bbox = [xmin, ymin, xmax + 1, ymax + 1]

            inst = {
                'bbox': bbox,
                'mask': mask,
                'score': score,
                'label': label,
            }

            # Inverse-transform normals
            if has_normals:
                # FIX: Original code had uninitialized inst_n for 'id' transform
                # causing NameError or wrong variable usage on the line below.
                n = preds.normals[k].cpu().clone()  # (3,)
                nx, ny, nz = float(n[0]), float(n[1]), float(n[2])

                if transform == 'id':
                    # No transform — vector is already in original frame
                    inv_nx, inv_ny, inv_nz = nx, ny, nz
                elif transform == 'rot90':
                    # Image was rotated +90 CCW. In image coords with origin top-left,
                    # rot90 CCW maps vector (vx,vy) -> (-vy, vx).
                    # Inverse: rotate -90: (vx,vy) -> (vy, -vx).
                    inv_nx, inv_ny, inv_nz = _rotate_normal_vector(nx, ny, nz, -90)
                elif transform == 'rot180':
                    # Inverse of rot180 is rot180: (vx,vy) -> (-vx, -vy)
                    inv_nx, inv_ny, inv_nz = _rotate_normal_vector(nx, ny, nz, 180)
                elif transform == 'rot270':
                    # Image was rotated +270 CCW (= -90 CW). Vector: (vx,vy) -> (vy,-vx).
                    # Inverse: rotate +90: (vx,vy) -> (-vy, vx).
                    inv_nx, inv_ny, inv_nz = _rotate_normal_vector(nx, ny, nz, 90)
                elif transform == 'hflip':
                    # Horizontal flip negates x-component of normal
                    # Inverse is same flip: nx -> -nx, ny unchanged
                    inv_nx, inv_ny, inv_nz = -nx, ny, nz
                else:
                    inv_nx, inv_ny, inv_nz = nx, ny, nz

                # Re-normalize after inverse transform (avoid floating point drift)
                mag = (inv_nx**2 + inv_ny**2 + inv_nz**2) ** 0.5
                if mag > 1e-6:
                    inv_nx /= mag
                    inv_ny /= mag
                    inv_nz /= mag

                inst['normal'] = np.array([inv_nx, inv_ny, inv_nz], dtype=np.float32)

            if score > 0.1:
                all_pred_instances.append(inst)

    # 4. Aggregation via NMS
    if len(all_pred_instances) == 0:
        return {'instances': []}

    print(f"Aggregating {len(all_pred_instances)} instances from TTA...")

    boxes = torch.tensor(
        [inst['bbox'] for inst in all_pred_instances], dtype=torch.float32)
    agg_scores = torch.tensor(
        [inst['score'] for inst in all_pred_instances], dtype=torch.float32)

    keep_indices = nms(boxes, agg_scores, iou_threshold=0.6)
    final_instances = [all_pred_instances[i] for i in keep_indices]

    return {
        'instances': final_instances,
        'count': len(final_instances),
    }
