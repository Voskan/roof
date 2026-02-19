
import torch
import torch.nn.functional as F
import numpy as np
from mmengine.structures import InstanceData
from mmseg.structures import SegDataSample
from torchvision.ops import nms


def sliding_window_inference(model,
                             image: np.ndarray,
                             window_size: int = 1024,
                             stride: int = 512,
                             batch_size: int = 1,
                             score_threshold: float = 0.4,
                             nms_iou_threshold: float = 0.5,
                             device='cuda') -> dict:
    """
    Perform sliding window inference on a large image.

    Args:
        model: Loaded DeepRoofMask2Former model.
        image (np.ndarray): Large input image (H, W, 3) in RGB format.
        window_size (int): Size of the crop (default 1024).
        stride (int): Stride for sliding window (default 512 = 50% overlap).
        batch_size (int): Inference batch size.
        score_threshold (float): Min instance confidence to keep (default 0.4).
        nms_iou_threshold (float): IoU threshold for NMS (default 0.5).
        device (str): Device to run model on.

    Returns:
        dict: {'instances': list[dict], 'count': int}
    """
    model.eval()
    H, W, C = image.shape

    # 1. Generate Windows (deduplicated with a set)
    # FIX: Original loop had a bug — after edge-adjustment it could produce
    # duplicate (y1, x1) coordinates (e.g., when image is smaller than stride).
    # We use an ordered set to deduplicate while preserving raster order.
    seen_coords = set()
    windows = []
    coords = []

    for y in range(0, max(1, H - window_size + 1), stride):
        for x in range(0, max(1, W - window_size + 1), stride):
            y1 = min(y, max(0, H - window_size))
            x1 = min(x, max(0, W - window_size))
            y2 = y1 + window_size
            x2 = x1 + window_size
            if (y1, x1) in seen_coords:
                continue
            seen_coords.add((y1, x1))
            crop = image[y1:y2, x1:x2, :]
            windows.append(crop)
            coords.append((y1, x1))

    # Always include bottom-right corner for images larger than window_size
    for y1, x1 in [(max(0, H - window_size), max(0, W - window_size))]:
        if (y1, x1) not in seen_coords:
            seen_coords.add((y1, x1))
            crop = image[y1:y1 + window_size, x1:x1 + window_size, :]
            windows.append(crop)
            coords.append((y1, x1))

    if not windows:
        # Image is smaller than window_size: just run inference on the full image
        windows = [image]
        coords = [(0, 0)]

    # 2. Run Inference Per Window in Batches
    all_pred_instances = []

    print(f"Total windows to process: {len(windows)}")

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i: i + batch_size]
            batch_coords = coords[i: i + batch_size]

            # Convert to tensors — model data_preprocessor handles normalization
            # Images are uint8 HWC BGR (from cv2) or float RGB — pass as float CHW
            imgs = []
            for w in batch_windows:
                if w.dtype == np.uint8:
                    t = torch.from_numpy(w.copy()).permute(2, 0, 1).float()
                else:
                    t = torch.from_numpy(w.copy()).permute(2, 0, 1)
                imgs.append(t)
            imgs_tensor = torch.stack(imgs).to(device)

            # Build dummy SegDataSamples with correct shape metadata
            actual_h = batch_windows[0].shape[0]
            actual_w = batch_windows[0].shape[1]
            batch_samples = [
                SegDataSample(metainfo=dict(
                    img_shape=(actual_h, actual_w),
                    ori_shape=(actual_h, actual_w),
                    pad_shape=(actual_h, actual_w),
                ))
                for _ in batch_windows
            ]

            results = model.predict(imgs_tensor, batch_samples)

            for j, res in enumerate(results):
                y_off, x_off = batch_coords[j]

                preds = getattr(res, 'pred_instances', None)
                if preds is None or len(preds) == 0:
                    continue

                masks = preds.masks      # (N, H_win, W_win)
                scores = preds.scores
                labels = preds.labels
                has_normals = hasattr(preds, 'normals')

                for k in range(len(scores)):
                    score = scores[k].cpu().item()
                    # FIX: Raised default threshold from 0.3 → 0.4.
                    # 0.3 was too permissive and allowed too many false positive
                    # low-confidence detections to pass through to NMS.
                    if score < score_threshold:
                        continue

                    mask = masks[k].cpu().numpy()
                    label = labels[k].cpu().item()

                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if not np.any(rows) or not np.any(cols):
                        continue

                    ymin, ymax = np.where(rows)[0][[0, -1]]
                    xmin, xmax = np.where(cols)[0][[0, -1]]

                    # Global bbox in the full image
                    global_bbox = [
                        int(x_off + xmin),
                        int(y_off + ymin),
                        int(x_off + xmax + 1),
                        int(y_off + ymax + 1),
                    ]

                    # Store local crop with global offset for mask IoU merging
                    mask_crop = mask[ymin:ymax + 1, xmin:xmax + 1]

                    inst = {
                        'bbox': global_bbox,
                        'mask_crop': mask_crop,
                        'offset': (int(y_off + ymin), int(x_off + xmin)),
                        'score': score,
                        'label': label,
                    }

                    if has_normals:
                        inst['normal'] = preds.normals[k].cpu().numpy()

                    all_pred_instances.append(inst)

    # 3. NMS / Instance Merging
    if len(all_pred_instances) == 0:
        return {'instances': [], 'count': 0}

    print(f"Applying NMS on {len(all_pred_instances)} candidates...")

    boxes = torch.tensor(
        [inst['bbox'] for inst in all_pred_instances], dtype=torch.float32)
    scores_tensor = torch.tensor(
        [inst['score'] for inst in all_pred_instances], dtype=torch.float32)

    # Use torchvision box NMS (fast, GPU-accelerated if available)
    # For better results with segmentation, mask-IoU NMS (from post_processing.py)
    # can be applied as a second pass.
    keep_indices = nms(boxes, scores_tensor, iou_threshold=nms_iou_threshold)

    final_instances = [all_pred_instances[i] for i in keep_indices.tolist()]

    print(f"After NMS: {len(final_instances)} instances kept")
    return {
        'instances': final_instances,
        'count': len(final_instances),
    }
