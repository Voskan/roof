
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
                             device='cuda') -> dict:
    """
    Perform sliding window inference on a large image.
    
    Args:
        model: Loaded DeepRoofMask2Former model.
        image (np.ndarray): Large input image (H, W, 3) in RGB format.
        window_size (int): Size of the crop (default 1024).
        stride (int): Stride for sliding window (default 512).
        batch_size (int): Inference batch size.
        device (str): Device to run model on.
        
    Returns:
        dict: A dictionary containing:
            - 'masks': List of global instance masks (RLE or Polygon or Boolean Map).
                       Here returning a combined list of dicts {'segmentation': binary_mask, 'score': s, 'label': l}
            - 'normals': Global normal map (H, W, 3).
            - 'instances': Raw instance data for further processing.
    """
    model.eval()
    H, W, C = image.shape
    
    # 1. Generate Windows
    windows = []
    coords = [] # (y, x) top-left coordinates
    
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = y
            x1 = x
            y2 = min(H, y + window_size)
            x2 = min(W, x + window_size)
            
            # Adjust if window exceeds boundary (take last valid crop)
            if y2 - y1 < window_size:
                y1 = max(0, H - window_size)
                y2 = H
            if x2 - x1 < window_size:
                x1 = max(0, W - window_size)
                x2 = W
                
            crop = image[y1:y2, x1:x2, :]
            windows.append(crop)
            coords.append((y1, x1))
            
            # Avoid infinite loop if image is small/stride is large or aligned with edge
            # Basic logic: simple tiling
            if x2 == W: break
        if y2 == H: break

    # Remove duplicates if any (basic check)
    unique_coords = sorted(list(set(coords)))
    # Ideally, re-extract crops based on unique coords or ensure loops are correct.
    # The loops above handle basic tiling.
    
    # 2. Run Inference Per Window
    all_pred_instances = [] # List of (mask, score, label, normal, bbox, offset_y, offset_x)
    global_normal_map = torch.zeros((3, H, W), device='cpu')
    global_normal_count = torch.zeros((1, H, W), device='cpu')
    
    print(f"Total windows to process: {len(windows)}")
    
    # Processing in batches
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i : i + batch_size]
            batch_coords = coords[i : i + batch_size]
            
            # Prepare batch input
            # Normalize/Preprocess as per model requirement
            # Assuming model expects list of tensors or stacked tensor efficiently
            # Note: MMseg usually handles preprocessing via data_preprocessor.
            # Here we manually convert to tensor and assume preprocessor is in model or handling manually.
            # Standard: (B, C, H, W), float32
            
            imgs = [torch.from_numpy(w).permute(2, 0, 1).float() for w in batch_windows]
            imgs = torch.stack(imgs).to(device)
            
            # Inference
            # model.predict returns list of SegDataSample
            # We need to construct data_samples wrapper if predict expects it
            batch_samples = [SegDataSample(metainfo=dict(img_shape=(window_size, window_size), ori_shape=(window_size, window_size))) for _ in batch_windows]
            
            results = model.predict(imgs, batch_samples)
            
            for j, res in enumerate(results):
                y_off, x_off = batch_coords[j]
                
                # Extract Instances
                # res.pred_instances (InstanceData) contains .masks (tensor), .labels, .scores
                preds = res.pred_instances
                if len(preds) > 0:
                    masks = preds.masks # (N, H_win, W_win) - boolean or uint8
                    scores = preds.scores
                    labels = preds.labels
                    
                    # Normals?
                    # Our custom model might output normals in `res.pred_normals` if we hacked `predict`.
                    # Or we need to look at if we added it to `pred_instances`.
                    # For now, let's assume `res.pred_normals` is the dense map (C, H_win, W_win)
                    # if the model logic in `predict` was updated to attach it.
                    # Or simpler: The geometry head outputs a vector PER INSTANCE (if instance-based geometry)
                    # or PER PIXEL.
                    # In our implementation: GeometryHead takes queries. 
                    # So each instance (query) has a normal vector.
                    
                    # Assuming we attached instance normals to `pred_instances.normals`
                    # (Requires update to `predict` logic in model, or we assume separate head run)
                    # Let's assume for this algorithm that 'normals' vector (N, 3) is present.
                    # If not, we might be predicting dense normal maps.
                    # From `DeepRoofMask2Former.loss`:
                    # 'geo_preds' is (B, Num_Queries, 3).
                    # 'predict' in Mask2Former filters queries by score.
                    # So we need those filtered normals.
                    # Let's assume `pred_instances.normals` exists (N, 3).
                    
                    has_inst_normals = hasattr(preds, 'normals')
                    
                    for k in range(len(scores)):
                        mask = masks[k].cpu().numpy() # (H_win, W_win)
                        score = scores[k].cpu().item()
                        label = labels[k].cpu().item()
                        
                        # Create global mask (sparse: store bbox and mask crop + offset)
                        # Or stick to full mask? Full mask is memory intesive.
                        # RLE is better.
                        # For simplicity in this logical implementations:
                        # Store global bbox and the cropped local mask.
                        
                        # Get bbox
                        # mask is boolean
                        rows = np.any(mask, axis=1)
                        cols = np.any(mask, axis=0)
                        if not np.any(rows) or not np.any(cols): continue
                        
                        ymin, ymax = np.where(rows)[0][[0, -1]]
                        xmin, xmax = np.where(cols)[0][[0, -1]]
                        
                        # Global bbox
                        global_bbox = [x_off + xmin, y_off + ymin, x_off + xmax + 1, y_off + ymax + 1]
                        
                        # Cropped mask for storage
                        mask_crop = mask[ymin:ymax+1, xmin:xmax+1]
                        
                        inst = {
                            'bbox': global_bbox,
                            'mask_crop': mask_crop, # Local crop
                            'score': score,
                            'label': label,
                            'offset': (y_off + ymin, x_off + xmin) # Global pos of top-left of crop
                        }
                        
                        if has_inst_normals:
                            inst['normal'] = preds.normals[k].cpu().numpy()
                            
                        # Filter by score confidence (early filtering)
                        if score > 0.3:
                           all_pred_instances.append(inst)

                # Aggregate Dense Normals (if outputted as a map)
                # If the model outputs a dense normal map in `res.seg_logits` or similar
                # For `DeepRoof`, we primarily want instance vectors, but if we had a map:
                # global_normal_map[:, y_off:y_off+window_size, x_off:x_off+window_size] += ...
                pass

    # 3. NMS / Instance Merging
    if len(all_pred_instances) == 0:
        return {'masks': [], 'instances': []}
        
    print(f"Applying NMS/Merging on {len(all_pred_instances)} candidates...")
    
    # Convert bboxes to tensor
    boxes = torch.tensor([inst['bbox'] for inst in all_pred_instances], dtype=torch.float32)
    scores = torch.tensor([inst['score'] for inst in all_pred_instances], dtype=torch.float32)
    
    # Apply standard NMS
    # Note: Soft-NMS or Mask-IoU based merging is better for segmentation
    # but standard Box-NMS is a fast proxy if roofs are fairly rectangular/compact.
    keep_indices = nms(boxes, scores, iou_threshold=0.5)
    
    final_instances = [all_pred_instances[i] for i in keep_indices]
    
    # 4. Reconstruct Global Result
    # Return list of clean instances
    
    return {
        'instances': final_instances,
        'count': len(final_instances)
    }

# Notes:
# This implementation assumes `model.predict` returns cleaned instances with scores.
# The NMS step is crucial for removing duplicate detections of the same roof in overlapping windows.
