
import numpy as np
import torch
from typing import List, Dict, Union

def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    """
    # Overlap area
    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return 0.0
        
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
        
    return float(intersection) / float(union)

def merge_tiles(instances: List[Dict], 
                iou_threshold: float = 0.5, 
                method: str = 'score') -> List[Dict]:
    """
    Merge overlapping instances from different tiles.
    
    Args:
        instances (List[Dict]): List of instance dicts containing:
            - 'mask': Binary mask (np.ndarray) in GLOBAL coordinates (or sparse representation)
                      Note: Masks here are effectively RLE or cropped masks with offsets.
                      For IOU calculation, we need to place them in a common canvas or check bbox intersection first.
                      If masks are full-size images, it's memory heavy.
                      Ideally, inputs are bbox + cropped mask + offset.
            - 'bbox': [x1, y1, x2, y2]
            - 'score': Confidence score
            - 'label': Class label
            - 'normal': (Optional) Normal vector
        iou_threshold (float): IoU threshold to consider instances as the same object.
        method (str): Merge strategy. 'score' (keep best), 'union' (merge masks).
        
    Returns:
        List[Dict]: Filtered/Merged instances.
    """
    if not instances:
        return []
        
    # Sort by score descending
    sorted_instances = sorted(instances, key=lambda x: x['score'], reverse=True)
    
    keep = []
    
    # Simple NMS-like loop
    while len(sorted_instances) > 0:
        # Pick the highest score instance
        current = sorted_instances.pop(0)
        keep.append(current)
        
        # Compare with remaining
        remaining = []
        for other in sorted_instances:
            # 1. Check BBox Intersection first (fast filter)
            # x1, y1, x2, y2
            b1 = current['bbox']
            b2 = other['bbox']
            
            # Intersection box
            inter_x1 = max(b1[0], b2[0])
            inter_y1 = max(b1[1], b2[1])
            inter_x2 = min(b1[2], b2[2])
            inter_y2 = min(b1[3], b2[3])
            
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                # No overlap
                remaining.append(other)
                continue
                
            # 2. Compute Mask IoU on the union BBox or intersection BBox?
            # We need to construct the masks in a shared coordinate system.
            # Minimal BBox covering both
            min_x = min(b1[0], b2[0])
            min_y = min(b1[1], b2[1])
            max_x = max(b1[2], b2[2])
            max_y = max(b1[3], b2[3])
            
            w = max_x - min_x
            h = max_y - min_y
            
            # Canvas
            canvas_curr = np.zeros((h, w), dtype=bool)
            canvas_other = np.zeros((h, w), dtype=bool)
            
            # Place current mask
            # Adjust offsets
            # current['mask'] might be a crop or full mask.
            # Assuming 'mask_crop' and 'offset' from previous step inference.py
            # If input is raw full mask, adjust logic.
            # Let's assume input format from inference.py: 'mask_crop', 'offset' (y,x)
            
            if 'mask_crop' in current:
                y_off, x_off = current['offset']
                cy = y_off - min_y
                cx = x_off - min_x
                cm_h, cm_w = current['mask_crop'].shape
                # Clamp to canvas bounds (symmetric: same safety as 'other' path below)
                y1_c = max(0, cy)
                x1_c = max(0, cx)
                y2_c = min(h, cy + cm_h)
                x2_c = min(w, cx + cm_w)
                cr_y1 = y1_c - cy
                cr_x1 = x1_c - cx
                cr_y2 = cr_y1 + (y2_c - y1_c)
                cr_x2 = cr_x1 + (x2_c - x1_c)
                canvas_curr[y1_c:y2_c, x1_c:x2_c] = current['mask_crop'][cr_y1:cr_y2, cr_x1:cr_x2]
            elif 'mask' in current and current['mask'] is not None:
                # Full-size mask: place at bbox offset relative to canvas origin
                b1 = current['bbox']
                c_y1 = int(b1[1]) - min_y
                c_x1 = int(b1[0]) - min_x
                # The mask may be full image size or cropped to bbox — handle both
                m = current['mask']
                mh, mw = m.shape[:2]
                c_y2 = min(h, c_y1 + mh)
                c_x2 = min(w, c_x1 + mw)
                if c_y2 > c_y1 and c_x2 > c_x1:
                    canvas_curr[max(0, c_y1):c_y2, max(0, c_x1):c_x2] = \
                        m[:c_y2 - max(0, c_y1), :c_x2 - max(0, c_x1)]
            
            if 'mask_crop' in other:
                y_off, x_off = other['offset']
                cy = y_off - min_y
                cx = x_off - min_x
                cm_h, cm_w = other['mask_crop'].shape
                
                # Careful with bounds if something is wrong, but bbox checks should handle it
                # Ensure we don't go out of bounds (just in case)
                y1 = max(0, cy)
                x1 = max(0, cx)
                y2 = min(h, cy+cm_h)
                x2 = min(w, cx+cm_w)
                
                # Check overlapping region in mask_crop
                crop_y1 = y1 - cy
                crop_x1 = x1 - cx
                crop_y2 = crop_y1 + (y2 - y1)
                crop_x2 = crop_x1 + (x2 - x1)
                
                canvas_other[y1:y2, x1:x2] = other['mask_crop'][crop_y1:crop_y2, crop_x1:crop_x2]
            
            iou = calculate_mask_iou(canvas_curr, canvas_other)
            
            if iou > iou_threshold:
                # Merge or Suppress
                if method == 'union':
                    # Union mask into 'current'
                    # We need to update current's bbox, mask, etc.
                    # This is complex because 'current' is already in 'keep'.
                    # Updating it requires re-extracting crop etc.
                    
                    # Implementation of Union Merge:
                    # 1. Update global bbox
                    new_bbox = [min_x, min_y, max_x, max_y]
                    current['bbox'] = new_bbox
                    
                    # 2. Union Mask
                    # Union on canvas
                    union_mask = np.logical_or(canvas_curr, canvas_other)
                    current['mask_crop'] = union_mask # Now full canvas crop
                    current['offset'] = (min_y, min_x)
                    
                    # 3. Update Score? Max or Mean?
                    # Usually keep max score of the "seed".
                    # current['score'] = max(current['score'], other['score'])
                    pass
                elif method == 'score':
                    # Suppress 'other' (implicitly done by not adding to remaining)
                    # Edge case: Roof cut in half.
                    # 'current' (higher score) is the "full" roof from Tile A. 
                    # 'other' (lower score) is the "half" roof from Tile B.
                    # Overlap might be > 0.5 if half roof is mostly inside full roof.
                    # If overlap is high, we assume 'current' covers 'other'.
                    # So we drop 'other'.
                    pass
            else:
                remaining.append(other)
                
        sorted_instances = remaining
        
    return keep
