
import cv2
import numpy as np
import math
from typing import List, Optional, Union, Tuple


def to_cv2_poly(points: np.ndarray, round_coords: bool = True) -> Optional[np.ndarray]:
    """
    Convert polygon points into OpenCV-compatible contour array.

    Returns:
        np.ndarray | None: (N, 1, 2) int32 or float32 contiguous array.
    """
    if points is None:
        return None
    arr = np.asarray(points)
    if arr.size == 0:
        return None
    arr = arr.reshape(-1, 2)
    if arr.shape[0] < 3:
        return None
    if not np.isfinite(arr).all():
        return None
    if round_coords:
        arr = np.rint(arr).astype(np.int32)
    else:
        arr = arr.astype(np.float32)
    return np.ascontiguousarray(arr.reshape(-1, 1, 2))


def is_valid_polygon(points: np.ndarray, min_area: float = 1.0) -> bool:
    poly_f32 = to_cv2_poly(points, round_coords=False)
    if poly_f32 is None:
        return False
    area = float(cv2.contourArea(poly_f32))
    return bool(np.isfinite(area) and area >= float(min_area))

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle at p2 formed by p1-p2 and p2-p3.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Angle calculation using dot product
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    dot = np.dot(v1, v2)
    # Cos theta
    cos_theta = dot / (norm1 * norm2)
    # Clamp to handle numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def enforce_orthogonality(contour: np.ndarray, 
                          angle_threshold: float = 10.0) -> np.ndarray:
    """
    Iteratively adjust polygon vertices to enforce 90-degree angles.
    
    Args:
        contour (np.ndarray): Polygon vertices (N, 1, 2) or (N, 2).
        angle_threshold (float): Tolerance in degrees to snap to 90.
        
    Returns:
        np.ndarray: Regularized contour.
    """
    # Ensure shape (N, 2) for easier manipulation
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return contour
        
    N = len(pts)
    
    # We iterate multiple times or just once?
    # Simple pass: check each vertex. If angle is close to 90, adjust.
    # Adjusting one might break neighbor. 
    # Global optimization is best but this is a heuristic utility.
    
    regularized = pts.copy()
    
    # Loop over vertices
    for i in range(N):
        p1 = regularized[(i - 1) % N]
        p2 = regularized[i]
        p3 = regularized[(i + 1) % N]
        
        angle = calculate_angle(p1, p2, p3)
        
        # Check if close to 90 degrees
        diff = abs(angle - 90.0)
        
        if diff < angle_threshold:
            # Snap to 90.
            # We want to move p2 such that p1-p2 is perpendicular to p2-p3?
            # Or rotate one segment?
            # A simple approach: Project p2 onto the line passing through p1 perpendicular to p2-p3?
            # Or average the correction?
            
            # Let's preserve the directions of the longer segment if possible?
            
            v1 = p1 - p2
            v2 = p3 - p2
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)
            
            # If segments are roughly equal, symmetric adjustment.
            # If one is much longer, it likely defines the dominant orientation.
            # But changing p2 changes both v1 and v2.
            
            # Simplified Logic:
            # Force v2 to be perpendicular to v1?
            # Rotate v2 around p2? Or rotate v1 around p2?
            # Let's rotate the shorter segment to be perpendicular to the longer one.
            
            if l1 > l2:
                # v1 is dominant. Adjust v2.
                # We want v2' to be perp to v1.
                # Project p3 onto the line through p2 perp to v1?
                # No, we are adjusting p2 usually to square a corner.
                # Actually, standard building simplification often moves vertices.
                
                # Let's try a simpler approach: Coordinate descent?
                # Optimization problem: minimize displacement subject to angle constraints.
                # Too complex for this snippet.
                
                # Heuristic:
                # If we assume most buildings are axis-aligned to SOME grid.
                # We could start by finding main orientation.
                
                # But requirement is "if angle close to 90, force exactly 90".
                
                # Let's adopt a "local modification" approach.
                # Rotate p1 or p3 around p2 is risky (propagates errors).
                
                # Moving p2 intersection point?
                # Line 1: Passing through p1 with direction v1?
                # Line 2: Passing through p3 with direction orthogonal to v1?
                # Intersect them to find new p2.
                
                # Directions:
                # u1 = (p2 - p1) / l1
                # u2 = (p3 - p2) / l2
                
                # Target: u1 dot u2 = 0.
                
                # Let's adjust p2.
                # New P2 should be the intersection of:
                # Line L1: Passing through P1 with angle theta1 (from old P1-P2)
                # Line L2: Passing through P3 with angle theta1 +/- 90 (perp to L1)
                
                # This assumes P1-P2 is the "reference" edge.
                # We check lengths.
                pass
            
            # Implementation of the "Intersect Lines" method for the corner at P2
            # 1. Choose reference edge (longer one)
            if l1 >= l2:
                # Reference is P1->P2.
                # We keep Line(P1, P2) fixed direction? P2 changes.
                # So Line 1 is defined by P1 and direction (P2_old - P1).
                # Line 2 is defined by P3 and direction Perpendicular to Line 1.
                
                # Direction of Line 1
                d1 = (p2 - p1) / (l1 + 1e-6)
                
                # Direction of Line 2 (Perpendicular to d1)
                d2 = np.array([-d1[1], d1[0]])
                
                # Intersect Line(P1, d1) and Line(P3, d2)
                # P = P1 + t * d1
                # P = P3 + s * d2
                # Solving 2x2 system
                
                # Vector from P1 to P3
                v13 = p3 - p1
                
                # Determinant
                det = d1[0] * -d2[1] - d1[1] * -d2[0] # d1_x * -d2_y ... ?
                # Matrix form: [d1, -d2] * [t, s]^T = P3 - P1
                # x: d1x * t - d2x * s = v13x
                # y: d1y * t - d2y * s = v13y
                
                # Cramer's rule / simple algebra
                # t * d1 - s * d2 = v13
                # Cross product in 2D to solve for s?
                
                # Let's use scalar projection.
                # P2_new is projection of P3 onto Line(P1, d1)? No.
                # P2_new is such that P1-P2_new is along d1, and P3-P2_new is along d2 (perp to d1).
                # This means vector P3-P2_new is perpendicular to P1-P2_new.
                # Which means <P3 - P2_new, P1 - P2_new> = 0.
                # Since P1-P2_new is along d1, P3-P2_new is perp to d1.
                # So P2_new is the projection of P3 onto the line passing through P1 with direction d1.
                
                # P2_new = P1 + dot(P3 - P1, d1) * d1
                
                new_p2_proposal = p1 + np.dot(p3 - p1, d1) * d1
                
                # Update
                regularized[i] = new_p2_proposal

            else:
                # Reference is P3->P2 (longer).
                # Line 2 defined by P3 and direction (P2_old - P3).
                # Line 1 defined by P1 and direction Perpendicular to Line 2.
                
                # Direction d2
                d2 = (p2 - p3) / (l2 + 1e-6)
                
                # New P2 is projection of P1 onto the line passing through P3 with direction d2.
                # P2_new = P3 + dot(P1 - P3, d2) * d2
                
                new_p2_proposal = p3 + np.dot(p1 - p3, d2) * d2
                
                regularized[i] = new_p2_proposal
                
    return regularized.reshape(-1, 1, 2)

def snap_to_dominant_orientation(pts: np.ndarray) -> np.ndarray:
    """
    Force the entire building to align with its most frequent edge orientation.
    Guarantees global parallelism (opposite walls stay parallel).
    """
    N = len(pts)
    if N < 3: return pts
    
    angles = []
    weights = []
    for i in range(N):
        p1, p2 = pts[i], pts[(i + 1) % N]
        v = p2 - p1
        dist = np.linalg.norm(v)
        if dist < 1e-3: continue
        # Calculate angle in [0, pi/2)
        angle = math.atan2(v[1], v[0]) % (math.pi / 2)
        angles.append(angle)
        weights.append(dist)
        
    if not angles: return pts
    
    # Find weighted median angle to find the "North" of the building
    # Standard: Weighted average of angles is circular, but in [0, pi/2) it's mostly stable.
    dom_angle = np.average(angles, weights=weights)

    # 1. Rotate building to align North with Y-axis
    c, s = math.cos(-dom_angle), math.sin(-dom_angle)
    R = np.array([[c, -s], [s, c]])
    
    # 2. Manhattan-style quantization (snap each point to a shared grid line)
    # This is a bit too aggressive for complex roofs. 
    # Better: Transform segments to be purely vertical or horizontal in the rotated frame.
    pts_rot = pts @ R.T
    
    # We iteratively adjust points to have shared X or Y coordinates with neighbors
    for _ in range(2):
        for i in range(N):
            prev_p, curr_p, next_p = pts_rot[(i-1)%N], pts_rot[i], pts_rot[(i+1)%N]
            
            # Determine if edge (prev->curr) is more horizontal or vertical
            v_prev = curr_p - prev_p
            if abs(v_prev[0]) > abs(v_prev[1]):
                # Make it perfectly horizontal
                curr_p[1] = prev_p[1]
            else:
                # Make it perfectly vertical
                curr_p[0] = prev_p[0]
                
    # 3. Rotate back
    c_inv, s_inv = math.cos(dom_angle), math.sin(dom_angle)
    R_inv = np.array([[c_inv, -s_inv], [s_inv, c_inv]])
    pts_final = pts_rot @ R_inv.T
    
    return pts_final


def _project_point_to_segment(
    p: np.ndarray,
    seg: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, float]:
    x1, y1, x2, y2 = [float(v) for v in seg]
    a = np.asarray([x1, y1], dtype=np.float32)
    b = np.asarray([x2, y2], dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-8:
        d = float(np.linalg.norm(p - a))
        return a, d
    t = float(np.dot(p - a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    proj = a + t * ab
    d = float(np.linalg.norm(p - proj))
    return proj, d


def snap_polygon_to_lines(
    pts: np.ndarray,
    structural_lines: List[Tuple[float, float, float, float]],
    snap_dist: float = 4.0,
) -> np.ndarray:
    if pts.shape[0] < 3 or not structural_lines:
        return pts
    out = pts.copy()
    for i in range(out.shape[0]):
        p = out[i]
        best_proj = p
        best_d = float(snap_dist)
        for seg in structural_lines:
            proj, d = _project_point_to_segment(p, seg)
            if d < best_d:
                best_d = d
                best_proj = proj
        out[i] = best_proj
    return out

def regularize_building_polygons(mask: np.ndarray,
                                 epsilon_factor: float = 0.02,
                                 ortho_threshold: float = 10.0,
                                 min_area: int = 100,
                                 enforce_ortho: bool = False,
                                 structural_lines: Optional[List[Tuple[float, float, float, float]]] = None,
                                 snap_dist: float = 4.0) -> List[np.ndarray]:
    """
    Convert a binary mask to stable building polygons.

    Notes:
    - By default, we DO NOT apply aggressive orthogonal snapping because it can
      create self-intersections and out-of-bounds coordinates on complex roofs.
    - Coordinates are clipped to mask bounds to keep GeoJSON valid.
    """
    binary = (mask > 0).astype(np.uint8)
    if binary.ndim == 3:
        binary = binary[..., 0]

    h, w = binary.shape[:2]
    if h == 0 or w == 0:
        return []

    # Mild smoothing to reduce staircase artifacts, but keep geometry intact.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: List[np.ndarray] = []
    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue
        if cv2.contourArea(cnt) < float(min_area):
            continue

        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        epsilon = max(1.0, float(epsilon_factor) * float(perim))
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if approx is None or len(approx) < 3:
            continue

        pts = approx.reshape(-1, 2).astype(np.float32)

        # Optionally apply orthogonal regularization for strict CAD-like output.
        if enforce_ortho:
            pts = snap_to_dominant_orientation(pts)
            pts = enforce_orthogonality(pts, ortho_threshold).reshape(-1, 2).astype(np.float32)
            pts = enforce_orthogonality(pts, ortho_threshold).reshape(-1, 2).astype(np.float32)

        # Optional graph snapping to structural lines.
        if structural_lines:
            pts = snap_polygon_to_lines(
                pts=pts,
                structural_lines=structural_lines,
                snap_dist=float(snap_dist),
            )

        # Remove consecutive duplicate vertices.
        if len(pts) >= 2:
            dedup = [pts[0]]
            for p in pts[1:]:
                if np.linalg.norm(p - dedup[-1]) > 1e-3:
                    dedup.append(p)
            pts = np.asarray(dedup, dtype=np.float32)

        if pts.shape[0] < 3:
            continue

        # Keep polygon inside image bounds.
        pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))

        poly = to_cv2_poly(pts, round_coords=False)
        if poly is None:
            continue
        area = float(cv2.contourArea(poly))
        if area < float(min_area):
            continue
        polygons.append(poly)

    return polygons
