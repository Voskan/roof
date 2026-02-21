from typing import Dict, List, Tuple

import cv2
import numpy as np


def edge_map_from_masks(masks: List[np.ndarray], height: int, width: int) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    for m in masks:
        if m is None:
            continue
        mm = (np.asarray(m).astype(np.uint8) > 0).astype(np.uint8)
        if mm.shape != canvas.shape:
            mm = cv2.resize(mm, (width, height), interpolation=cv2.INTER_NEAREST)
        canvas = np.maximum(canvas, mm)
    if canvas.max() == 0:
        return canvas
    grad = cv2.morphologyEx(canvas, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return grad


def detect_line_segments(edge_map: np.ndarray, threshold: int = 30) -> List[Tuple[float, float, float, float]]:
    e = (edge_map > 0).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(
        e,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(int(threshold), 1),
        minLineLength=12,
        maxLineGap=6,
    )
    out = []
    if lines is None:
        return out
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line.tolist()]
        out.append((x1, y1, x2, y2))
    return out


def _segment_angle(seg: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = seg
    return float(np.arctan2(y2 - y1, x2 - x1))


def _snap_angle(seg: Tuple[float, float, float, float], dominant_angles: List[float]) -> Tuple[float, float, float, float]:
    if not dominant_angles:
        return seg
    x1, y1, x2, y2 = seg
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length < 1e-6:
        return seg
    angle = _segment_angle(seg)
    diffs = [abs(np.arctan2(np.sin(angle - a), np.cos(angle - a))) for a in dominant_angles]
    best = dominant_angles[int(np.argmin(diffs))]
    dx = 0.5 * length * np.cos(best)
    dy = 0.5 * length * np.sin(best)
    return (cx - dx, cy - dy, cx + dx, cy + dy)


def optimize_roof_graph(
    segments: List[Tuple[float, float, float, float]],
    snap_to_dominant: bool = True,
) -> List[Tuple[float, float, float, float]]:
    if not segments:
        return []
    if not snap_to_dominant:
        return segments
    angles = np.asarray([_segment_angle(s) for s in segments], dtype=np.float32)
    # Dominant pair: theta and theta + 90deg
    dom = float(np.median(angles))
    candidates = [dom, dom + np.pi / 2.0]
    return [_snap_angle(s, candidates) for s in segments]


def build_graph(
    segments: List[Tuple[float, float, float, float]],
    snap_distance: float = 5.0,
) -> Dict:
    nodes: List[Tuple[float, float]] = []
    edges: List[Tuple[int, int]] = []

    def _find_or_add(pt: Tuple[float, float]) -> int:
        for i, q in enumerate(nodes):
            if float(np.hypot(pt[0] - q[0], pt[1] - q[1])) <= float(snap_distance):
                return i
        nodes.append(pt)
        return len(nodes) - 1

    for s in segments:
        p1 = (float(s[0]), float(s[1]))
        p2 = (float(s[2]), float(s[3]))
        i = _find_or_add(p1)
        j = _find_or_add(p2)
        if i != j:
            edges.append((i, j))

    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.uint8)
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1
    return {
        'nodes': nodes,
        'edges': edges,
        'adjacency': adj.tolist(),
    }


def extract_and_optimize_graph_from_mask(
    roof_mask: np.ndarray,
    hough_threshold: int = 30,
    snap_distance: float = 5.0,
) -> Dict:
    segments = detect_line_segments(roof_mask, threshold=hough_threshold)
    segments = optimize_roof_graph(segments, snap_to_dominant=True)
    graph = build_graph(segments, snap_distance=snap_distance)
    graph['segments'] = segments
    return graph
