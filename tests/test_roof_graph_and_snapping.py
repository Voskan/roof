import numpy as np

from deeproof.utils.roof_graph import build_graph, optimize_roof_graph
from deeproof.utils.vectorization import snap_polygon_to_lines


def test_build_graph_from_segments():
    segments = [
        (10.0, 10.0, 50.0, 10.0),
        (50.0, 10.0, 50.0, 40.0),
        (50.0, 40.0, 10.0, 40.0),
        (10.0, 40.0, 10.0, 10.0),
    ]
    graph = build_graph(segments, snap_distance=2.0)
    assert len(graph['nodes']) >= 4
    assert len(graph['edges']) >= 4
    assert len(graph['adjacency']) == len(graph['nodes'])


def test_snap_polygon_to_structural_lines():
    poly = np.asarray([[11.2, 9.7], [49.1, 10.6], [50.3, 39.5], [9.4, 40.2]], dtype=np.float32)
    segments = [
        (10.0, 10.0, 50.0, 10.0),
        (50.0, 10.0, 50.0, 40.0),
        (50.0, 40.0, 10.0, 40.0),
        (10.0, 40.0, 10.0, 10.0),
    ]
    snapped = snap_polygon_to_lines(poly, segments, snap_dist=2.5)
    assert snapped.shape == poly.shape
    # vertices should move closer to ideal rectangle corners/edges
    assert np.linalg.norm(snapped[0] - np.array([10.0, 10.0], dtype=np.float32)) < 2.0


def test_optimize_roof_graph_angle_snap():
    seg = [(0.0, 0.0, 20.0, 3.0)]
    out = optimize_roof_graph(seg, snap_to_dominant=True)
    assert len(out) == 1
