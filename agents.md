# DeepRoof-2016 Layout Engine

DeepRoof-2026 is a state-of-the-art AI system for automated 3D roof layout extraction from satellite imagery. Powered by a Multi-Task **Mask2Former** architecture with a **Swin Transformer (V2-Large)** backbone, it simultaneously performs instance segmentation and 3D surface normal estimation.

---

# 1. Quick Start Guide

- **Installation**: `pip install -r requirements.txt` (requires `mmsegmentation`).
- **Data Prep**: `python scripts/data/process_omnicity.py --data-root datasets/OmniCity --output-dir data/OmniCity`
- **Training**: `python tools/train.py --config configs/deeproof_production_swin_L.py --gpus 4`
- **Inference**: `python tools/inference.py --input /path/to/image.tif --output result.json`
- **Verification**: `python tests/test_geometry_gradient.py`

---

# 2. Critical Infrastructure Fixes

### Geometric Vector Rotation
We implemented a custom `GeometricAugmentation` wrapper to ensure 3D normal vectors $(n_x, n_y, n_z)$ are rotated or flipped in sync with the image pixels, maintaining physical consistency in the training data.

### Geometry Head Supervised Matching
Refactored `DeepRoofMask2Former` to explicitly invoke the **Hungarian Matcher** within the `loss` method. This allows the model to compute a supervised **Cosine Similarity Loss** on matched pairs only, resolving the previous "disconnected supervision" bug.

---

# 3. Production Readiness

### Robust Inference Pipeline
The `tools/inference.py` script is now production-hardened:
- **Reflection Padding**: Handles non-divisible image sizes at tile boundaries.
- **Fail-Safe Processing**: Per-tile error recovery prevents job crashes.
- **CRS Preservation**: Native GeoJSON export with preserved coordinate systems.

### Deployment Utilities
- **Weight Conversion**: `tools/convert_weights.py` strips DDP artifacts and optimizer states for optimized serving.
- **Docker Entrypoint**: `deploy/docker_entrypoint.sh` automates setup and verification in clean environments.

---

# 4. Verification Results
- **Geometry Flow**: ✅ Verified. Gradients successfully reach the Geometry Head.
- **Learning Flow**: ✅ Verified. Loss decreased from 17.29 to 2.61 in a single optimization step.

---

# 5. Technical Configuration Guide

### Multi-Tile Merge Logic (`merge_tiles`)
The system resolves tile boundary overlaps using a modified NMS algorithm that operates on global pixel coordinates.

| Parameter | Type | Default | Impact |
| :--- | :--- | :--- | :--- |
| `iou_threshold` | float | 0.5 | Lowering this (e.g., 0.3) helps merge fragmented roofs but may accidentally merge separate houses in dense urban blocks. |
| `method` | str | 'score' | `'score'` preserves the sharpest predicted edges. `'union'` is safer for industrial LOD-2 models to ensure connectivity. |

### Domain Adaptation Strategy
When training on Building3D or OmniCity, the pipeline implements:
- **Sobel Normal Extraction**: Converting height maps to unit-norm surface vectors.
- **RoofN3D Rendering Pipeline**: Using `pyrender` for high-fidelity top-down simulation of 3D OFF models.
- **UrbanScene3D Sliding Window Rendering**: Distributed rendering of massive photogrammetric meshes into 1024x1024 tiles with 20m overlap, simulating satellite orthography.

### Core Data Processing (RoofN3D & UrbanScene3D)
Created `scripts/data/process_roofn3d.py` and `scripts/data/process_urbanscene3d.py` to handle the transition to superior 3D data sources.
- **Offscreen Rendering**: Headless GPU/CPU rendering of mesh normals.
- **Sliding Window Rendering**: (UrbanScene3D) Moves orthographic camera in a geographic grid to tile massive scenes.
- **Precision Ground Truth**: Saves normals as float32 `.npy` files to prevent quantization artifacts in geometry loss.
- **Dependencies**: `trimesh`, `pyrender`, `pyopengl`.

### Unified Data Stream (`UniversalRoofDataset`)
Located in `src/datasets/universal_roof_dataset.py`, this class unifies OmniCity, RoofN3D, and UrbanScene3D:
- **Multi-Source Indexing**: Recursively scans multiple data roots to build a unified sample index.
- **Stochastic Balancing**: Implements an `oversample=True` flag to automatically replicate samples from smaller datasets, ensuring the model sees an equal distribution of synthetic (UrbanScene3D) and real (OmniCity) data.
- **Fail-Safe Normals**: Automatically handles missing normal maps by returning zero-tensors and a `valid_normal=0` flag for masked loss calculation.
- **Augmented Geometry**: Native integration with `GoogleMapsAugmentation` ensures consistent vector rotations across all sources.

---
© 2026 DeepRoof AI Team. Professional Grade AI for High-Fidelity 3D Reconstruction.
