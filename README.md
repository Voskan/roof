# DeepRoof-2026: High-Fidelity 3D Roof Layout Engine

DeepRoof-2026 is an enterprise-grade AI system for Level of Detail 2 (LOD-2) roof reconstruction from high-resolution satellite imagery. It utilizes a **Multi-Task Mask2Former** architecture with a **Swin Transformer V2** backbone to achieve precision in instance segmentation and 3D geometry estimation.

## Features
- **Instance Segmentation**: Segment individual roof facets with sharp boundaries.
- **3D Geometry**: Pixel-wise surface normal estimation for pitch and azimuth calculation.
- **Production Hardened**: Robust inference with reflection padding and fail-safe tile processing.
- **GIS Integration**: CRS-aware GeoJSON export and GeoTIFF processing.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/DeepRoof-2026.git
cd DeepRoof-2026

# Install dependencies
pip install -r requirements.txt

# Install OpenMMLab components
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmsegmentation>=1.0.0"
```

## Data Preparation

To process the OmniCity dataset for training:

```bash
python scripts/data/process_omnicity.py \
    --data-root datasets/OmniCity \
    --output-dir data/OmniCity
```

## Training

Launch production training on a 4-GPU cluster (optimized for NVIDIA A100):

```bash
python tools/train.py --config configs/deeproof_production_swin_L.py --gpus 4
```

## Inference

Run the robust inference pipeline on high-resolution GeoTIFFs:

```bash
python tools/inference.py \
    --config configs/deeproof_production_swin_L.py \
    --checkpoint checkpoints/deeproof_deploy.pth \
    --input input_area.tif \
    --output result.json \
    --min_confidence 0.6 \
    --save_viz
```

## Testing & Verification

To verify the mathematical integrity and gradient flow of the Geometry Head:

```bash
python tests/test_geometry_gradient.py
```

## Project Structure
- `configs/`: Production-ready model and training configurations.
- `deeproof/`: Core library (models, datasets, utils).
- `scripts/`: Data processing and utility scripts.
- `tools/`: Training, inference, and weight conversion tools.
- `tests/`: Unit and integration tests.

---
© 2026 DeepRoof AI Team. Professional Grade AI for High-Fidelity 3D Reconstruction.
# roof
