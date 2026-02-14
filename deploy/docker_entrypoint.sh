#!/bin/bash
set -e

echo "--------------------------------------------------------"
echo "    DEEPROOF-2026: DOCKER DEPLOYMENT ENTRYPOINT"
echo "--------------------------------------------------------"

# 1. Verification of Pre-trained Weights
# Ensure Swin-L backbone weights are present for fallback/init
BACKBONE_WEIGHTS="pretrain/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth"

if [ ! -f "$BACKBONE_WEIGHTS" ]; then
    echo "[Entrypoint] Pre-trained backbone weights missing. Downloading..."
    mkdir -p pretrain
    wget -O "$BACKBONE_WEIGHTS" "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth"
else
    echo "[Entrypoint] Backbone weights found."
fi

# 2. Logic & Geometry Verification
# Prevent deployment if core logic is broken
echo "[Entrypoint] Running logic verification tests..."
python tests/test_geometry_gradient.py

if [ $? -eq 0 ]; then
    echo "[Entrypoint] Verification PASSED."
else
    echo "[Entrypoint] CRITICAL: Verification FAILED. Aborting startup."
    exit 1
fi

# 3. Startup API Server
# Note: Using production config and converted weights
echo "[Entrypoint] Starting DeepRoof-2026 Production API Server..."
# Assuming a production-ready server startup command
# exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
echo "API Server initialization stub... (Deployment ready)"
