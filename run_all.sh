#!/bin/bash

# DeepRoof-2026: Project Readiness & Deployment Trigger Script

echo "===================================================="
echo "    DEEPROOF-2026: FINAL PROJECT VERIFICATION"
echo "===================================================="

# 1. Run Core Logic & Geometry Flow Unit Tests
echo "[1/2] Running Geometry Flow Unit Tests..."
python tests/test_geometry_flow.py

if [ $? -eq 0 ]; then
    echo "----------------------------------------------------"
    echo "SUCCESS: Geometry Logic & Gradient Flow Verified."
    echo "----------------------------------------------------"
    
    # 2. Check for Training Configuration
    if [ -f "configs/deeproof_production_swin_L.py" ]; then
        echo "[2/2] Production Configuration Found."
        echo ""
        echo "PROJECT DEEPROOF-2026 IS READY FOR DEPLOYMENT."
        echo ""
        echo "To start production training on 4 GPUs, run:"
        echo ">> python tools/train.py --config configs/deeproof_production_swin_L.py --gpus 4"
        echo ""
    else
        echo "ERROR: Production configuration missing!"
        exit 1
    fi
else
    echo "----------------------------------------------------"
    echo "FAIL: Geometry Logic Verification Failed."
    echo "----------------------------------------------------"
    exit 1
fi
