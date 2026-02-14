#!/bin/bash

# DeepRoof-2026: Master Verification & Readiness Script

echo "=========================================================="
echo "    DEEPROOF-2026: SYSTEM READINESS CHECK"
echo "=========================================================="

# 1. Run Core Logic Tests
echo "[1/2] Verifying Geometry Gradient Flow..."
python tests/test_geometry_gradient.py

if [ $? -eq 0 ]; then
    echo "[2/2] Verifying Model Connectivity & Flow..."
    python tests/test_geometry_flow.py
    
    if [ $? -eq 0 ]; then
        echo "----------------------------------------------------------"
        echo "ALL SYSTEM CHECKS PASSED: DEEPROOF-2026 SYSTEM READY."
        echo "----------------------------------------------------------"
        echo ""
        echo "To commence production-grade training (4x A100 Cluster):"
        echo ">> python tools/train.py --config configs/deeproof_production_swin_L.py --gpus 4"
        echo ""
    else
        echo "ERROR: Model Connectivity Test Failed."
        exit 1
    fi
else
    echo "ERROR: Geometry Gradient Verification Failed."
    exit 1
fi
