#!/bin/bash

# Download OmniCity Dataset for DeepRoof-2026

set -e

DATASET_ID="OpenDataLab/OmniCity"
TARGET_DIR="datasets/OmniCity"

echo "Checking for openxlab CLI..."
if ! command -v openxlab &> /dev/null; then
    echo "openxlab CLI not found. Installing..."
    pip install openxlab
fi

# Note: openxlab requires login for certain datasets.
# If this fails, run 'openxlab login' manually first.
echo "Downloading OmniCity to ${TARGET_DIR}..."
mkdir -p "${TARGET_DIR}"

openxlab dataset download \
    --dataset-repo "${DATASET_ID}" \
    --source-path / \
    --target-path "${TARGET_DIR}"

echo "OmniCity download complete."
