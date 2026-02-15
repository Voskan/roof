#!/bin/bash

# Download OmniCity Dataset for DeepRoof-2026

set -e

# Determine the project root (assumes script is in scripts/data/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"

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
