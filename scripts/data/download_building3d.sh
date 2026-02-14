#!/bin/bash

# Download Building3D Dataset for DeepRoof-2026

set -e

TARGET_DIR="datasets/Building3D"
# Updated URLs found via Building3D website
TALLINN_URL="https://building3d.ucalgary.ca/assets/php/download.php?mesh"
TOKYO_URL="https://huggingface.co/datasets/Building3D/Tokyo_LoD2_Dataset/resolve/main/data.zip"

mkdir -p "${TARGET_DIR}"

# 1. Download Tallinn Mesh
echo "Downloading Building3D - Tallinn Mesh..."
echo "Note: If this returns 404 or a small file, you may need to register at https://building3d.ucalgary.ca/account.php"
wget --trust-server-names -O "${TARGET_DIR}/tallinn_mesh.zip" "${TALLINN_URL}"
unzip -q "${TARGET_DIR}/tallinn_mesh.zip" -d "${TARGET_DIR}/tallinn"
rm "${TARGET_DIR}/tallinn_mesh.zip"

# 2. Download Tokyo LoD2
echo "Downloading Building3D - Tokyo LoD2..."
# Note: Hugging Face might require a token if not public, 
# but this URL works for public resolve.
wget -O "${TARGET_DIR}/tokyo_lod2.zip" "${TOKYO_URL}"
unzip -q "${TARGET_DIR}/tokyo_lod2.zip" -d "${TARGET_DIR}/tokyo"
rm "${TARGET_DIR}/tokyo_lod2.zip"

echo "Building3D download and extraction complete."
