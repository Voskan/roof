#!/bin/bash

# DeepRoof-2026: Unified Dataset Acquisition Script
# This script downloads and extracts OmniCity, RoofN3D, and UrbanScene3D.

set -e

# Target RAW Directories
RAW_OMNICITY="datasets/OmniCity"
RAW_ROOFN3D="datasets/RoofN3D"
RAW_URBANSCENE3D="datasets/UrbanScene3D"

# Target PROCESSED Directories (as requested by user)
PROC_OMNICITY="data/processed/omnicity"
PROC_ROOFN3D="data/processed/roofn3d"
PROC_URBANSCENE3D="data/processed/urbanscene3d"

mkdir -p "${RAW_OMNICITY}" "${RAW_ROOFN3D}" "${RAW_URBANSCENE3D}"
mkdir -p "${PROC_OMNICITY}" "${PROC_ROOFN3D}" "${PROC_URBANSCENE3D}"

# --- 1. OmniCity (via OpenXLab) ---
echo "--- Acquiring OmniCity ---"
if ! command -v openxlab &> /dev/null; then
    echo "Installing openxlab CLI..."
    pip install openxlab
fi

# Note: openxlab login might be required interactively if not already logged in
echo "Downloading OmniCity to ${RAW_OMNICITY}..."
# Using corrected flags: -r for repo and -s / for entire directory
openxlab dataset download --dataset-repo OpenDataLab/OmniCity --source-path / --target-path "${RAW_OMNICITY}"

# Trigger OmniCity Processing (TIF/Mask -> RGB/Normal)
echo "Processing OmniCity..."
python scripts/data/process_omnicity.py --data-root "${RAW_OMNICITY}" --output-dir "${PROC_OMNICITY}"

# --- 2. RoofN3D (via TU Berlin) ---
echo "--- Acquiring RoofN3D ---"
ROOFN3D_URL="https://roofn3d.gis.tu-berlin.de/data/roofn3d_raw_data.zip"
echo "Downloading RoofN3D from TU Berlin..."
wget -c --show-progress -O "${RAW_ROOFN3D}/roofn3d_raw.zip" "${ROOFN3D_URL}"
unzip -q "${RAW_ROOFN3D}/roofn3d_raw.zip" -d "${RAW_ROOFN3D}"
rm "${RAW_ROOFN3D}/roofn3d_raw.zip"

# Trigger RoofN3D Processing (OFF -> RGB/Normal Rendering)
echo "Processing RoofN3D..."
python scripts/data/process_roofn3d.py --data-root "${RAW_ROOFN3D}" --output-dir "${PROC_ROOFN3D}"

# --- 3. UrbanScene3D (via Dropbox) ---
echo "--- Acquiring UrbanScene3D ---"
# Note: UrbanScene3D is massive (1.43TB for whole set). 
# We default to a smaller scene (e.g. Town) for safety, or prompt for full download.
URBAN_TOWN_URL="https://www.dropbox.com/sh/d2v54y64o5mcb4t/AAAnZkBQErAKNz3QeG9_qT8pa?dl=1"
echo "Downloading UrbanScene3D (Town Scene) from Dropbox..."
echo "Warning: Whole dataset is 1.43TB. Downloading Town Scene (lighter) for demonstration."
wget -c --show-progress -O "${RAW_URBANSCENE3D}/urbanscene_town.zip" "${URBAN_TOWN_URL}"
unzip -q "${RAW_URBANSCENE3D}/urbanscene_town.zip" -d "${RAW_URBANSCENE3D}"
rm "${RAW_URBANSCENE3D}/urbanscene_town.zip"

# Trigger UrbanScene3D Sliding Window Rendering
echo "Processing UrbanScene3D (Rendering Tiles)..."
# Find the mesh file in the extracted dir
MESH_FILE=$(find "${RAW_URBANSCENE3D}" -name "*.obj" -o -name "*.ply" | head -n 1)
if [ -n "$MESH_FILE" ]; then
    python scripts/data/process_urbanscene3d.py --mesh-path "$MESH_FILE" --output-dir "${PROC_URBANSCENE3D}"
else
    echo "Error: No mesh file found for UrbanScene3D."
fi

echo "--- All Datasets Downloaded and Processed ---"
echo "OmniCity: ${PROC_OMNICITY}"
echo "RoofN3D: ${PROC_ROOFN3D}"
echo "UrbanScene3D: ${PROC_URBANSCENE3D}"
