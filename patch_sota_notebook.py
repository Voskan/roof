import nbformat
import sys

nb_path = '/Users/voskan/Desktop/DeepRoof-2026/notebooks/train_deeproof.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        # Update Markdown Cell for Training Overview
        if cell.cell_type == 'markdown' and '## ⚙️ 2. Fine-Tune Training Configuration' in cell.source:
            cell.source = (
                "## 🚀 2. Absolute Ideal SOTA Training Configuration\n\n"
                "Resuming from `iter_40000.pth` with **Physical Consistency Refinements**:\n"
                "- **Config**: `deeproof_production_swin_L.py` (300 Queries, MSTrain, ShadowAug)\n"
                "- **Geometry Loss**: `PhysicallyWeightedNormalLoss` (Azimuth Stability)\n"
                "- **Vectorization**: Global Dominant Orientation Snapping (CAD-ready)\n"
                "- **Load From**: `iter_40000.pth` (Baseline weights)"
            )

        # Update Code Cell for Configuration
        if cell.cell_type == 'code' and 'CONFIG_PATH = str(project_root / \'configs/deeproof_scratch_swin_L.py\')' in cell.source:
            # Replace paths
            cell.source = cell.source.replace(
                "CONFIG_PATH = str(project_root / 'configs/deeproof_scratch_swin_L.py')",
                "CONFIG_PATH = str(project_root / 'configs/deeproof_production_swin_L.py')"
            )
            cell.source = cell.source.replace(
                "WORK_DIR = str(project_root / 'work_dirs/swin_l_finetune_v3')",
                "WORK_DIR = str(project_root / 'work_dirs/deeproof_absolute_ideal_v1')"
            )
            # Update print headers
            cell.source = cell.source.replace(
                "print('FINE-TUNE CONFIG LOADED (v3 - all regression bugs fixed)')",
                "print('SOTA CONFIG LOADED (Absolute Ideal - Physically Weighted Loss)')"
            )
            # Remove redundant safety logic that might conflict with production config
            # Production config already has window_size and data_preprocessor set correctly.
            # But let's keep it safe.

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print("Successfully patched notebooks/train_deeproof.ipynb")

except Exception as e:
    print(f"Error patching notebook: {e}")
    sys.exit(1)
