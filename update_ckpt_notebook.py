import nbformat
import sys
import os

nb_path = '/Users/voskan/Desktop/DeepRoof-2026/notebooks/train_deeproof.ipynb'
new_ckpt_path = '/workspace/roof/work_dirs/swin_l_finetune_v3/best_mIoU_iter_25000.pth'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == 'code' and 'CONFIG_PATH = str(project_root / \'configs/deeproof_production_swin_L.py\')' in cell.source:
            # Update the load_from logic specifically
            import re
            
            # Find the load_from assignment and replace it
            pattern = r"cfg\.load_from\s*=\s*str\(project_root\s*/\s*['\"]work_dirs/swin_l_scratch_v1/iter_40000\.pth['\"]\)"
            replacement = f"cfg.load_from = '{new_ckpt_path}'"
            
            if re.search(pattern, cell.source):
                cell.source = re.sub(pattern, replacement, cell.source)
            else:
                # Fallback: find any load_from and just insert our specific one
                cell.source = cell.source.replace("cfg.load_from = ", f"# cfg.load_from = \ncfg.load_from = '{new_ckpt_path}'")

            # Also update the print for confirmation
            cell.source = cell.source.replace("print(f'Load From:     {cfg.load_from}')", f"print(f'Load From:     {new_ckpt_path} (SOTA Best Checkpoint)')")

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Successfully updated notebook to use {new_ckpt_path}")

except Exception as e:
    print(f"Error updating notebook: {e}")
    sys.exit(1)
