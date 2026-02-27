import json

notebook_path = '/Users/voskan/Desktop/DeepRoof-2026/notebooks/train_deeproof.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "DATA_ROOT = project_root / 'data' / 'OmniCity'" in line:
                source[i] = "DATA_ROOT = project_root / 'data' / 'MassiveMasterDataset'\n"
            if "NOTEBOOK_TARGET_GPU_MEM_UTIL =" in line:
                source[i] = "NOTEBOOK_TARGET_GPU_MEM_UTIL = 0.90\n"

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Training notebook successfully patched!")
