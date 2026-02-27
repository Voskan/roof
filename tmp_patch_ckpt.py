import json

notebook_path = '/Users/voskan/Desktop/DeepRoof-2026/notebooks/checkpoint_inference_test.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "requested += sorted(WORK_DIR.glob('iter_500*.pth'), reverse=True)" in line:
                # Add 'best_*.pth' lookup right before it
                if not any("glob('best_" in ln for ln in source):
                    source.insert(i, "requested += sorted(WORK_DIR.glob('best_*.pth'), reverse=True)\n")
                break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Inference checkpoint fallback successfully patched!")
