import json
import os

files_to_check = [
    '/Users/voskan/roofscope_data/ROOF3D/train/annotation_plane.json',
    '/Users/voskan/roofscope_data/ROOF3D/train/annotation_sec.json'
]

for filepath in files_to_check:
    print(f"\n--- Checking {filepath} ---")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                print(f"Keys: {data.keys()}")
                if 'categories' in data:
                    print(f"Categories: {data['categories']}")
                if 'info' in data:
                    print(f"Info: {data['info']}")
            else:
                print("Root is not a dict")
    else:
        print("File not found")
