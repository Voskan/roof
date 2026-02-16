import os
import zipfile
from pathlib import Path
import argparse
import shutil

def extract_zip(zip_path: Path, extract_to: Path):
    print(f"Extracting {zip_path.name} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract and Organize OmniCity v2.0')
    parser.add_argument('--data-root', type=str, default=None, help='Root directory containing OmniCity/')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    base_dir = Path(args.data_root) if args.data_root else project_root / 'datasets' / 'OmniCity'
    
    # Locate the raw directory based on report
    raw_dir = base_dir / 'OpenDataLab___OmniCity' / 'raw'
    if not raw_dir.exists():
        # Try finding anywhere inside base_dir
        for d in base_dir.rglob('raw'):
            if d.is_dir():
                raw_dir = d
                break
    
    if not raw_dir.exists():
        print(f"Error: Could not find 'raw' directory in {base_dir}")
        return

    print(f"Found OmniCity Raw directory: {raw_dir}")
    
    dataset_dir = raw_dir / 'OmniCity-dataset'
    
    # Define tasks for extraction
    tasks = [
        # Satellite Imagery
        (dataset_dir / 'satellite-level' / 'image-satellite' / 'satellite-image-view1.zip', dataset_dir / 'satellite-level' / 'image-satellite' / 'view1'),
        (dataset_dir / 'satellite-level' / 'image-satellite' / 'satellite-image-view2.zip', dataset_dir / 'satellite-level' / 'image-satellite' / 'view2'),
        (dataset_dir / 'satellite-level' / 'image-satellite' / 'satellite-image-view3.zip', dataset_dir / 'satellite-level' / 'image-satellite' / 'view3'),
        
        # Height Maps
        (dataset_dir / 'satellite-level' / 'annotation-height' / 'annotation-height-train.zip', dataset_dir / 'satellite-level' / 'annotation-height' / 'train'),
        (dataset_dir / 'satellite-level' / 'annotation-height' / 'annotation-height-test.zip', dataset_dir / 'satellite-level' / 'annotation-height' / 'test'),
        
        # Street Level (Optional but good for completeness)
        (dataset_dir / 'street-level' / 'image-mono' / 'image-mono-train.zip', dataset_dir / 'street-level' / 'image-mono' / 'train'),
        (dataset_dir / 'street-level' / 'image-mono' / 'image-mono-test.zip', dataset_dir / 'street-level' / 'image-mono' / 'test'),
    ]

    for zip_file, dest in tasks:
        if zip_file.exists():
            os.makedirs(dest, exist_ok=True)
            extract_zip(zip_file, dest)
            # Optional: remove zip after extraction to save space if requested
            # os.remove(zip_file)
        else:
            print(f"Skipping {zip_file.name} (Not found)")

    # Organizing for processing
    print("\nOrganizing files for DeepRoof processing...")
    
    # We want a simplified structure:
    # datasets/OmniCity/images/ (from satellite view1)
    # datasets/OmniCity/height/ (from height train/test)
    # datasets/OmniCity/annotations/ (already .json)
    
    target_root = base_dir
    os.makedirs(target_root / 'images', exist_ok=True)
    os.makedirs(target_root / 'height', exist_ok=True)
    os.makedirs(target_root / 'annotations', exist_ok=True)

    # 1. Copy Satellite View 1 (Images)
    view1_dir = dataset_dir / 'satellite-level' / 'image-satellite' / 'view1'
    if view1_dir.exists():
        print("Linking/Moving View1 images...")
        for img in view1_dir.rglob('*.tif'):
            shutil.copy2(img, target_root / 'images' / img.name)
            
    # 2. Copy Height Maps
    height_dir = dataset_dir / 'satellite-level' / 'annotation-height'
    if height_dir.exists():
        print("Linking/Moving Height maps...")
        for hmap in height_dir.rglob('*.tif'):
            shutil.copy2(hmap, target_root / 'height' / hmap.name)
            
    # 3. Copy Annotations
    ann_dir = dataset_dir / 'satellite-level' / 'annotation-seg'
    if ann_dir.exists():
        print("Linking/Moving Annotations...")
        for ann in ann_dir.rglob('*.json'):
            shutil.copy2(ann, target_root / 'annotations' / ann.name)

    print(f"\nOmniCity v2.0 extraction and organization complete.")
    print(f"Data organized in: {target_root}")
    print("Next step: Run python scripts/data/analyze_dataset.py and verify counts.")

if __name__ == "__main__":
    main()
