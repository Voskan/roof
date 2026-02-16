import argparse
import os
import json
from pathlib import Path
from collections import defaultdict

def analyze_directory_deep(path: Path):
    """
    Perform a deep analysis of the directory, counting files by extension 
    and identifying key subdirectories.
    """
    stats = {
        "file_counts": defaultdict(int),
        "total_files": 0,
        "key_paths": [],
        "zip_files": []
    }
    
    if not path.exists():
        return stats

    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        
        # Identify "Key" folders (e.g., those containing many images or specific names)
        if any(name in root_path.name.lower() for name in ["image", "height", "annotation", "mask", "satellite", "street"]):
            stats["key_paths"].append(str(root_path.relative_to(path.parent)))
            
        for file in files:
            if file.startswith('.'): continue
            
            stats["total_files"] += 1
            ext = Path(file).suffix.lower()
            if not ext: ext = "no_ext"
            stats["file_counts"][ext] += 1
            
            if ext == ".zip":
                stats["zip_files"].append(str(root_path / file))
                
    return stats

def get_dir_tree(path: Path, max_depth: int = 4, current_depth: int = 0):
    """Generate a simple directory tree string."""
    if not path.exists() or current_depth > max_depth:
        return ""
    
    tree = ""
    try:
        items = sorted(list(path.iterdir()))
        for i, item in enumerate(items):
            if item.name.startswith('.'): continue
            
            is_last = (i == len(items) - 1)
            prefix = "  " * current_depth + ("└── " if is_last else "├── ")
            tree += prefix + item.name + ("/" if item.is_dir() else "") + "\n"
            if item.is_dir():
                tree += get_dir_tree(item, max_depth, current_depth + 1)
    except Exception as e:
        tree += "  " * current_depth + f"└── [Error: {str(e)}]\n"
    return tree

def main():
    parser = argparse.ArgumentParser(description='Deep Analysis of DeepRoof datasets')
    parser.add_argument('--data-root', type=str, default=None, help='Root directory containing datasets/')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    base_dir = Path(args.data_root) if args.data_root else project_root
    
    # Locate "datasets" directory
    datasets_dir = base_dir / 'datasets'
    if not datasets_dir.exists() and base_dir.name == 'datasets':
        datasets_dir = base_dir
    elif not datasets_dir.exists():
        # Fallback: search for a directory named "datasets" one level down
        for d in base_dir.iterdir():
            if d.is_dir() and d.name == 'datasets':
                datasets_dir = d
                break

    report = {
        "datasets_root": str(datasets_dir),
        "per_dataset": {},
        "issues": []
    }
    
    print(f"\n{'='*60}")
    print(f"DeepRoof-2026: Deep Dataset Analysis")
    print(f"Root: {datasets_dir}")
    print(f"{'='*60}\n")
    
    if not datasets_dir.exists():
        print(f"Error: Datasets directory not found. Please specify --data-root.")
        return

    target_datasets = ["OmniCity", "RoofN3D", "UrbanScene3D", "Building3D"]
    
    for ds_name in target_datasets:
        ds_path = datasets_dir / ds_name
        if not ds_path.exists():
            # Try lowercase
            ds_path = datasets_dir / ds_name.lower()
            
        if ds_path.exists():
            print(f"Analyzing {ds_name}...")
            ds_stats = analyze_directory_deep(ds_path)
            report["per_dataset"][ds_name] = {
                "stats": ds_stats,
                "structure": get_dir_tree(ds_path, max_depth=5)
            }
            
            print(f"  - Total Files: {ds_stats['total_files']}")
            for ext, count in ds_stats['file_counts'].items():
                print(f"    - {ext}: {count}")
            
            if ds_stats['zip_files']:
                print(f"  - Found {len(ds_stats['zip_files'])} ZIP archives (require extraction)")
            
            print(f"  - Key Folders Identified: {len(ds_stats['key_paths'])}")
        else:
            print(f"Skipping {ds_name} (Not found).")

    # Save Report
    report_path = project_root / 'deep_dataset_report.json'
    with open(report_path, 'w') as f:
        # Convert defaultdict to dict for JSON serialization
        clean_report = json.loads(json.dumps(report, default=lambda x: dict(x) if isinstance(x, defaultdict) else x))
        json.dump(clean_report, f, indent=4)
        
    print(f"\n{'='*60}")
    print(f"Analysis complete. Full JSON report saved to: {report_path}")
    print(f"Please share the content of the report if you need me to write a conversion script.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
