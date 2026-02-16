import os
import json
from pathlib import Path
import argparse
from collections import defaultdict

def get_folder_profile(root_path: Path):
    """
    Profile a specific folder: depth, breadth, file distribution, and samples.
    """
    profile = {
        "max_depth": 0,
        "total_files": 0,
        "extensions": defaultdict(int),
        "subdirs": [],
        "samples": []
    }
    
    if not root_path.exists():
        return None

    # Breadth: Immediate subdirs
    try:
        profile["subdirs"] = [d.name for d in root_path.iterdir() if d.is_dir()]
    except Exception:
        pass

    # Walk to get depth and distribution
    root_depth = len(root_path.parts)
    for root, dirs, files in os.walk(root_path):
        curr_path = Path(root)
        curr_depth = len(curr_path.parts) - root_depth
        profile["max_depth"] = max(profile["max_depth"], curr_depth)
        
        for file in files:
            if file.startswith('.'): continue
            profile["total_files"] += 1
            ext = Path(file).suffix.lower()
            if not ext: ext = "no_ext"
            profile["extensions"][ext] += 1
            
            # Samples (limit to first 5)
            if len(profile["samples"]) < 5:
                profile["samples"].append(str(curr_path.relative_to(root_path) / file))
                
    return profile

def main():
    parser = argparse.ArgumentParser(description='Granular Inspection of OmniCity Paths')
    parser.add_argument('--report-path', type=str, default='deep_dataset_report.json', help='Path to initial JSON report')
    parser.add_argument('--data-root', type=str, default=None, help='Override datasets/ parent directory')
    args = parser.parse_args()

    if not os.path.exists(args.report_path):
        print(f"Error: Initial report not found at {args.report_path}")
        return

    with open(args.report_path, 'r') as f:
        initial_report = json.load(f)

    # Determine base path for data
    # The report has "datasets_root" which is e.g. /workspace/roof/scripts/data/datasets
    report_datasets_root = Path(initial_report.get("datasets_root", ""))
    
    base_dir = report_datasets_root
    if args.data_root:
        # If user provides data-root, we assume it's the parent of "datasets"
        passed_root = Path(args.data_root)
        if passed_root.name == 'datasets':
            base_dir = passed_root
        elif (passed_root / 'datasets').exists():
            base_dir = passed_root / 'datasets'
        else:
            base_dir = passed_root

    print(f"--- Granular OmniCity Inspection ---")
    print(f"Base Directory for Search: {base_dir}")

    inspection_results = {
        "base_dir": str(base_dir),
        "paths": {}
    }

    omnicity_info = initial_report.get("per_dataset", {}).get("OmniCity", {})
    key_paths = omnicity_info.get("stats", {}).get("key_paths", [])

    if not key_paths:
        print("No key paths found for OmniCity in report.")
        # Fallback: manually add common expected paths if report is empty
        key_paths = [
            "OmniCity/OpenDataLab___OmniCity/raw/OmniCity-dataset/satellite-level/image-satellite",
            "OmniCity/OpenDataLab___OmniCity/raw/OmniCity-dataset/satellite-level/annotation-height",
            "OmniCity/OpenDataLab___OmniCity/raw/OmniCity-dataset/satellite-level/annotation-seg"
        ]

    for path_rel in key_paths:
        # path_rel usually starts with "OmniCity/..."
        # base_dir is usually ending in ".../datasets"
        
        # Try to resolve the path logically
        full_path = base_dir / Path(path_rel).name # Simple fallback
        
        # Better: try to find where path_rel fits
        potential_path = base_dir.parent / path_rel
        if not potential_path.exists():
            potential_path = base_dir / path_rel
            if not potential_path.exists() and "OmniCity" in path_rel:
                # Handle cases where OmniCity is duplicated or missing in the join
                rel_parts = Path(path_rel).parts
                if rel_parts[0] == "OmniCity":
                    potential_path = base_dir / Path(*rel_parts[1:])
        
        full_path = potential_path
        
        # Last resort: search for the specific leaf folder name anywhere in base_dir
        if not full_path.exists():
            leaf_name = rel_parts[-1]
            print(f"  [~] Not at standard path. Searching for '{leaf_name}' recursively...")
            for found in base_dir.rglob(leaf_name):
                if found.is_dir():
                    full_path = found
                    print(f"  [+] Found at: {full_path}")
                    break

        print(f"Inspecting: {full_path}...")
        
        if full_path.exists():
            profile = get_folder_profile(full_path)
            if profile:
                profile["extensions"] = dict(profile["extensions"])
                inspection_results["paths"][str(full_path)] = profile
        else:
            print(f"  [!] Path not found: {full_path}")

    # Final Summary Output
    output_path = Path("omnicity_granular_inspection.json")
    with open(output_path, 'w') as f:
        json.dump(inspection_results, f, indent=4)
        
    print(f"\nInspection complete. Detailed report saved to: {output_path.absolute()}")
    print("Please share the content of this file for the next step.")

if __name__ == "__main__":
    main()
