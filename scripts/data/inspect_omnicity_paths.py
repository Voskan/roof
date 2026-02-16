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
    # The report has "datasets_root" which is /workspace/roof/scripts/data/datasets
    # Key paths in report are like "OmniCity/OpenDataLab___OmniCity/..."
    # So base is the parent of datasets_root
    base_dir = Path(initial_report["datasets_root"]).parent
    if args.data_root:
        base_dir = Path(args.data_root)

    print(f"--- Granular OmniCity Inspection ---")
    print(f"Base Directory: {base_dir}")

    inspection_results = {
        "base_dir": str(base_dir),
        "paths": {}
    }

    omnicity_info = initial_report.get("per_dataset", {}).get("OmniCity", {})
    key_paths = omnicity_info.get("stats", {}).get("key_paths", [])

    if not key_paths:
        print("No key paths found for OmniCity in report.")
        return

    for path_rel in key_paths:
        full_path = base_dir / path_rel
        print(f"Inspecting: {path_rel}...")
        
        profile = get_folder_profile(full_path)
        if profile:
            # Convert defaultdict to dict
            profile["extensions"] = dict(profile["extensions"])
            inspection_results["paths"][path_rel] = profile
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
