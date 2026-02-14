import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional

def parse_args():
    parser = argparse.ArgumentParser(description='Process Building3D Dataset for DeepRoof-2026')
    parser.add_argument('--data-root', type=str, required=True, help='Path to Building3D dataset root (containing .obj files)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--image-size', type=int, default=1024, help='Output image size (square)')
    parser.add_argument('--scale', type=float, default=10.0, help='Scale factor to fit mesh into image (pixels per unit)')
    return parser.parse_args()

def load_obj(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load vertices and faces from an OBJ file.
    Assumes simple triangular mesh.
    """
    vertices = []
    faces = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.strip().split()[1:4]])
            elif line.startswith('f '):
                # Handle face formats (v, v/vt, v/vt/vn)
                # We only care about vertex indices
                face_indices = [int(idx.split('/')[0]) - 1 for idx in line.strip().split()[1:]]
                if len(face_indices) == 3:
                    faces.append(face_indices)
                elif len(face_indices) == 4:
                    # Triangulate quad
                    faces.append([face_indices[0], face_indices[1], face_indices[2]])
                    faces.append([face_indices[0], face_indices[2], face_indices[3]])
                    
    return np.array(vertices), np.array(faces)

def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute normal vector for each face.
    """
    # Get vertices for each face
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Edges
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Cross product
    normals = np.cross(edge1, edge2)
    
    # Normalize
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    normals = normals / norm
    
    return normals

def rasterize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    image_size: int,
    scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterize mesh faces into a 2D normal map using Painter's Algorithm.
    """
    # 1. Project vertices to 2D (Orthographic Top-Down)
    # Assume Z is up, map X, Y to image coordinates
    # Center the mesh
    center = np.mean(vertices, axis=0)
    v_centered = vertices - center
    
    # Scale and translate to image center
    # Image coords: (0,0) is top-left
    # World X -> Image X
    # World Y -> Image Y (negated if Y is Up in 3D, but usually map/satellite Y is North/Up)
    
    # Simple projection:
    # x_img = x_world * scale + W/2
    # y_img = -y_world * scale + H/2 (flip Y for standard image coords)
    
    v_2d = np.zeros((len(vertices), 2))
    v_2d[:, 0] = v_centered[:, 0] * scale + image_size / 2
    v_2d[:, 1] = -v_centered[:, 1] * scale + image_size / 2
    
    # 2. Sort faces by depth (Painter's Algorithm)
    # Calculate centroid Z for each face
    face_z = np.mean(vertices[faces][:, :, 2], axis=1)
    
    # Sort indices ascending (draw low Z first? No, draw high Z (closest to camera) LAST?)
    # Satellite view: Camera is at Z = +inf looking down.
    # We see the highest points.
    # So we should draw lowest Z (ground) FIRST, highest Z (roof peaks) LAST.
    sorted_indices = np.argsort(face_z)
    
    # 3. Draw
    # Initialize buffers
    # Normal Map: RGB (0-255)
    normal_map = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    # Depth buffer (optional, for debugging)
    
    # Pre-compute colors for each face
    # Map normal [-1, 1] to [0, 255]
    # Standard Normal Map: R=X, G=Y, B=Z
    # OpenCV uses BGR
    face_colors = ((normals + 1) * 127.5).astype(np.uint8)
    # RGB -> BGR
    face_colors_bgr = face_colors[:, ::-1] 
    
    for idx in sorted_indices:
        face = faces[idx]
        pts = v_2d[face].astype(np.int32)
        
        # Color for this face
        color = face_colors_bgr[idx].tolist()
        
        # Draw filled polygon
        # If vertices are out of bounds, cv2 clips automatically
        cv2.fillPoly(normal_map, [pts], color)
        
    return normal_map

def process_single_mesh(
    file_path: Path,
    output_dir: Path,
    image_size: int,
    scale: float
):
    mesh_id = file_path.stem
    
    try:
        vertices, faces = load_obj(file_path)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return
        
    if len(faces) == 0:
        return
        
    normals = compute_face_normals(vertices, faces)
    
    # Rasterize
    normal_map = rasterize_mesh(vertices, faces, normals, image_size, scale)
    
    # Save
    out_path = output_dir / 'normals' / f"{mesh_id}.png"
    cv2.imwrite(str(out_path), normal_map)

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    
    (output_dir / 'normals').mkdir(parents=True, exist_ok=True)
    
    mesh_files = list(data_root.glob('*.obj'))
    print(f"Found {len(mesh_files)} meshes.")
    
    for f in tqdm(mesh_files):
        process_single_mesh(f, output_dir, args.image_size, args.scale)

if __name__ == '__main__':
    main()
