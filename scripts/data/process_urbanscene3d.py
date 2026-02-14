import os
import argparse
import numpy as np
import trimesh
import pyrender
from pathlib import Path
from tqdm import tqdm
import cv2

# Set up headless rendering environment variable
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

def parse_args():
    parser = argparse.ArgumentParser(description='Process UrbanScene3D Dataset for DeepRoof-2026')
    parser.add_argument('--mesh-path', type=str, required=True, help='Path to UrbanScene3D .ply or .obj file')
    parser.add_argument('--output-dir', type=str, default='data/processed/urbanscene3d', help='Output directory')
    parser.add_argument('--image-size', type=int, default=1024, help='Resolution of output images (pixels)')
    parser.add_argument('--step-size', type=float, default=20.0, help='Sliding window step size (meters)')
    parser.add_argument('--meters-per-pixel', type=float, default=0.1, help='Scale factor (m/px)')
    parser.add_argument('--camera-height', type=float, default=100.0, help='Orthographic camera height')
    return parser.parse_args()

def is_valid_tile(rgb_img, normal_data, threshold=0.1):
    """
    Filter out tiles that are mostly empty (background) or have invalid geometry.
    """
    # Check for empty pixels (black in RGB and zero/nan in normals)
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    occupancy = np.mean(gray > 5) # Threshold for non-black
    
    # Check for valid normals (length should be close to 1)
    norm_lens = np.linalg.norm(normal_data, axis=-1)
    valid_normals = np.mean(norm_lens > 0.5)
    
    return occupancy > threshold and valid_normals > threshold

def process_scene(mesh_path: Path, output_dir: Path, image_size: int, step_size: float, m_per_px: float, camera_height: float):
    scene_id = mesh_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Massive Mesh
    print(f"Loading massive mesh: {mesh_path}...")
    try:
        mesh = trimesh.load(str(mesh_path))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
    except Exception as e:
        print(f"Error loading {mesh_path}: {e}")
        return

    # 2. Setup pyrender Mesh
    # Use mesh textures if available, otherwise flat grey
    if mesh.visual.kind == 'texture':
        py_mesh = pyrender.Mesh.from_trimesh(mesh)
    else:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.5, 0.5, 0.5, 1.0],
            metallicFactor=0.2,
            roughnessFactor=0.8
        )
        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])
    scene.add(py_mesh)

    # 3. Setup Camera (Orthographic)
    # xmag/ymag define the view area in meters
    view_width = image_size * m_per_px
    camera = pyrender.OrthographicCamera(xmag=view_width/2, ymag=view_width/2)
    
    # 4. Setup Renderer
    renderer = pyrender.OffscreenRenderer(image_size, image_size)

    # 5. Sliding Window Rendering
    bounds = mesh.bounds
    x_min, y_min = bounds[0, 0], bounds[0, 1]
    x_max, y_max = bounds[1, 0], bounds[1, 1]
    
    x_coords = np.arange(x_min + view_width/2, x_max, step_size)
    y_coords = np.arange(y_min + view_width/2, y_max, step_size)
    
    total_tiles = len(x_coords) * len(y_coords)
    print(f"Generating up to {total_tiles} tiles...")

    # Lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    light_pose = np.eye(4)
    light_pose[:3, :3] = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 1, 0])[:3, :3]
    scene.add(light, pose=light_pose)

    saved_count = 0
    for i, x in enumerate(tqdm(x_coords, desc="X Grid")):
        for j, y in enumerate(y_coords):
            tile_id = f"{scene_id}_tile_{i}_{j}"
            
            # Position camera
            camera_pose = np.eye(4)
            camera_pose[0, 3] = x
            camera_pose[1, 3] = y
            camera_pose[2, 3] = camera_height
            
            cam_node = scene.add(camera, pose=camera_pose)
            
            # Pass 1: RGB
            rgb, _ = renderer.render(scene)
            
            # Pass 2: Geometry (Normal Map)
            # Temporary scene for normal extraction
            normal_mesh = mesh.copy()
            normal_mesh.visual.vertex_colors = (normal_mesh.vertex_normals + 1.0) / 2.0
            normal_py_mesh = pyrender.Mesh.from_trimesh(normal_mesh, smooth=False)
            
            normal_scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
            normal_scene.add(normal_py_mesh)
            normal_scene.add(camera, pose=camera_pose)
            
            normal_img, _ = renderer.render(normal_scene, flags=pyrender.RenderFlags.FLAT)
            
            # Clean up scenes
            scene.remove_node(cam_node)
            
            # Validation
            normal_data = (normal_img.astype(np.float32) / 127.5) - 1.0
            if is_valid_tile(rgb, normal_data):
                # Save
                cv2.imwrite(str(output_dir / f"{tile_id}_rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                np.save(str(output_dir / f"{tile_id}_normal.npy"), normal_data)
                saved_count += 1
                
    renderer.delete()
    print(f"Finished. Saved {saved_count} valid tiles to {output_dir}")

def main():
    args = parse_args()
    process_scene(
        Path(args.mesh_path),
        Path(args.output_dir),
        args.image_size,
        args.step_size,
        args.meters_per_pixel,
        args.camera_height
    )

if __name__ == '__main__':
    main()
