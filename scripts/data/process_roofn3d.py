import os
import argparse
import numpy as np
import trimesh
import pyrender
from pathlib import Path
from tqdm import tqdm
import cv2

# Set up headless rendering environment variable
# Note: For server-side rendering, you may need libosmesa6-dev and 
# to set PYOPENGL_PLATFORM to 'osmesa' or 'egl'.
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

def parse_args():
    parser = argparse.ArgumentParser(description='Process RoofN3D Dataset for DeepRoof-2026')
    parser.add_argument('--data-root', type=str, required=True, help='Path to RoofN3D .off files')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--image-size', type=int, default=512, help='Resolution of output images')
    parser.add_argument('--camera-dist', type=float, default=20.0, help='Orthographic camera distance')
    return parser.parse_args()

def process_mesh(mesh_path: Path, output_dir: Path, image_size: int, camera_dist: float):
    file_id = mesh_path.stem
    
    # 1. Load Mesh
    try:
        mesh = trimesh.load(str(mesh_path))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
    except Exception as e:
        print(f"Error loading {mesh_path}: {e}")
        return

    # 2. Normalize Mesh Position
    mesh.vertices -= mesh.bounding_box.centroid
    
    # 3. Setup Pyrender Scene
    # Create pyrender mesh
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.5, 0.5, 0.5, 1.0],
        metallicFactor=0.2,
        roughnessFactor=0.8
    )
    py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(py_mesh)

    # 4. Setup Camera (Orthographic Top-Down)
    # View looking down the -Z axis? Or -Y? Standard is looking down -Z.
    camera = pyrender.OrthographicCamera(xmag=mesh.extents.max()/1.5, ymag=mesh.extents.max()/1.5)
    
    # Position camera at [0, 0, dist] looking at origin
    camera_pose = np.eye(4)
    camera_pose[2, 3] = camera_dist
    scene.add(camera, pose=camera_pose)

    # 5. Add Lighting (Simulated Sun)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    light_pose = np.eye(4)
    light_pose[:3, :3] = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 1, 0])[:3, :3]
    scene.add(light, pose=light_pose)

    # 6. Render RGB
    r = pyrender.OffscreenRenderer(image_size, image_size)
    color, _ = r.render(scene)
    
    # 7. Generate Normal Map
    # Extract face normals and map to colors for a custom rendering pass
    # Alternatively, use trimesh face colors
    normal_mesh = mesh.copy()
    # Encode normals [-1, 1] to [0, 1] for vertex colors
    # We want (nx, ny, nz)
    v_normals = normal_mesh.vertex_normals
    normal_colors = (v_normals + 1.0) / 2.0
    normal_mesh.visual.vertex_colors = normal_colors
    
    # Re-render scene with normals as color
    normal_py_mesh = pyrender.Mesh.from_trimesh(normal_mesh, smooth=False)
    normal_scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0]) # Flat lighting
    normal_scene.add(normal_py_mesh)
    normal_scene.add(camera, pose=camera_pose)
    
    normal_img, _ = r.render(normal_scene, flags=pyrender.RenderFlags.FLAT)
    
    # 8. Save Outputs
    # RGB image
    cv2.imwrite(str(output_dir / f"{file_id}.jpg"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    
    # Precision Normal Map (Float32) - Map back from [0, 255] to [-1, 1]
    # We use the rendered image to get the per-pixel normal
    normal_data = (normal_img.astype(np.float32) / 127.5) - 1.0
    np.save(str(output_dir / f"{file_id}_normal.npy"), normal_data)
    
    r.delete()

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_files = list(data_root.glob('*.off')) + list(data_root.glob('*.obj'))
    print(f"Processing {len(mesh_files)} meshes from {data_root}...")
    
    for f in tqdm(mesh_files):
        process_mesh(f, output_dir, args.image_size, args.camera_dist)

if __name__ == '__main__':
    main()
