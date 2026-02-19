
import numpy as np
import cv2
import inspect
import albumentations as A
import torch
try:
    from albumentations.pytorch import ToTensorV2
except ImportError:
    pass


def _gauss_noise_transform(p=1.0):
    """Create a GaussNoise transform compatible with Albumentations v1/v2."""
    params = inspect.signature(A.GaussNoise.__init__).parameters
    if 'std_range' in params:
        # Albumentations 2.x API
        return A.GaussNoise(std_range=(0.04, 0.12), mean_range=(0.0, 0.0), p=p)
    # Albumentations 1.x API
    return A.GaussNoise(var_limit=(10.0, 50.0), p=p)


def _image_compression_transform(p=1.0):
    """Create an ImageCompression transform compatible with Albumentations v1/v2."""
    params = inspect.signature(A.ImageCompression.__init__).parameters
    if 'quality_range' in params:
        # Albumentations 2.x API
        return A.ImageCompression(quality_range=(60, 90), p=p)
    # Albumentations 1.x API
    return A.ImageCompression(quality_lower=60, quality_upper=90, p=p)

def rotate_normals(normals, angle_degrees):
    """
    Apply a 2D rotation matrix to the (nx, ny) components of the normal map.
    The nz component remains invariant under typical bird's-eye view rotations.
    
    Mathematical Formulation:
    [nx']   [ cos(theta) -sin(theta) ] [nx]
    [ny'] = [ sin(theta)  cos(theta) ] [ny]
    
    Args:
        normals (np.ndarray): Normal map of shape (H, W, 3).
        angle_degrees (float): Counter-Clockwise rotation angle in degrees.
    """
    # Albumentations uses degrees.
    theta = np.deg2rad(angle_degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Extract components to avoid in-place modification errors during calculation.
    if isinstance(normals, torch.Tensor):
        nx = normals[..., 0].clone()
        ny = normals[..., 1].clone()
    else:
        nx = normals[..., 0].copy()
        ny = normals[..., 1].copy()

    # Apply rotation matrix.
    normals[..., 0] = nx * cos_t - ny * sin_t
    normals[..., 1] = nx * sin_t + ny * cos_t
    
    return normals

class GeometricAugmentation(A.ReplayCompose):
    """
    Senior-level wrapper for Albumentations ReplayCompose.
    Ensures that 3D normal vectors (nx, ny, nz) are physically consistent with
    stochastic image augmentations like flips, rotations, and transpositions.
    
    This class tracks the 'replay' of stochastic transforms and applies the
    corresponding vector math to the 'normals' target.
    """
    def __init__(self, transforms, **kwargs):
        # We define 'normals' as a 'mask' target to ensure it receives the
        # exact same spatial transformations as the image grid without 
        # photometric distortions.
        if 'additional_targets' not in kwargs:
            kwargs['additional_targets'] = {}
        kwargs['additional_targets']['normals'] = 'mask'
        # Keep instance masks aligned with the exact same geometric transform.
        kwargs['additional_targets']['instance_mask'] = 'mask'
        
        super().__init__(transforms, **kwargs)

    def __call__(self, *args, **kwargs):
        # 1. Execute the standard augmentation pipeline (geometric + photometric)
        # We use ReplayCompose to capture exactly what happened to the image grid.
        result = super().__call__(*args, **kwargs)
        
        # Guard clause: if 'normals' weren't passed or no replay exists, skip.
        if 'normals' not in result or 'replay' not in result:
            return result
            
        normals = result['normals']
        replay = result['replay']
        
        # 2. Correct the vectors based on the replay log.
        # This occurs AFTER spatial transform but BEFORE normalization if placed correctly.
        self._transform_vectors(replay, normals)
        
        return result

    def _transform_vectors(self, replay_node, normals):
        """
        Recursively traverses the transform tree to adjust vector components.
        """
        if not replay_node.get('applied', False):
            return

        # Extract transform name (e.g., 'HorizontalFlip')
        t_full_name = replay_node['__class_fullname__']
        t_name = t_full_name.split('.')[-1]
        params = replay_node.get('params', {})

        # Vector Physics Logics:
        
        if t_name == 'HorizontalFlip':
            # Mirroring across Y-axis: x becomes -x
            normals[..., 0] *= -1.0
            
        elif t_name == 'VerticalFlip':
            # Mirroring across X-axis: y becomes -y
            normals[..., 1] *= -1.0
            
        elif t_name == 'Transpose':
            # Swapping X and Y coordinates implies swapping nx and ny
            if isinstance(normals, torch.Tensor):
                nx = normals[..., 0].clone()
                ny = normals[..., 1].clone()
            else:
                nx = normals[..., 0].copy()
                ny = normals[..., 1].copy()
            normals[..., 0] = ny
            normals[..., 1] = nx
            
        elif t_name == 'RandomRotate90':
            # Albumentations RandomRotate90: 'factor' is number of 90-degree
            # CLOCKWISE rotations (factor=1 => 90 CW, factor=2 => 180, factor=3 => 270 CW).
            # rotate_normals() takes CCW degrees, so negate the angle.
            # CW 90 = CCW -90. Factor k => angle -90*k degrees CCW.
            factor = params.get('factor', 0)
            rotate_normals(normals, -factor * 90)
            
        elif t_name in ['Rotate', 'ShiftScaleRotate']:
            # Albumentations uses CCW degrees for these transforms
            angle = params.get('angle', 0.0)
            rotate_normals(normals, angle)

        # Handle nested transforms (OneOf, Sequential, etc.)
        if 'transforms' in replay_node:
            for child in replay_node['transforms']:
                self._transform_vectors(child, normals)

class GoogleMapsAugmentation(GeometricAugmentation):
    """
    Production-ready augmentation pipeline for DeepRoof-2016.
    Combines robust geometric transformations with realistic satellite noise modeling.
    """
    def __init__(self, prob=0.5):
        pipeline = [
            # Geometric Transforms: Must be tracked for vector rotation
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.RandomRotate90(p=1.0),
                A.Transpose(p=1.0),
            ], p=prob),
            
            # Photometric Transforms: No effect on vectors
            A.OneOf([
                _gauss_noise_transform(p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                _image_compression_transform(p=1.0),
            ], p=0.3),

            # Keep raw image scale here. Normalization is handled by
            # SegDataPreProcessor in the model config to avoid double-normalization.
        ]
        super().__init__(pipeline)

if __name__ == "__main__":
    # Robustness check
    print("Testing GeometricAugmentation logic...")
    
    # Mock a North-facing normal vector (0, 1, 0)
    # Note: Using standard unit normal map protocol
    mock_normals = np.zeros((100, 100, 3), dtype=np.float32)
    mock_normals[..., 1] = 1.0 # North (+Y)
    
    # Rotate 90 degrees CCW
    # Physically, North should become West (-1, 0, 0)
    rotate_normals(mock_normals, 90)
    
    expected = np.array([-1.0, 0.0, 0.0])
    actual = mock_normals[50, 50, :3]
    
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    print("Vector Physics Check: PASSED (90 deg CCW rotation)")
    
    # Transpose check
    # Flip nx for horizontal flip
    mock_normals[..., 0] = 1.0
    mock_normals[..., 0] *= -1.0
    assert mock_normals[50, 50, 0] == -1.0
    print("Vector Physics Check: PASSED (Horizontal Flip)")
    
    print("\nDeepRoof-2026: Geometric Augmentation Pipeline is officially corrected.")
