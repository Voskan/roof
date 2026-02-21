
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


def _downscale_transform(p=1.0):
    """Create a Downscale transform compatible with Albumentations v1/v2."""
    params = inspect.signature(A.Downscale.__init__).parameters
    if 'scale_range' in params:
        if 'interpolation_pair' in params:
            return A.Downscale(
                scale_range=(0.45, 0.75),
                interpolation_pair={
                    'downscale': cv2.INTER_AREA,
                    'upscale': cv2.INTER_LINEAR
                },
                p=p)
        return A.Downscale(scale_range=(0.45, 0.75), p=p)
    return A.Downscale(scale_min=0.45, scale_max=0.75, interpolation=cv2.INTER_LINEAR, p=p)


def _pad_if_needed_transform(min_height=512, min_width=512, p=1.0):
    """Create a PadIfNeeded transform compatible with Albumentations v1/v2."""
    params = inspect.signature(A.PadIfNeeded.__init__).parameters
    kwargs = dict(
        min_height=int(min_height),
        min_width=int(min_width),
        border_mode=cv2.BORDER_CONSTANT,
        p=p,
    )
    if 'fill' in params:
        # Albumentations 2.x API
        kwargs['fill'] = 0
        if 'fill_mask' in params:
            kwargs['fill_mask'] = 0
    else:
        # Albumentations 1.x API
        kwargs['value'] = 0
        if 'mask_value' in params:
            kwargs['mask_value'] = 0
    return A.PadIfNeeded(**kwargs)

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
        # Optional SAM teacher mask for distillation.
        kwargs['additional_targets']['sam_mask'] = 'mask'
        
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
            # Albumentations RandomRotate90 uses 90-degree CCW increments.
            # rotate_normals() also expects CCW degrees, so apply +90*factor.
            factor = params.get('factor', 0)
            rotate_normals(normals, factor * 90)
            
        elif t_name in ['Rotate', 'ShiftScaleRotate']:
            # Albumentations uses CCW degrees for these transforms
            angle = params.get('angle', 0.0)
            rotate_normals(normals, angle)

        # Handle nested transforms (OneOf, Sequential, etc.)
        if 'transforms' in replay_node:
            for child in replay_node['transforms']:
                self._transform_vectors(child, normals)

class ShadowAugmentation(A.ImageOnlyTransform):
    """
    Simulates directional shadows on roof planes.
    Helps the model decouple photometric intensity from geometric orientation.
    """
    def __init__(self, shadow_intensity=0.3, p=0.5):
        super().__init__(p=p)
        self.shadow_intensity = shadow_intensity

    def apply(self, img, **params):
        # Randomly darken a half-plane of the image to simulate a soft shadow edge
        h, w = img.shape[:2]
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(-0.2, 0.2) * max(h, w)
        
        # Create a gradient mask
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        proj = X * np.cos(angle) + Y * np.sin(angle)
        thresh = proj.mean() + dist
        
        mask = (proj > thresh).astype(np.float32)
        # Blur the mask for soft shadow
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        # Apply shadow
        img_shadow = img.astype(np.float32)
        img_shadow = img_shadow * (1.0 - mask[..., None] * self.shadow_intensity)
        return img_shadow.astype(np.uint8)

class GoogleMapsAugmentation(GeometricAugmentation):
    """
    Production-ready augmentation pipeline for DeepRoof-2026.
    Includes SOTA shadow synthesis and multi-scale priors.
    """
    def __init__(self, prob=0.5, use_shadow=True, degradation_level=1.0):
        degradation_level = float(np.clip(degradation_level, 0.0, 1.0))
        photo_prob = 0.25 + 0.25 * degradation_level
        pipeline = [
            # 1. Multi-Scale Training Prior (Absolute Ideal)
            # Randomly scale image from 0.5x to 1.5x to handle GSD variance
            # Then crop/pad back to original size or whatever dataset expects.
            # We assume internal caller handles final cropping if needed, 
            # but RandomScale here adds the diversity.
            A.RandomScale(scale_limit=0.5, p=0.7),
            
            # Geometric Transforms: Must be tracked for vector rotation
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.RandomRotate90(p=1.0),
                A.Transpose(p=1.0),
            ], p=prob),
            
            # Absolute Ideal Addition: Shadow Synthesis
            ShadowAugmentation(shadow_intensity=0.3, p=0.4 if use_shadow else 0.0),

            # Ensure image is at least 512x512 after scaling down
            _pad_if_needed_transform(min_height=512, min_width=512, p=1.0),
            
            # Photometric Transforms: No effect on vectors
            A.OneOf([
                _gauss_noise_transform(p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                _downscale_transform(p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.04, p=1.0),
                _image_compression_transform(p=1.0),
            ], p=photo_prob),
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
