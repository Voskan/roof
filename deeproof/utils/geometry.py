
import numpy as np
import torch
from typing import Union

def get_slope(normal: Union[np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Calculate the slope (pitch) angle from the normal vector.
    
    The slope is the angle between the normal vector and the vertical UP vector (0, 0, 1).
    A flat roof has a normal (0, 0, 1) -> Slope 0 deg.
    A vertical wall has a normal (x, y, 0) -> Slope 90 deg.
    
    Args:
        normal (np.ndarray or torch.Tensor): Normal vector(s) of shape (..., 3).
                                             Assumed to be normalized to unit length.
                                             
    Returns:
        float or array: Slope angle in degrees. Range [0, 90].
    """
    # Extract z component
    if isinstance(normal, torch.Tensor):
        nz = normal[..., 2]
        # Clamp to avoid numerical errors with arccos
        nz = torch.clamp(nz, -1.0, 1.0)
        # Pitch = arccos(nz)
        # Note: If normal points DOWN (nz < 0), slope > 90. 
        # Usually roof normals point UP. We take abs to be safe or assume valid data.
        slope_rad = torch.arccos(torch.abs(nz))
        slope_deg = torch.rad2deg(slope_rad)
    else:
        nz = normal[..., 2]
        nz = np.clip(nz, -1.0, 1.0)
        slope_rad = np.arccos(np.abs(nz))
        slope_deg = np.degrees(slope_rad)
        
    return slope_deg

def get_azimuth(normal: Union[np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Calculate the azimuth (orientation/aspect) from the normal vector.
    
    Azimuth is the direction the roof surface faces, projected onto the XY plane.
    Convention:
    0 deg: East (x+)
    90 deg: North (y+)
    180 deg: West (x-)
    270 deg / -90 deg: South (y-)
    
    Args:
        normal (np.ndarray or torch.Tensor): Normal vector(s) of shape (..., 3).
        
    Returns:
        float or array: Azimuth angle in degrees. Range [0, 360).
    """
    if isinstance(normal, torch.Tensor):
        nx = normal[..., 0]
        ny = normal[..., 1]
        azimuth_rad = torch.atan2(ny, nx)
        azimuth_deg = torch.rad2deg(azimuth_rad)
        # Transform range from (-180, 180] to [0, 360)
        azimuth_deg = (azimuth_deg + 360) % 360
    else:
        nx = normal[..., 0]
        ny = normal[..., 1]
        azimuth_rad = np.arctan2(ny, nx)
        azimuth_deg = np.degrees(azimuth_rad)
        azimuth_deg = (azimuth_deg + 360) % 360
        
    return azimuth_deg
