import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS

@MODELS.register_module()
class GeometryHead(BaseModule):
    """
    Geometry Head for DeepRoof-2026.
    Predicts 3D surface normal vector (nx, ny, nz) for each query.
    
    Args:
        embed_dims (int): Input embedding dimension.
        num_layers (int): Number of MLP layers.
        hidden_dims (int): Hidden dimension size.
    """
    def __init__(self,
                 embed_dims=256,
                 num_layers=3,
                 hidden_dims=256,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(embed_dims if i == 0 else hidden_dims, hidden_dims))
            layers.append(nn.ReLU(inplace=True))
            
        # Final layer outputs 3 values (nx, ny, nz)
        layers.append(nn.Linear(hidden_dims if num_layers > 1 else embed_dims, 3))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): Query embeddings of shape (batch_size, num_queries, embed_dims)
            
        Returns:
            Tensor: Predicted normal vectors of shape (batch_size, num_queries, 3)
                    Vectors are normalized to unit length.
        """
        out = self.layers(x)
        
        # L2 Normalization to ensure unit vectors
        # eps added for numerical stability
        out = F.normalize(out, p=2, dim=-1, eps=1e-6)
        
        return out
