
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# --- 1. MINIMAL RESILIENT MOCKING ---
def mock_module(full_name, mock_obj=None):
    if mock_obj is None: mock_obj = MagicMock()
    sys.modules[full_name] = mock_obj
    return mock_obj

# Mock OpenMMLab
mock_module('mmseg')
mock_module('mmseg.models')
mock_module('mmseg.models.segmentors')
mock_module('mmseg.models.segmentors.mask2former')
mock_module('mmseg.models.losses')
mock_module('mmseg.registry')
mock_module('mmseg.structures')
mock_module('mmengine')
mock_module('mmengine.structures')
mock_module('mmengine.config')
mock_module('mmseg.apis')

# CRITICAL: Mock BaseModule as nn.Module but with flexible __init__
class MockBaseModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

m_model = mock_module('mmengine.model')
m_model.BaseModule = MockBaseModule

# Mock Register
sys.modules['mmseg.registry'].MODELS.register_module = lambda **kwargs: lambda x: x
sys.modules['mmseg.registry'].DATASETS.register_module = lambda **kwargs: lambda x: x

# Define Dummy Mask2Former
class DummyMask2Former(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.decode_head = MagicMock()
        self.data_preprocessor = MagicMock()
    def extract_feat(self, x):
        return [torch.randn(x.size(0), 256, 32, 32)]

sys.modules['mmseg.models.segmentors.mask2former'].Mask2Former = DummyMask2Former

# --- 2. IMPORT DEEPROOF ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deeproof.models.heads.geometry_head import GeometryHead

# Re-mock build to return a real instance
sys.modules['mmseg.registry'].MODELS.build = lambda cfg: GeometryHead(**{k:v for k,v in cfg.items() if k != 'type'})

from deeproof.models.deeproof_model import DeepRoofMask2Former

# --- 3. TEST LOGIC ---

class MockAssignResult:
    def __init__(self, num_queries):
        self.gt_inds = torch.zeros(num_queries, dtype=torch.long)
        self.gt_inds[0] = 1

def test_geometry_flow():
    print("Initializing Geometry Flow Test (FIXED INIT ENV)...")
    
    geometry_cfg = dict(type='GeometryHead', embed_dims=256, num_layers=3, hidden_dims=256)
    model = DeepRoofMask2Former(geometry_head=geometry_cfg, geometry_loss_weight=10.0)
    
    params = list(model.geometry_head.parameters())
    print(f"Geometry Head Params Count: {len(params)}")
    assert len(params) > 0, "No parameters found in Geometry Head!"
    
    # Input
    B, C, H, W = 2, 3, 256, 256
    inputs = torch.randn(B, C, H, W)
    target_n = torch.tensor([0.707, 0.707, 0.0])
    
    # GT Data
    class MockMask:
        def to_tensor(self, **kwargs): return torch.ones(1, 256, 256, dtype=torch.bool)
    
    data_samples = []
    for _ in range(B):
        ds = MagicMock()
        ds.metainfo = dict(img_shape=(H, W))
        gt_inst = MagicMock()
        gt_inst.__len__.return_value = 1
        gt_inst.normals = target_n.unsqueeze(0)
        gt_inst.masks = MockMask()
        ds.gt_instances = gt_inst
        ds.gt_normals = MagicMock()
        ds.gt_normals.data = target_n.view(3, 1, 1).expand(3, H, W)
        data_samples.append(ds)
    
    num_queries = 100
    model.decode_head.return_value = ([torch.randn(B, num_queries, 3)], [torch.randn(B, num_queries, 64, 64)])
    model.decode_head.assigner.assign.return_value = MockAssignResult(num_queries)
    model.decode_head.loss_by_feat.return_value = {'loss_cls': torch.tensor(0.1)}
    model.decode_head.last_query_embeddings = torch.randn(B, num_queries, 256, requires_grad=True)

    # Optimizer (SGD for simplicity)
    optimizer = optim.SGD(model.geometry_head.parameters(), lr=0.1)
    
    print("Starting Optimization Loop...")
    initial_loss = None
    for i in range(15):
        optimizer.zero_grad()
        losses = model.loss(inputs, data_samples)
        loss_geo = losses['loss_geometry']
        if i == 0: initial_loss = loss_geo.item()
        loss_geo.backward()
        optimizer.step()
        if i % 5 == 0: print(f"  Iteration {i}: Loss = {loss_geo.item():.6f}")

    final_loss = loss_geo.item()
    print(f"Result: Initial {initial_loss:.6f} -> Final {final_loss:.6f}")
    assert final_loss < initial_loss, "Loss didn't decrease"
    print("SUCCESS: Gradient flow verified.")

if __name__ == "__main__":
    test_geometry_flow()
