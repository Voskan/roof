
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# --- 1. RESILIENT MOCKING OF OPENMMLAB ---
def mock_module(full_name):
    m = MagicMock()
    sys.modules[full_name] = m
    return m

mock_mmseg = mock_module('mmseg')
mock_module('mmseg.models')
mock_module('mmseg.models.segmentors')
mock_module('mmseg.models.segmentors.mask2former')
mock_module('mmseg.models.losses')
mock_module('mmseg.registry')
mock_module('mmseg.structures')
mock_module('mmengine')
mock_mmengine_model = mock_module('mmengine.model')
mock_module('mmengine.structures')
mock_module('mmengine.config')
mock_module('mmseg.apis')

# Real Base Class for parameter tracking
class MockBaseModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

mock_mmengine_model.BaseModule = MockBaseModule

# Mock Registry behavior
sys.modules['mmseg.registry'].MODELS.register_module = lambda **kwargs: lambda x: x
sys.modules['mmseg.registry'].DATASETS.register_module = lambda **kwargs: lambda x: x

# Mock InstanceData and SegDataSample
class MockInstanceData:
    def __init__(self, normals=None):
        self.normals = normals
        self.masks = MagicMock()
        self.masks.to_tensor = lambda **kwargs: torch.ones(1, 256, 256, dtype=torch.bool)
    def __len__(self):
        return 1

sys.modules['mmengine.structures'].InstanceData = MockInstanceData

# Define Dummy Mask2Former for Inheritance
class DummyMask2Former(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.decode_head = MagicMock()
        self.data_preprocessor = MagicMock()
    def extract_feat(self, x):
        return [torch.randn(x.size(0), 256, 32, 32)]

sys.modules['mmseg.models.segmentors.mask2former'].Mask2Former = DummyMask2Former

# --- 2. IMPORT DEEPROOF COMPONENTS ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deeproof.models.heads.geometry_head import GeometryHead

# Re-mock build logic to return real GeometryHead
sys.modules['mmseg.registry'].MODELS.build = lambda cfg: GeometryHead(**{k:v for k,v in cfg.items() if k != 'type'})

from deeproof.models.deeproof_model import DeepRoofMask2Former

# --- 3. TEST LOGIC ---

class MockAssignResult:
    def __init__(self, num_queries):
        self.gt_inds = torch.zeros(num_queries, dtype=torch.long)
        self.gt_inds[0] = 1 # Match query 0 to GT instance 0

def test_geometry_gradient():
    print("====================================================")
    print("      DEEPROOF-2026: GEOMETRY GRADIENT TEST         ")
    print("====================================================")
    torch.manual_seed(0)

    # 3.1. Setup Model
    geometry_cfg = dict(type='GeometryHead', embed_dims=256, num_layers=3, hidden_dims=256)
    model = DeepRoofMask2Former(geometry_head=geometry_cfg, geometry_loss_weight=10.0)
    
    # 3.2. Create Synthetic Batch (Diagonal Normal: 0.707, 0.707, 0)
    B, C, H, W = 2, 3, 256, 256
    inputs = torch.randn(B, C, H, W)
    target_n = torch.tensor([0.707, 0.707, 0.0])
    
    data_samples = []
    for _ in range(B):
        ds = MagicMock()
        ds.metainfo = dict(img_shape=(H, W))
        ds.gt_instances = MockInstanceData(normals=target_n.unsqueeze(0))
        data_samples.append(ds)
    
    # 3.3. Mock Decode Head state
    num_queries = 100
    model.decode_head.return_value = ([torch.randn(B, num_queries, 3)], [torch.randn(B, num_queries, 64, 64)])
    model.decode_head.assigner.assign.return_value = MockAssignResult(num_queries)
    model.decode_head.loss_by_feat.return_value = {'loss_mask': torch.tensor(0.1)}
    
    # Embeddings that will be updated
    model.decode_head.last_query_embeddings = torch.randn(B, num_queries, 256, requires_grad=True)

    # 3.4. Optimizer setup
    optimizer = optim.Adam(model.geometry_head.parameters(), lr=0.01)

    # 3.5. Forward Pass 1 (Initial Loss)
    losses = model.loss(inputs, data_samples)
    loss_geo_initial = losses['loss_geometry']
    
    print(f"Initial Geometry Loss: {loss_geo_initial.item():.6f}")
    
    # Assertion 1: Verify loss exists and is positive
    assert isinstance(loss_geo_initial, torch.Tensor), "Loss is not a tensor"
    assert loss_geo_initial.item() > 0, "Initial loss should be positive"
    print("Assertion 1 Passed: loss_geometry is a valid positive scalar.")

    # 3.6. Backward Pass & Parameter Update
    loss_geo_initial.backward()
    
    # Verify gradients exist in Geometry Head
    has_grads = any(p.grad is not None for p in model.geometry_head.parameters())
    assert has_grads, "Gradients are not reaching Geometry Head parameters!"
    print("Gradients successfully detected in Geometry Head.")

    optimizer.step()

    # 3.7. Additional Optimization Steps (Verify Reduction Robustly)
    best_loss = loss_geo_initial.item()
    loss_geo_final = loss_geo_initial
    for _ in range(5):
        optimizer.zero_grad()
        losses_new = model.loss(inputs, data_samples)
        loss_geo_final = losses_new['loss_geometry']
        loss_geo_final.backward()
        optimizer.step()
        best_loss = min(best_loss, loss_geo_final.item())
    
    print(f"Final Geometry Loss:   {loss_geo_final.item():.6f}")
    
    # Assertion 3: Loss should decrease over a short optimization window
    assert best_loss < loss_geo_initial.item(), "Geometry loss DID NOT decrease after update steps!"
    print(f"Assertion 3 Passed: Best loss improved by {loss_geo_initial.item() - best_loss:.6f}")

    print("\nTEST SUCCESSFUL: Gradient flow verified from loss to head.")

if __name__ == "__main__":
    test_geometry_gradient()


def test_geometry_embedding_dim_mismatch_is_handled():
    torch.manual_seed(0)

    geometry_cfg = dict(type='GeometryHead', embed_dims=256, num_layers=3, hidden_dims=256)
    model = DeepRoofMask2Former(geometry_head=geometry_cfg, geometry_loss_weight=10.0)

    B, C, H, W = 2, 3, 64, 64
    inputs = torch.randn(B, C, H, W)
    target_n = torch.tensor([0.0, 1.0, 0.0])

    data_samples = []
    for _ in range(B):
        ds = MagicMock()
        ds.metainfo = dict(img_shape=(H, W))
        ds.gt_instances = MockInstanceData(normals=target_n.unsqueeze(0))
        data_samples.append(ds)

    num_queries = 100
    # Mimic class-logit-shaped fallback embeddings [B, Q, C_cls=4].
    model.decode_head.return_value = (
        [torch.randn(B, num_queries, 4)],
        [torch.randn(B, num_queries, 16, 16)],
    )
    model.decode_head.assigner.assign.return_value = MockAssignResult(num_queries)
    model.decode_head.loss_by_feat.return_value = {'loss_mask': torch.tensor(0.1)}
    model.decode_head.last_query_embeddings = torch.randn(B, num_queries, 4, requires_grad=True)

    losses = model.loss(inputs, data_samples)
    loss_geo = losses['loss_geometry']
    assert torch.is_tensor(loss_geo)
    assert torch.isfinite(loss_geo)
    loss_geo.backward()


def test_assigner_receives_size_aligned_masks():
    torch.manual_seed(0)

    geometry_cfg = dict(type='GeometryHead', embed_dims=256, num_layers=3, hidden_dims=256)
    model = DeepRoofMask2Former(geometry_head=geometry_cfg, geometry_loss_weight=10.0)

    B, C, H, W = 2, 3, 128, 128
    inputs = torch.randn(B, C, H, W)
    target_n = torch.tensor([0.0, 0.0, 1.0])

    data_samples = []
    for _ in range(B):
        ds = MagicMock()
        ds.metainfo = dict(img_shape=(H, W))
        ds.gt_instances = MockInstanceData(normals=target_n.unsqueeze(0))
        # Simulate larger GT mask resolution than decoder mask resolution.
        ds.gt_instances.masks.to_tensor = lambda **kwargs: torch.ones(1, 256, 256, dtype=torch.bool)
        data_samples.append(ds)

    num_queries = 100
    pred_h, pred_w = 64, 64
    model.decode_head.return_value = (
        [torch.randn(B, num_queries, 3)],
        [torch.randn(B, num_queries, pred_h, pred_w)],
    )
    model.decode_head.loss_by_feat.return_value = {'loss_mask': torch.tensor(0.1)}
    model.decode_head.last_query_embeddings = torch.randn(B, num_queries, 256, requires_grad=True)

    def _assign_side_effect(pred_instances, gt_instances, img_meta=None):
        assert pred_instances.masks.shape[-2:] == gt_instances.masks.shape[-2:]
        return MockAssignResult(pred_instances.scores.shape[0])

    model.decode_head.assigner.assign.side_effect = _assign_side_effect

    losses = model.loss(inputs, data_samples)
    assert torch.is_tensor(losses['loss_geometry'])
