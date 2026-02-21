import os
import sys
import types
import tempfile
import shutil

import numpy as np
import cv2

# Minimal OpenMMLab surface to import dataset modules.
mmseg_mod = types.ModuleType('mmseg')
mmseg_registry_mod = types.ModuleType('mmseg.registry')
mmseg_datasets_mod = types.ModuleType('mmseg.datasets')
mmseg_structures_mod = types.ModuleType('mmseg.structures')
mmengine_mod = types.ModuleType('mmengine')
mmengine_structures_mod = types.ModuleType('mmengine.structures')


class DummyRegistry:
    def register_module(self, **kwargs):
        def decorator(cls):
            return cls
        return decorator


class DummyBaseSegDataset:
    def get_data_info(self, idx):
        return self.data_list[idx]


class DummySegDataSample:
    def __init__(self):
        self.metainfo = {}

    def set_metainfo(self, info):
        self.metainfo = info


class DummyPixelData:
    def __init__(self, data=None):
        self.data = data


class DummyInstanceData:
    pass


mmseg_registry_mod.DATASETS = DummyRegistry()
mmseg_datasets_mod.BaseSegDataset = DummyBaseSegDataset
mmseg_structures_mod.SegDataSample = DummySegDataSample
mmengine_structures_mod.PixelData = DummyPixelData
mmengine_structures_mod.InstanceData = DummyInstanceData

sys.modules['mmseg'] = mmseg_mod
sys.modules['mmseg.registry'] = mmseg_registry_mod
sys.modules['mmseg.datasets'] = mmseg_datasets_mod
sys.modules['mmseg.structures'] = mmseg_structures_mod
sys.modules['mmengine'] = mmengine_mod
sys.modules['mmengine.structures'] = mmengine_structures_mod

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deeproof.datasets.universal_roof_dataset import UniversalRoofDataset


def _make_sample(root: str, sample_id: str, with_normal: bool):
    os.makedirs(f'{root}/images', exist_ok=True)
    os.makedirs(f'{root}/masks', exist_ok=True)
    os.makedirs(f'{root}/normals', exist_ok=True)

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    cv2.imwrite(f'{root}/images/{sample_id}.jpg', img)
    cv2.imwrite(f'{root}/masks/{sample_id}.png', mask)
    if with_normal:
        normal = np.zeros((32, 32, 3), dtype=np.float32)
        normal[..., 2] = 1.0
        np.save(f'{root}/normals/{sample_id}.npy', normal)


def test_universal_roof_dataset_balancing_and_valid_normal():
    root_a = tempfile.mkdtemp(prefix='uniroof_a_')
    root_b = tempfile.mkdtemp(prefix='uniroof_b_')
    try:
        _make_sample(root_a, 'a1', with_normal=True)
        _make_sample(root_b, 'b1', with_normal=False)
        _make_sample(root_b, 'b2', with_normal=True)

        with open(f'{root_a}/train.txt', 'w', encoding='utf-8') as f:
            f.write('a1\n')
        with open(f'{root_b}/train.txt', 'w', encoding='utf-8') as f:
            f.write('b1\n')
            f.write('b2\n')

        dataset = object.__new__(UniversalRoofDataset)
        dataset.sources = [
            UniversalRoofDataset._normalize_source(
                dict(name='A', data_root=root_a, ann_file='train.txt')),
            UniversalRoofDataset._normalize_source(
                dict(name='B', data_root=root_b, ann_file='train.txt')),
        ]
        dataset.oversample = True
        dataset.balance_seed = 42
        dataset.hard_examples_file = None
        dataset.hard_example_repeat = 1
        dataset.data_root = None

        data_list = UniversalRoofDataset.load_data_list(dataset)
        assert len(data_list) == 4  # max source size is 2, two sources => 2 + 2
        assert sum(1 for d in data_list if d.get('source') == 'A') == 2
        assert sum(1 for d in data_list if d.get('source') == 'B') == 2
        assert any(float(d.get('valid_normal', 0.0)) == 0.0 for d in data_list)
        assert any(float(d.get('valid_normal', 0.0)) == 1.0 for d in data_list)
    finally:
        shutil.rmtree(root_a, ignore_errors=True)
        shutil.rmtree(root_b, ignore_errors=True)
