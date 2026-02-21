import os.path as osp
from typing import Dict, List

import numpy as np
from mmseg.registry import DATASETS

from deeproof.datasets.roof_dataset import DeepRoofDataset


@DATASETS.register_module()
class UniversalRoofDataset(DeepRoofDataset):
    """
    Unified multi-source dataset for OmniCity / RoofN3D / UrbanScene3D.

    Each source entry supports:
    - data_root: dataset root
    - ann_file: split txt (contains sample ids)
    - img_suffix / seg_map_suffix / normal_suffix
    - images_dir / masks_dir / normals_dir / sam_masks_dir (optional)
    - allow_missing_normals (default: True)
    """

    def __init__(
        self,
        sources: List[Dict],
        oversample: bool = True,
        balance_seed: int = 42,
        **kwargs,
    ):
        if not sources:
            raise ValueError('UniversalRoofDataset requires a non-empty `sources` list.')
        self.sources = [self._normalize_source(s) for s in sources]
        self.oversample = bool(oversample)
        self.balance_seed = int(balance_seed)

        first = self.sources[0]
        super().__init__(
            ann_file=first['ann_file'],
            data_root=first['data_root'],
            img_suffix=first['img_suffix'],
            seg_map_suffix=first['seg_map_suffix'],
            normal_suffix=first['normal_suffix'],
            sam_map_suffix=first.get('sam_map_suffix', '.png'),
            **kwargs,
        )

    @staticmethod
    def _normalize_source(source: Dict) -> Dict:
        out = dict(source)
        out.setdefault('name', osp.basename(str(out.get('data_root', 'source'))))
        out.setdefault('img_suffix', '.jpg')
        out.setdefault('seg_map_suffix', '.png')
        out.setdefault('normal_suffix', '.npy')
        out.setdefault('sam_map_suffix', '.png')
        out.setdefault('images_dir', 'images')
        out.setdefault('masks_dir', 'masks')
        out.setdefault('normals_dir', 'normals')
        out.setdefault('sam_masks_dir', 'sam_masks')
        out.setdefault('allow_missing_normals', True)
        if 'data_root' not in out or 'ann_file' not in out:
            raise ValueError('Each source must provide both `data_root` and `ann_file`.')
        return out

    def _resolve_ann_path(self, source: Dict) -> str:
        ann_file = str(source['ann_file'])
        if osp.isabs(ann_file):
            return ann_file
        return osp.join(str(source['data_root']), ann_file)

    def _load_one_source(self, source: Dict) -> List[Dict]:
        ann_path = self._resolve_ann_path(source)
        if not osp.isfile(ann_path):
            return []

        data_root = str(source['data_root'])
        img_suffix = str(source['img_suffix'])
        seg_map_suffix = str(source['seg_map_suffix'])
        normal_suffix = str(source['normal_suffix'])
        sam_map_suffix = str(source['sam_map_suffix'])
        images_dir = str(source['images_dir'])
        masks_dir = str(source['masks_dir'])
        normals_dir = str(source['normals_dir'])
        sam_masks_dir = str(source['sam_masks_dir'])
        allow_missing_normals = bool(source['allow_missing_normals'])

        rows: List[Dict] = []
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                img_id = line.strip()
                if not img_id:
                    continue
                normal_path = osp.join(data_root, normals_dir, img_id + normal_suffix)
                valid_normal = 1.0 if osp.isfile(normal_path) else 0.0
                if valid_normal == 0.0 and not allow_missing_normals:
                    continue
                rows.append(
                    dict(
                        img_path=osp.join(data_root, images_dir, img_id + img_suffix),
                        seg_map_path=osp.join(data_root, masks_dir, img_id + seg_map_suffix),
                        normal_path=normal_path,
                        sam_map_path=osp.join(data_root, sam_masks_dir, img_id + sam_map_suffix),
                        img_id=img_id,
                        source=source['name'],
                        valid_normal=valid_normal,
                    ))
        return rows

    def _apply_source_balancing(self, grouped: Dict[str, List[Dict]]) -> List[Dict]:
        if not grouped:
            return []
        if not self.oversample:
            merged = []
            for source_rows in grouped.values():
                merged.extend(source_rows)
            return merged

        max_size = max(len(v) for v in grouped.values() if v)
        rng = np.random.default_rng(self.balance_seed)
        merged: List[Dict] = []
        for source_rows in grouped.values():
            if not source_rows:
                continue
            if len(source_rows) >= max_size:
                merged.extend(source_rows)
                continue
            full_reps = max_size // len(source_rows)
            rem = max_size % len(source_rows)
            for _ in range(full_reps):
                merged.extend([dict(r) for r in source_rows])
            if rem > 0:
                idx = rng.choice(len(source_rows), size=rem, replace=False)
                for i in idx.tolist():
                    merged.append(dict(source_rows[i]))
        return merged

    def load_data_list(self) -> List[Dict]:
        grouped: Dict[str, List[Dict]] = {}
        for source in self.sources:
            source_rows = self._load_one_source(source)
            grouped[source['name']] = source_rows

        data_list = self._apply_source_balancing(grouped)

        if self.hard_examples_file:
            hard_path = self.hard_examples_file
            if self.data_root is not None and not osp.isabs(hard_path):
                hard_path = osp.join(self.data_root, hard_path)
            if osp.isfile(hard_path):
                with open(hard_path, 'r', encoding='utf-8') as f:
                    hard_ids = {line.strip() for line in f if line.strip()}
                if hard_ids and self.hard_example_repeat > 1:
                    hard_samples = [d for d in data_list if d.get('img_id', '') in hard_ids]
                    if hard_samples:
                        extra = []
                        for _ in range(self.hard_example_repeat - 1):
                            extra.extend([dict(s) for s in hard_samples])
                        data_list.extend(extra)
        return data_list
