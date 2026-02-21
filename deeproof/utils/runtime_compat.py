from __future__ import annotations

from typing import Any


def _to_plain_dict(obj: Any):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj)
    except Exception:
        return obj


def apply_runtime_compat(cfg):
    """Normalize config to be robust across MMEngine/MMSeg runtime variants."""
    if cfg is None:
        return cfg

    model = cfg.get('model', None)
    if isinstance(model, dict):
        data_preprocessor = model.setdefault('data_preprocessor', {})
        has_size = data_preprocessor.get('size') is not None
        has_div = data_preprocessor.get('size_divisor') is not None
        if has_size and has_div:
            data_preprocessor.pop('size', None)
        elif (not has_size) and (not has_div):
            data_preprocessor['size_divisor'] = 32

        decode_head = _to_plain_dict(model.get('decode_head', None))
        if isinstance(decode_head, dict):
            try:
                from deeproof.models.heads.mask2former_head import DeepRoofMask2FormerHead
                decode_head = DeepRoofMask2FormerHead._upgrade_legacy_transformer_cfg(decode_head)
            except Exception:
                pass
            model['decode_head'] = decode_head

    if cfg.get('val_dataloader') is not None and cfg.get('val_evaluator') is not None and cfg.get('val_cfg') is None:
        cfg.val_cfg = dict(type='ValLoop')
    if cfg.get('test_dataloader') is not None and cfg.get('test_evaluator') is not None and cfg.get('test_cfg') is None:
        cfg.test_cfg = dict(type='TestLoop')

    if cfg.get('train_cfg') is not None:
        train_cfg = cfg.train_cfg
        loop_type = train_cfg.get('type', '')
        if loop_type == 'EpochBasedTrainLoop' and 'max_iters' in train_cfg:
            train_cfg.pop('max_iters')
        if loop_type == 'IterBasedTrainLoop' and 'max_epochs' in train_cfg:
            train_cfg.pop('max_epochs')

    for scheduler in cfg.get('param_scheduler', []):
        if isinstance(scheduler, dict) and scheduler.get('type') == 'PolyLR':
            if 'min_lr' in scheduler and 'eta_min' not in scheduler:
                scheduler['eta_min'] = scheduler.pop('min_lr')

    return cfg
