import inspect
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mmengine.config import ConfigDict
except Exception:  # pragma: no cover
    ConfigDict = None
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from mmseg.models.segmentors.mask2former import Mask2Former as Mask2FormerBase
except Exception:
    try:
        from mmseg.models.segmentors import Mask2Former as Mask2FormerBase
    except Exception:
        from mmseg.models.segmentors import EncoderDecoder as Mask2FormerBase
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmengine.structures import InstanceData
from deeproof.models.losses import CosineSimilarityLoss


@MODELS.register_module()
class DeepRoofMask2Former(Mask2FormerBase):
    """
    DeepRoof-2026 Multi-Task Segmentor.

    Architecture:
    - Backbone: Swin Transformer V2.
    - Decoder: Mask2Former Transformer Decoder.
    - Heads: Instance Mask, Classification, and Geometry (Surface Normals).

    Fixes applied in this version:
    - Bug #1: clip_grad now at max_norm=1.0 (in config, not model code)
    - Bug #3: GeometryHead now uses real per-image decoder output embeddings,
              not static learnable weights that are identical for every image.
    - Bug #5: Geometry supervision reuses Hungarian matching from decode_head
              instead of running a second independent matching (avoids inconsistency).
    """

    def __init__(self,
                 geometry_head: dict,
                 geometry_loss_weight: float = 2.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.test_cfg = self._normalize_test_cfg(getattr(self, 'test_cfg', None))

        # 1. Initialize Geometry Head (MLP that takes query embeddings)
        self.geometry_head = MODELS.build(geometry_head)

        # 2. Define Geometry Loss (Cosine Similarity)
        self.geometry_loss = CosineSimilarityLoss(loss_weight=geometry_loss_weight)
        self.geometry_loss_weight = geometry_loss_weight

    @staticmethod
    def _normalize_test_cfg(test_cfg):
        default_cfg = dict(mode='whole')
        if test_cfg is None:
            return ConfigDict(default_cfg) if ConfigDict else default_cfg
        if isinstance(test_cfg, dict):
            merged = dict(default_cfg)
            merged.update(test_cfg)
            return ConfigDict(merged) if ConfigDict else merged
        if not hasattr(test_cfg, 'mode'):
            try:
                setattr(test_cfg, 'mode', 'whole')
            except Exception:
                return ConfigDict(default_cfg) if ConfigDict else default_cfg
        if not hasattr(test_cfg, 'get'):
            return ConfigDict(dict(mode=getattr(test_cfg, 'mode', 'whole'))) if ConfigDict \
                else dict(mode=getattr(test_cfg, 'mode', 'whole'))
        return test_cfg

    def _geometry_embed_dim(self) -> int:
        if hasattr(self.geometry_head, 'embed_dims'):
            return int(self.geometry_head.embed_dims)
        for module in self.geometry_head.modules():
            if isinstance(module, nn.Linear):
                return int(module.in_features)
        return 256

    @staticmethod
    def _to_bqc(query_like, batch_size: int) -> Optional[torch.Tensor]:
        """Normalize query-like tensors to shape [B, Q, C]."""
        if query_like is None:
            return None
        if isinstance(query_like, (list, tuple)):
            if not query_like:
                return None
            query_like = query_like[-1]
        if not torch.is_tensor(query_like):
            return None
        q = query_like
        if q.ndim == 4:
            q = q[-1]
        elif q.ndim == 2:
            q = q.unsqueeze(0).expand(batch_size, -1, -1)
        if q.ndim != 3:
            return None
        if q.shape[0] != batch_size and q.shape[1] == batch_size:
            q = q.permute(1, 0, 2).contiguous()
        if q.shape[0] != batch_size:
            return None
        return q

    def _normalize_query_embeddings(
        self,
        query_like,
        batch_size: int,
        all_cls_scores,
    ) -> Optional[torch.Tensor]:
        """
        Return [B, Q, C_geo] query embeddings compatible with GeometryHead.

        FIX Bug #3: We now receive `query_like` which is the image-specific
        decoder output (set by DeepRoofMask2FormerHead after each forward).
        We only fall back to class logits (not static init weights) if the
        dynamic embedding isn't available.
        """
        query = self._to_bqc(query_like, batch_size)

        # If the head gave us dynamic embeddings, use them directly
        expected_dim = self._geometry_embed_dim()
        if query is not None:
            if query.shape[-1] == expected_dim:
                return query
            # Adapt dimension to geometry head expectation
            if query.shape[-1] < expected_dim:
                pad = query.new_zeros(*query.shape[:-1], expected_dim - query.shape[-1])
                query = torch.cat([query, pad], dim=-1)
            else:
                query = query[..., :expected_dim]
            return query

        # Last resort: class logits — at least image-specific
        cls_proxy = self._to_bqc(all_cls_scores, batch_size)
        if cls_proxy is not None:
            if cls_proxy.shape[-1] < expected_dim:
                pad = cls_proxy.new_zeros(*cls_proxy.shape[:-1], expected_dim - cls_proxy.shape[-1])
                cls_proxy = torch.cat([cls_proxy, pad], dim=-1)
            else:
                cls_proxy = cls_proxy[..., :expected_dim]
            return cls_proxy

        return None

    @staticmethod
    def _masks_to_tensor(masks, device: torch.device) -> torch.Tensor:
        if hasattr(masks, 'to_tensor'):
            masks = masks.to_tensor(device=device)
        elif not torch.is_tensor(masks):
            masks = torch.as_tensor(masks)
        return masks.to(device=device)

    @staticmethod
    def _instances_from_semantic(
        sem_data: torch.Tensor,
        min_area: int = 64,
    ) -> InstanceData:
        """Build instance predictions from semantic map as inference fallback."""
        if sem_data.ndim == 3:
            sem_map = sem_data.squeeze(0)
        else:
            sem_map = sem_data
        sem_map = sem_map.long()
        device = sem_map.device
        H, W = int(sem_map.shape[-2]), int(sem_map.shape[-1])

        masks: List[torch.Tensor] = []
        labels: List[int] = []
        scores: List[float] = []

        for cls_id in torch.unique(sem_map).tolist():
            cls_id = int(cls_id)
            if cls_id <= 0 or cls_id == 255:
                continue
            cls_mask = (sem_map == cls_id)
            if int(cls_mask.sum().item()) < min_area:
                continue

            if cv2 is None:
                masks.append(cls_mask.bool())
                labels.append(cls_id)
                scores.append(1.0)
                continue

            comp_map = cls_mask.detach().cpu().numpy().astype(np.uint8)
            num_comp, comp_labels = cv2.connectedComponents(comp_map, connectivity=8)
            for comp_idx in range(1, int(num_comp)):
                comp = (comp_labels == comp_idx)
                if int(comp.sum()) < min_area:
                    continue
                masks.append(torch.from_numpy(comp).to(device=device))
                labels.append(cls_id)
                scores.append(1.0)

        pred_instances = InstanceData()
        if masks:
            pred_instances.masks = torch.stack([m.bool() for m in masks], dim=0)
            pred_instances.labels = torch.tensor(labels, dtype=torch.long, device=device)
            pred_instances.scores = torch.tensor(scores, dtype=torch.float32, device=device)
        else:
            pred_instances.masks = torch.zeros((0, H, W), dtype=torch.bool, device=device)
            pred_instances.labels = torch.zeros((0,), dtype=torch.long, device=device)
            pred_instances.scores = torch.zeros((0,), dtype=torch.float32, device=device)
        return pred_instances

    def _prepare_gt_instances_for_assigner(
        self,
        gt_instances: InstanceData,
        pred_mask_shape: torch.Size,
        device: torch.device,
    ) -> InstanceData:
        """Resize GT masks to prediction mask resolution for Hungarian costs."""
        prepared = InstanceData()
        gt_masks = self._masks_to_tensor(gt_instances.masks, device=device)
        if gt_masks.ndim == 2:
            gt_masks = gt_masks.unsqueeze(0)

        labels = getattr(gt_instances, 'labels', None)
        if labels is None:
            labels = torch.zeros((int(gt_masks.shape[0]),), dtype=torch.long, device=device)
        elif not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        else:
            labels = labels.to(device=device, dtype=torch.long)

        target_h, target_w = int(pred_mask_shape[-2]), int(pred_mask_shape[-1])
        if gt_masks.shape[-2:] != (target_h, target_w):
            gt_masks = F.interpolate(
                gt_masks.float().unsqueeze(1),
                size=(target_h, target_w),
                mode='nearest',
            ).squeeze(1)
        prepared.labels = labels
        prepared.masks = gt_masks.float()
        return prepared

    def _loss_by_feat_compat(
        self,
        all_cls_scores: List[torch.Tensor],
        all_mask_preds: List[torch.Tensor],
        data_samples: List[SegDataSample],
    ) -> dict:
        """Support both legacy and current mmseg `loss_by_feat` signatures."""
        loss_by_feat = self.decode_head.loss_by_feat
        batch_gt_instances = [sample.gt_instances for sample in data_samples]
        batch_img_metas = [sample.metainfo for sample in data_samples]

        call_mode = 'instances_and_metas'
        for cls in type(self.decode_head).__mro__:
            func = cls.__dict__.get('loss_by_feat')
            if func is None:
                continue
            try:
                sig = inspect.signature(func)
            except (TypeError, ValueError):
                continue
            params = [p for p in sig.parameters.values() if p.name != 'self']
            if any(
                p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                for p in params
            ):
                continue
            param_names = [p.name for p in params]
            if 'data_samples' in param_names:
                call_mode = 'data_samples'
            else:
                call_mode = 'instances_and_metas'
            break

        if call_mode == 'data_samples':
            return loss_by_feat(all_cls_scores, all_mask_preds, data_samples)
        return loss_by_feat(all_cls_scores, all_mask_preds, batch_gt_instances, batch_img_metas)

    def _compute_geometry_loss_with_reused_matching(
        self,
        geo_preds: torch.Tensor,
        all_cls_scores: List[torch.Tensor],
        all_mask_preds: List[torch.Tensor],
        data_samples: List[SegDataSample],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute geometry loss reusing the Hungarian matching from decode_head.

        FIX Bug #5: Original code ran assign() again here, which could produce
        different matches than the segmentation loss, making geometry supervision
        inconsistent. Now we either reuse cached results from the head, or run
        one matching and reuse it for both segmentation and geometry.
        """
        total_geo_loss = torch.tensor(0.0, device=device, requires_grad=False)
        num_pos_total = 0

        # Try to reuse matching cached by DeepRoofMask2FormerHead.loss_by_feat
        cached_assigns = getattr(self.decode_head, 'last_assign_results', None)

        for i in range(len(data_samples)):
            img_geo_pred = geo_preds[i]          # (Q, 3)
            img_cls_pred = all_cls_scores[-1][i] # (Q, C+1)
            img_mask_pred = all_mask_preds[-1][i] # (Q, H, W)

            gt_instances = data_samples[i].gt_instances
            img_meta = data_samples[i].metainfo

            if len(gt_instances) == 0:
                continue

            # --- Get assignment result ---
            assign_result = None

            # Try cached result from decode_head (same matching as seg loss)
            if cached_assigns is not None and i < len(cached_assigns):
                assign_result = cached_assigns[i]

            # If not cached, run matching once (single, not double)
            if assign_result is None:
                pred_instances = InstanceData()
                pred_instances.scores = img_cls_pred
                pred_instances.masks = img_mask_pred
                gt_instances_for_assign = self._prepare_gt_instances_for_assigner(
                    gt_instances=gt_instances,
                    pred_mask_shape=img_mask_pred.shape,
                    device=device,
                )
                try:
                    assign_result = self.decode_head.assigner.assign(
                        pred_instances, gt_instances_for_assign, img_meta=img_meta)
                except TypeError:
                    assign_result = self.decode_head.assigner.assign(
                        pred_instances, gt_instances_for_assign, img_meta)

            # Extract positive matches
            pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1)
            if pos_inds.numel() == 0:
                continue

            matched_src_idx = pos_inds
            matched_tgt_idx = assign_result.gt_inds[pos_inds] - 1

            matched_preds = img_geo_pred[matched_src_idx]    # (N, 3)

            # Extract GT normals
            if hasattr(gt_instances, 'normals') and gt_instances.normals is not None:
                gt_normals_tensor = gt_instances.normals
                if torch.is_tensor(gt_normals_tensor):
                    gt_normals_tensor = gt_normals_tensor.to(device=device)
                else:
                    gt_normals_tensor = torch.as_tensor(gt_normals_tensor, device=device)
                matched_targets = gt_normals_tensor[matched_tgt_idx]  # (N, 3)
            else:
                # Fallback: extract from dense normal map
                dense_normals = data_samples[i].gt_normals.data  # (3, H, W)
                raw_masks = gt_instances.masks
                if hasattr(raw_masks, 'to_tensor'):
                    gt_masks = raw_masks.to_tensor(dtype=torch.bool, device=device)
                else:
                    gt_masks = raw_masks
                    if not torch.is_tensor(gt_masks):
                        gt_masks = torch.as_tensor(gt_masks)
                    gt_masks = gt_masks.to(device=device, dtype=torch.bool)

                gt_normals_list = []
                for idx in matched_tgt_idx:
                    mask = gt_masks[idx]
                    if mask.sum() == 0:
                        avg_n = torch.tensor([0.0, 0.0, 1.0], device=device)
                    else:
                        avg_n = dense_normals[:, mask].mean(dim=1)
                        avg_n = F.normalize(avg_n, p=2, dim=0)
                    gt_normals_list.append(avg_n)
                matched_targets = torch.stack(gt_normals_list)

            # Compute cosine similarity loss ONLY on valid foreground objects (label > 0)
            # Find the gt_labels for the matched targets.
            matched_labels = []
            if hasattr(gt_instances, 'labels') and gt_instances.labels is not None:
                matched_labels = gt_instances.labels[matched_tgt_idx]
            else:
                # If labels are somehow missing, assume foreground.
                matched_labels = torch.ones(len(matched_tgt_idx), dtype=torch.long, device=device)
            
            fg_mask = matched_labels > 0
            
            if fg_mask.any():
                matched_preds_fg = matched_preds[fg_mask]
                matched_targets_fg = matched_targets[fg_mask]
                loss_geo = self.geometry_loss(matched_preds_fg, matched_targets_fg)
                total_geo_loss = total_geo_loss + loss_geo * matched_preds_fg.shape[0]
                num_pos_total += matched_preds_fg.shape[0]

        if num_pos_total > 0:
            return total_geo_loss / num_pos_total
        # Zero loss that preserves gradients
        return geo_preds.sum() * 0.0

    def loss(self, inputs: torch.Tensor, data_samples: List[SegDataSample]) -> dict:
        """
        Multi-task loss: segmentation + classification + geometry.
        """
        # A. Feature Extraction (Backbone + Neck)
        x = self.extract_feat(inputs)

        # B. Mask2Former Forward (also updates decode_head.last_query_embeddings)
        all_cls_scores, all_mask_preds = self.decode_head(x, data_samples)

        # C. Segmentation Losses (also captures Hungarian matching in decode_head)
        losses = self._loss_by_feat_compat(all_cls_scores, all_mask_preds, data_samples)

        # D. Geometry Head Prediction
        if not hasattr(self.decode_head, 'last_query_embeddings'):
            losses['loss_geometry'] = all_cls_scores[0].sum() * 0.0
            return losses

        query_embeddings = self._normalize_query_embeddings(
            getattr(self.decode_head, 'last_query_embeddings', None),
            batch_size=len(data_samples),
            all_cls_scores=all_cls_scores,
        )
        if query_embeddings is None:
            losses['loss_geometry'] = all_cls_scores[0].sum() * 0.0
            return losses

        geo_preds = self.geometry_head(query_embeddings)  # [B, Q, 3]

        # E. Geometry Loss (reusing match from step C — Bug #5 fix)
        losses['loss_geometry'] = self._compute_geometry_loss_with_reused_matching(
            geo_preds=geo_preds,
            all_cls_scores=all_cls_scores,
            all_mask_preds=all_mask_preds,
            data_samples=data_samples,
            device=inputs.device,
        )

        return losses

    def predict(self, inputs: torch.Tensor, data_samples: List[SegDataSample]) -> List[SegDataSample]:
        """
        Inference with geometry predictions attached to each detected instance.
        """
        self.test_cfg = self._normalize_test_cfg(getattr(self, 'test_cfg', None))

        # Reset cache
        if hasattr(self.decode_head, 'last_query_embeddings'):
            self.decode_head.last_query_embeddings = None
        if hasattr(self.decode_head, 'last_cls_scores'):
            self.decode_head.last_cls_scores = None

        results = super().predict(inputs, data_samples)

        # Build instance predictions from semantic map if model returned semantic-only output
        for sample in results:
            pred_instances = getattr(sample, 'pred_instances', None)
            has_instances = pred_instances is not None and len(pred_instances) > 0
            if has_instances:
                continue
            pred_sem = getattr(sample, 'pred_sem_seg', None)
            sem_data = getattr(pred_sem, 'data', None) if pred_sem is not None else None
            if torch.is_tensor(sem_data):
                sample.pred_instances = self._instances_from_semantic(sem_data)

        # Attach geometry predictions using cached query embeddings
        query_cache = getattr(self.decode_head, 'last_query_embeddings', None)
        if query_cache is not None:
            query_embeddings = self._normalize_query_embeddings(
                query_cache,
                batch_size=len(data_samples),
                all_cls_scores=getattr(self.decode_head, 'last_cls_scores', []),
            )
            if query_embeddings is None:
                return results

            with torch.no_grad():
                geo_preds = self.geometry_head(query_embeddings)

            for i in range(len(results)):
                if not hasattr(results[i], 'pred_instances'):
                    continue
                insts = results[i].pred_instances
                if len(insts) > 0:
                    pred_normals = geo_preds[i]
                    if pred_normals.shape[0] < len(insts):
                        if pred_normals.shape[0] == 0:
                            pad = torch.zeros(
                                (len(insts), 3),
                                dtype=pred_normals.dtype,
                                device=pred_normals.device)
                            pred_normals = pad
                        else:
                            repeat_n = len(insts) - pred_normals.shape[0]
                            pad = pred_normals[-1:].repeat(repeat_n, 1)
                            pred_normals = torch.cat([pred_normals, pad], dim=0)
                    insts.normals = pred_normals[:len(insts)]

        return results
