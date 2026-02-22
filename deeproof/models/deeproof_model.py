import inspect
from typing import Any, Dict, Optional, Tuple, List
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
                 geometry_loss: Optional[dict] = None,
                 geometry_loss_weight: float = 2.0,
                 topology_loss_weight: float = 0.0,
                 dense_geometry_head: Optional[dict] = None,
                 dense_normal_loss: Optional[dict] = None,
                 dense_geometry_loss_weight: float = 0.0,
                 piecewise_planar_loss_weight: float = 0.0,
                 edge_head: Optional[dict] = None,
                 edge_loss: Optional[dict] = None,
                 edge_loss_weight: float = 0.0,
                 sam_distill_loss: Optional[dict] = None,
                 sam_distill_weight: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.test_cfg = self._normalize_test_cfg(getattr(self, 'test_cfg', None))

        # 1. Initialize Geometry Head (MLP that takes query embeddings)
        self.geometry_head = MODELS.build(geometry_head)

        # 2. Define Geometry Loss (SOTA Refinement)
        if geometry_loss is not None:
             self.geometry_loss = MODELS.build(geometry_loss)
             self.geometry_loss_weight = geometry_loss.get('loss_weight', 1.0)
        else:
             self.geometry_loss = CosineSimilarityLoss(loss_weight=geometry_loss_weight)
             self.geometry_loss_weight = geometry_loss_weight
        self.topology_loss_weight = float(topology_loss_weight)

        self.dense_geometry_head = MODELS.build(dense_geometry_head) if dense_geometry_head is not None else None
        if self.dense_geometry_head is not None:
            if dense_normal_loss is not None:
                self.dense_normal_loss = MODELS.build(dense_normal_loss)
            else:
                self.dense_normal_loss = MODELS.build(
                    dict(type='DeepRoofDenseNormalLoss', angular_weight=1.0, l1_weight=0.5, loss_weight=1.0))
        else:
            self.dense_normal_loss = None
        self.dense_geometry_loss_weight = float(dense_geometry_loss_weight)
        self.piecewise_planar_loss_weight = float(piecewise_planar_loss_weight)

        self.edge_head = MODELS.build(edge_head) if edge_head is not None else None
        if self.edge_head is not None:
            if edge_loss is not None:
                self.edge_loss = MODELS.build(edge_loss)
            else:
                self.edge_loss = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.edge_loss = None
        self.edge_loss_weight = float(edge_loss_weight)

        if sam_distill_weight > 0.0:
            if sam_distill_loss is not None:
                self.sam_distill_loss = MODELS.build(sam_distill_loss)
            else:
                self.sam_distill_loss = MODELS.build(dict(type='DeepRoofSAMDistillLoss', loss_weight=1.0))
        else:
            self.sam_distill_loss = None
        self.sam_distill_weight = float(sam_distill_weight)

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
                # Keep fallback confidence low to avoid overwhelming true
                # query-based instances when semantic-only fallback is used.
                scores.append(0.05)
                continue

            comp_map = cls_mask.detach().cpu().numpy().astype(np.uint8)
            num_comp, comp_labels = cv2.connectedComponents(comp_map, connectivity=8)
            for comp_idx in range(1, int(num_comp)):
                comp = (comp_labels == comp_idx)
                if int(comp.sum()) < min_area:
                    continue
                masks.append(torch.from_numpy(comp).to(device=device))
                labels.append(cls_id)
                scores.append(0.05)

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

    @staticmethod
    def _instances_from_queries(
        all_cls_scores: Any,
        all_mask_preds: Any,
        sample_index: int,
        out_hw: Tuple[int, int],
        num_classes: int,
        score_thr: float = 0.05,
        min_area: int = 64,
        max_instances: int = 200,
    ) -> InstanceData:
        """Build instance predictions directly from query cls/mask outputs."""
        device = None
        cls_scores = all_cls_scores
        mask_preds = all_mask_preds

        if isinstance(cls_scores, (list, tuple)):
            cls_scores = cls_scores[-1] if len(cls_scores) > 0 else None
        if isinstance(mask_preds, (list, tuple)):
            mask_preds = mask_preds[-1] if len(mask_preds) > 0 else None

        if not torch.is_tensor(cls_scores) or not torch.is_tensor(mask_preds):
            out = InstanceData()
            h, w = int(out_hw[0]), int(out_hw[1])
            out.masks = torch.zeros((0, h, w), dtype=torch.bool)
            out.labels = torch.zeros((0,), dtype=torch.long)
            out.scores = torch.zeros((0,), dtype=torch.float32)
            out.query_indices = torch.zeros((0,), dtype=torch.long)
            return out

        if cls_scores.ndim == 4:
            cls_scores = cls_scores[-1]
        if mask_preds.ndim == 5:
            mask_preds = mask_preds[-1]
        if cls_scores.ndim != 3 or mask_preds.ndim != 4:
            out = InstanceData()
            h, w = int(out_hw[0]), int(out_hw[1])
            out.masks = torch.zeros((0, h, w), dtype=torch.bool, device=cls_scores.device)
            out.labels = torch.zeros((0,), dtype=torch.long, device=cls_scores.device)
            out.scores = torch.zeros((0,), dtype=torch.float32, device=cls_scores.device)
            out.query_indices = torch.zeros((0,), dtype=torch.long, device=cls_scores.device)
            return out

        bsz = int(cls_scores.shape[0])
        if sample_index < 0 or sample_index >= bsz:
            out = InstanceData()
            h, w = int(out_hw[0]), int(out_hw[1])
            out.masks = torch.zeros((0, h, w), dtype=torch.bool, device=cls_scores.device)
            out.labels = torch.zeros((0,), dtype=torch.long, device=cls_scores.device)
            out.scores = torch.zeros((0,), dtype=torch.float32, device=cls_scores.device)
            out.query_indices = torch.zeros((0,), dtype=torch.long, device=cls_scores.device)
            return out

        cls_logits = cls_scores[sample_index]   # [Q, C]
        mask_logits = mask_preds[sample_index]  # [Q, Hm, Wm]
        device = cls_logits.device

        target_h, target_w = int(out_hw[0]), int(out_hw[1])
        if tuple(mask_logits.shape[-2:]) != (target_h, target_w):
            mask_logits = F.interpolate(
                mask_logits.unsqueeze(1),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)

        probs = F.softmax(cls_logits, dim=-1)
        total_cls_dim = int(probs.shape[-1])
        # Config uses num_classes foreground/background + no-object as extra logit.
        if total_cls_dim > int(num_classes):
            probs = probs[:, : int(num_classes)]

        if int(num_classes) <= 1 or probs.shape[-1] <= 1:
            out = InstanceData()
            out.masks = torch.zeros((0, target_h, target_w), dtype=torch.bool, device=device)
            out.labels = torch.zeros((0,), dtype=torch.long, device=device)
            out.scores = torch.zeros((0,), dtype=torch.float32, device=device)
            out.query_indices = torch.zeros((0,), dtype=torch.long, device=device)
            return out

        # Exclude class_id=0 (background) from instance generation.
        fg_probs = probs[:, 1:]
        top_scores, top_labels_rel = fg_probs.max(dim=-1)
        top_labels = top_labels_rel + 1

        mask_probs = torch.sigmoid(mask_logits)
        order = torch.argsort(top_scores, descending=True)

        keep_masks: List[torch.Tensor] = []
        keep_labels: List[torch.Tensor] = []
        keep_scores: List[torch.Tensor] = []
        keep_query_idx: List[torch.Tensor] = []

        for q_idx in order.tolist():
            cls_score = float(top_scores[q_idx].item())
            if cls_score < float(score_thr):
                continue
            q_mask_prob = mask_probs[q_idx]
            q_mask = q_mask_prob >= 0.5
            area = int(q_mask.sum().item())
            if area < int(min_area):
                continue
            # Couple classification confidence with mask confidence.
            mask_conf = float(q_mask_prob[q_mask].mean().item()) if area > 0 else 0.0
            final_score = cls_score * mask_conf
            if final_score < float(score_thr):
                continue

            keep_masks.append(q_mask.bool())
            keep_labels.append(top_labels[q_idx].long())
            keep_scores.append(torch.tensor(final_score, dtype=torch.float32, device=device))
            keep_query_idx.append(torch.tensor(q_idx, dtype=torch.long, device=device))
            if int(max_instances) > 0 and len(keep_masks) >= int(max_instances):
                break

        out = InstanceData()
        if keep_masks:
            out.masks = torch.stack(keep_masks, dim=0)
            out.labels = torch.stack(keep_labels, dim=0)
            out.scores = torch.stack(keep_scores, dim=0)
            out.query_indices = torch.stack(keep_query_idx, dim=0)
        else:
            out.masks = torch.zeros((0, target_h, target_w), dtype=torch.bool, device=device)
            out.labels = torch.zeros((0,), dtype=torch.long, device=device)
            out.scores = torch.zeros((0,), dtype=torch.float32, device=device)
            out.query_indices = torch.zeros((0,), dtype=torch.long, device=device)
        return out

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

            valid_normal = 1.0
            if isinstance(img_meta, dict):
                valid_normal = float(img_meta.get('valid_normal', 1.0))
            else:
                valid_normal = float(getattr(img_meta, 'valid_normal', 1.0))
            if valid_normal <= 0.0:
                continue

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

    def _compute_topology_regularization(
        self,
        all_mask_preds: List[torch.Tensor],
        data_samples: List[SegDataSample],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Topology-aware regularization for matched foreground queries.

        Penalizes:
        - tiny unstable components
        - hole-like artifacts
        - excessive fragmentation (open/close inconsistency)
        """
        if self.topology_loss_weight <= 0.0:
            return all_mask_preds[-1].sum() * 0.0

        cached_assigns = getattr(self.decode_head, 'last_assign_results', None)
        total_loss = all_mask_preds[-1].sum() * 0.0
        num_terms = 0

        for i in range(len(data_samples)):
            gt_instances = data_samples[i].gt_instances
            if len(gt_instances) == 0:
                continue

            assign_result = None
            if cached_assigns is not None and i < len(cached_assigns):
                assign_result = cached_assigns[i]
            if assign_result is None:
                continue

            pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1)
            if pos_inds.numel() == 0:
                continue

            matched_tgt_idx = assign_result.gt_inds[pos_inds] - 1
            gt_labels = getattr(gt_instances, 'labels', None)
            if gt_labels is None:
                fg_mask = torch.ones_like(pos_inds, dtype=torch.bool, device=device)
            else:
                if not torch.is_tensor(gt_labels):
                    gt_labels = torch.as_tensor(gt_labels, dtype=torch.long, device=device)
                else:
                    gt_labels = gt_labels.to(device=device, dtype=torch.long)
                fg_mask = gt_labels[matched_tgt_idx] > 0

            fg_pos_inds = pos_inds[fg_mask]
            if fg_pos_inds.numel() == 0:
                continue

            pred_logits = all_mask_preds[-1][i][fg_pos_inds]   # [N, H, W]
            pred_prob = torch.sigmoid(pred_logits).unsqueeze(1)  # [N, 1, H, W]

            dil = F.max_pool2d(pred_prob, kernel_size=3, stride=1, padding=1)
            ero = -F.max_pool2d(-pred_prob, kernel_size=3, stride=1, padding=1)
            opened = F.max_pool2d(ero, kernel_size=3, stride=1, padding=1)
            closed = -F.max_pool2d(-dil, kernel_size=3, stride=1, padding=1)

            hole_penalty = F.l1_loss(pred_prob, closed, reduction='mean')
            frag_penalty = F.l1_loss(pred_prob, opened, reduction='mean')

            tv_h = torch.abs(pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :]).mean()
            tv_w = torch.abs(pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1]).mean()
            tv_penalty = 0.5 * (tv_h + tv_w)

            area = pred_prob.mean(dim=(2, 3))  # [N,1]
            tiny_penalty = torch.relu(0.02 - area).mean()

            loss_i = hole_penalty + frag_penalty + 0.2 * tv_penalty + 0.2 * tiny_penalty
            total_loss = total_loss + loss_i
            num_terms += 1

        if num_terms == 0:
            return all_mask_preds[-1].sum() * 0.0
        return (total_loss / float(num_terms)) * self.topology_loss_weight

    @staticmethod
    def _safe_get_metainfo(sample: SegDataSample, key: str, default=None):
        meta = getattr(sample, 'metainfo', {})
        if isinstance(meta, dict):
            return meta.get(key, default)
        return getattr(meta, key, default)

    def _compute_dense_geometry_loss(
        self,
        pred_dense_normals: torch.Tensor,
        data_samples: List[SegDataSample],
        device: torch.device,
    ) -> torch.Tensor:
        if self.dense_geometry_head is None or self.dense_normal_loss is None:
            return pred_dense_normals.sum() * 0.0
        if self.dense_geometry_loss_weight <= 0.0:
            return pred_dense_normals.sum() * 0.0

        b, _, h, w = pred_dense_normals.shape
        gt_batch = []
        valid_batch = []
        has_valid_sample = False
        for i in range(b):
            gt_n = getattr(data_samples[i], 'gt_normals', None)
            gt_data = getattr(gt_n, 'data', None) if gt_n is not None else None
            if gt_data is None:
                gt_map = torch.zeros((3, h, w), dtype=pred_dense_normals.dtype, device=device)
                valid_map = torch.zeros((h, w), dtype=pred_dense_normals.dtype, device=device)
                gt_batch.append(gt_map)
                valid_batch.append(valid_map)
                continue
            if not torch.is_tensor(gt_data):
                gt_data = torch.as_tensor(gt_data, dtype=pred_dense_normals.dtype, device=device)
            else:
                gt_data = gt_data.to(device=device, dtype=pred_dense_normals.dtype)
            if gt_data.ndim == 4:
                gt_data = gt_data.squeeze(0)
            if gt_data.shape[0] != 3 and gt_data.shape[-1] == 3:
                gt_data = gt_data.permute(2, 0, 1).contiguous()
            if gt_data.shape[-2:] != (h, w):
                gt_data = F.interpolate(
                    gt_data.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

            valid_normal = float(self._safe_get_metainfo(data_samples[i], 'valid_normal', 1.0))
            if valid_normal <= 0.0:
                valid_map = torch.zeros((h, w), dtype=pred_dense_normals.dtype, device=device)
            else:
                valid_map = (gt_data.abs().sum(dim=0) > 1e-6).float()
                if valid_map.sum() > 0:
                    has_valid_sample = True

            gt_batch.append(gt_data)
            valid_batch.append(valid_map)

        if not gt_batch:
            return pred_dense_normals.sum() * 0.0
        gt = torch.stack(gt_batch, dim=0)
        valid = torch.stack(valid_batch, dim=0)
        if not has_valid_sample:
            return pred_dense_normals.sum() * 0.0

        loss = self.dense_normal_loss(pred_dense_normals, gt, valid_mask=valid)
        return loss * self.dense_geometry_loss_weight

    def _compute_piecewise_planar_regularization(
        self,
        pred_dense_normals: torch.Tensor,
        data_samples: List[SegDataSample],
        device: torch.device,
    ) -> torch.Tensor:
        if self.piecewise_planar_loss_weight <= 0.0:
            return pred_dense_normals.sum() * 0.0

        b, _, h, w = pred_dense_normals.shape
        total = pred_dense_normals.sum() * 0.0
        n_terms = 0

        for i in range(b):
            gt_instances = getattr(data_samples[i], 'gt_instances', None)
            if gt_instances is None or len(gt_instances) == 0:
                continue
            gt_labels = getattr(gt_instances, 'labels', None)
            if gt_labels is None:
                continue
            if not torch.is_tensor(gt_labels):
                gt_labels = torch.as_tensor(gt_labels, dtype=torch.long, device=device)
            else:
                gt_labels = gt_labels.to(device=device, dtype=torch.long)

            raw_masks = gt_instances.masks
            if hasattr(raw_masks, 'to_tensor'):
                masks = raw_masks.to_tensor(dtype=torch.bool, device=device)
            else:
                if not torch.is_tensor(raw_masks):
                    masks = torch.as_tensor(raw_masks, device=device)
                else:
                    masks = raw_masks.to(device=device)
                masks = masks.bool()
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)

            if masks.shape[-2:] != (h, w):
                masks = F.interpolate(
                    masks.float().unsqueeze(1), size=(h, w), mode='nearest').squeeze(1).bool()

            pred = pred_dense_normals[i].permute(1, 2, 0).contiguous()  # [H,W,3]
            for j in range(min(masks.shape[0], gt_labels.shape[0])):
                if int(gt_labels[j].item()) <= 0:
                    continue
                m = masks[j]
                area = int(m.sum().item())
                if area < 32:
                    continue
                n_pix = pred[m]  # [N,3]
                mean_n = F.normalize(n_pix.mean(dim=0), p=2, dim=0, eps=1e-6)
                inconsistency = 1.0 - (n_pix * mean_n.unsqueeze(0)).sum(dim=-1)
                area_w = float(min(area / float(h * w), 1.0))
                total = total + inconsistency.mean() * area_w
                n_terms += 1

        if n_terms == 0:
            return pred_dense_normals.sum() * 0.0
        return (total / float(n_terms)) * self.piecewise_planar_loss_weight

    @staticmethod
    def _mask_to_edge(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        dil = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=1)
        ero = -F.max_pool2d(-mask.float(), kernel_size=3, stride=1, padding=1)
        edge = (dil - ero).clamp(min=0.0, max=1.0)
        return edge

    def _compute_edge_loss(
        self,
        edge_logits: torch.Tensor,
        data_samples: List[SegDataSample],
        device: torch.device,
    ) -> torch.Tensor:
        if self.edge_head is None or self.edge_loss is None:
            return edge_logits.sum() * 0.0
        if self.edge_loss_weight <= 0.0:
            return edge_logits.sum() * 0.0

        b, _, h, w = edge_logits.shape
        targets = []
        valid = []
        for i in range(b):
            gt_instances = getattr(data_samples[i], 'gt_instances', None)
            edge_t = torch.zeros((1, h, w), dtype=edge_logits.dtype, device=device)
            if gt_instances is not None and len(gt_instances) > 0:
                raw_masks = gt_instances.masks
                if hasattr(raw_masks, 'to_tensor'):
                    masks = raw_masks.to_tensor(dtype=torch.bool, device=device)
                else:
                    if not torch.is_tensor(raw_masks):
                        masks = torch.as_tensor(raw_masks, device=device)
                    else:
                        masks = raw_masks.to(device=device)
                    masks = masks.bool()
                if masks.ndim == 2:
                    masks = masks.unsqueeze(0)
                if masks.shape[-2:] != (h, w):
                    masks = F.interpolate(
                        masks.float().unsqueeze(1), size=(h, w), mode='nearest').squeeze(1).bool()
                union = masks.any(dim=0, keepdim=True).float()
                edge_t = self._mask_to_edge(union)[0]
            else:
                gt_sem = getattr(data_samples[i], 'gt_sem_seg', None)
                gt_sem_data = getattr(gt_sem, 'data', None) if gt_sem is not None else None
                if gt_sem_data is not None:
                    if not torch.is_tensor(gt_sem_data):
                        gt_sem_data = torch.as_tensor(gt_sem_data, device=device)
                    else:
                        gt_sem_data = gt_sem_data.to(device=device)
                    if gt_sem_data.ndim == 3:
                        roof = (gt_sem_data[0] > 0).float().unsqueeze(0)
                    else:
                        roof = (gt_sem_data > 0).float().unsqueeze(0)
                    if roof.shape[-2:] != (h, w):
                        roof = F.interpolate(
                            roof.unsqueeze(0), size=(h, w), mode='nearest').squeeze(0)
                    edge_t = self._mask_to_edge(roof)[0]
            targets.append(edge_t)
            valid.append(torch.ones((1, h, w), dtype=edge_logits.dtype, device=device))

        tgt = torch.stack(targets, dim=0)
        vm = torch.stack(valid, dim=0)

        if isinstance(self.edge_loss, nn.BCEWithLogitsLoss):
            loss = F.binary_cross_entropy_with_logits(edge_logits, tgt, reduction='none')
            loss = (loss * vm).sum() / vm.sum().clamp(min=1.0)
        else:
            loss = self.edge_loss(edge_logits, tgt)
        return loss * self.edge_loss_weight

    def _compute_sam_distill_loss(
        self,
        all_cls_scores: List[torch.Tensor],
        all_mask_preds: List[torch.Tensor],
        data_samples: List[SegDataSample],
        device: torch.device,
    ) -> torch.Tensor:
        if self.sam_distill_loss is None or self.sam_distill_weight <= 0.0:
            return all_mask_preds[-1].sum() * 0.0

        cls_logits = all_cls_scores[-1]     # [B,Q,C+1]
        mask_logits = all_mask_preds[-1]    # [B,Q,H,W]
        cls_prob = torch.softmax(cls_logits, dim=-1)
        if cls_prob.shape[-1] >= 3:
            roof_q = cls_prob[..., 1:-1].sum(dim=-1)  # foreground classes, exclude bg/no-object
        else:
            roof_q = cls_prob[..., :-1].sum(dim=-1)
        roof_q = roof_q.clamp(min=0.0, max=1.0)
        mask_prob = torch.sigmoid(mask_logits)
        weighted = roof_q[:, :, None, None] * mask_prob
        roof_prob = weighted.sum(dim=1) / roof_q.sum(dim=1, keepdim=True).clamp(min=1e-6)[:, :, None]
        roof_prob = roof_prob.squeeze(1) if roof_prob.ndim == 4 else roof_prob
        roof_logits = torch.logit(roof_prob.clamp(min=1e-5, max=1 - 1e-5))

        teacher_list = []
        valid_list = []
        b, h, w = roof_logits.shape
        for i in range(b):
            sam_seg = getattr(data_samples[i], 'gt_sam_seg', None)
            sam_data = getattr(sam_seg, 'data', None) if sam_seg is not None else None
            if sam_data is None:
                teacher = torch.zeros((h, w), dtype=roof_logits.dtype, device=device)
                valid = torch.zeros((h, w), dtype=roof_logits.dtype, device=device)
            else:
                if not torch.is_tensor(sam_data):
                    sam_data = torch.as_tensor(sam_data, device=device)
                else:
                    sam_data = sam_data.to(device=device)
                if sam_data.ndim == 3:
                    teacher = (sam_data[0] > 0).float()
                else:
                    teacher = (sam_data > 0).float()
                if teacher.shape[-2:] != (h, w):
                    teacher = F.interpolate(
                        teacher.unsqueeze(0).unsqueeze(0),
                        size=(h, w),
                        mode='nearest').squeeze(0).squeeze(0)
                valid = torch.ones((h, w), dtype=roof_logits.dtype, device=device)
            teacher_list.append(teacher)
            valid_list.append(valid)

        teacher_batch = torch.stack(teacher_list, dim=0)
        valid_batch = torch.stack(valid_list, dim=0)
        if valid_batch.sum() <= 0:
            return roof_logits.sum() * 0.0
        loss = self.sam_distill_loss(roof_logits, teacher_batch, valid_mask=valid_batch)
        return loss * self.sam_distill_weight

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

        # D. Query-geometry prediction/loss
        geo_loss = all_cls_scores[0].sum() * 0.0
        if hasattr(self.decode_head, 'last_query_embeddings'):
            query_embeddings = self._normalize_query_embeddings(
                getattr(self.decode_head, 'last_query_embeddings', None),
                batch_size=len(data_samples),
                all_cls_scores=all_cls_scores,
            )
            if query_embeddings is not None:
                geo_preds = self.geometry_head(query_embeddings)  # [B, Q, 3]
                geo_loss = self._compute_geometry_loss_with_reused_matching(
                    geo_preds=geo_preds,
                    all_cls_scores=all_cls_scores,
                    all_mask_preds=all_mask_preds,
                    data_samples=data_samples,
                    device=inputs.device,
                )
        losses['loss_geometry'] = geo_loss

        # E. Dense normal branch (optional)
        # Skip branch entirely when both related loss weights are disabled.
        use_dense_branch = (
            self.dense_geometry_head is not None
            and (self.dense_geometry_loss_weight > 0.0 or self.piecewise_planar_loss_weight > 0.0)
        )
        if use_dense_branch:
            pred_dense_normals = self.dense_geometry_head(
                x, output_size=(int(inputs.shape[-2]), int(inputs.shape[-1])))
            losses['loss_dense_normal'] = self._compute_dense_geometry_loss(
                pred_dense_normals=pred_dense_normals,
                data_samples=data_samples,
                device=inputs.device,
            )
            losses['loss_piecewise_planar'] = self._compute_piecewise_planar_regularization(
                pred_dense_normals=pred_dense_normals,
                data_samples=data_samples,
                device=inputs.device,
            )

        # F. Edge branch (optional)
        use_edge_branch = self.edge_head is not None and self.edge_loss_weight > 0.0
        if use_edge_branch:
            edge_logits = self.edge_head(
                x, output_size=(int(inputs.shape[-2]), int(inputs.shape[-1])))
            losses['loss_edge'] = self._compute_edge_loss(
                edge_logits=edge_logits,
                data_samples=data_samples,
                device=inputs.device,
            )

        # G. SAM distillation (optional)
        if self.sam_distill_weight > 0.0 and self.sam_distill_loss is not None:
            losses['loss_sam_distill'] = self._compute_sam_distill_loss(
                all_cls_scores=all_cls_scores,
                all_mask_preds=all_mask_preds,
                data_samples=data_samples,
                device=inputs.device,
            )

        # H. Topology regularization on matched foreground masks
        if self.topology_loss_weight > 0.0:
            losses['loss_topology'] = self._compute_topology_regularization(
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
        need_aux_maps = (self.dense_geometry_head is not None) or (self.edge_head is not None)
        aux_dense = None
        aux_edge = None
        decode_feats = None

        # Reset cache
        if hasattr(self.decode_head, 'last_query_embeddings'):
            self.decode_head.last_query_embeddings = None
        if hasattr(self.decode_head, 'last_cls_scores'):
            self.decode_head.last_cls_scores = None

        if need_aux_maps:
            with torch.no_grad():
                aux_feats = self.extract_feat(inputs)
                decode_feats = aux_feats
                if self.dense_geometry_head is not None:
                    aux_dense = self.dense_geometry_head(
                        aux_feats,
                        output_size=(int(inputs.shape[-2]), int(inputs.shape[-1])))
                if self.edge_head is not None:
                    aux_edge = torch.sigmoid(self.edge_head(
                        aux_feats,
                        output_size=(int(inputs.shape[-2]), int(inputs.shape[-1]))))

        results = super().predict(inputs, data_samples)

        # If runtime returned semantic-only output, recover query-based instances
        # directly from decode head logits/masks instead of semantic CC fallback.
        missing_indices: List[int] = []
        for i, sample in enumerate(results):
            pred_instances = getattr(sample, 'pred_instances', None)
            has_instances = pred_instances is not None and len(pred_instances) > 0
            if not has_instances:
                missing_indices.append(i)

        if missing_indices:
            with torch.no_grad():
                if decode_feats is None:
                    decode_feats = self.extract_feat(inputs)
                all_cls_scores, all_mask_preds = self.decode_head(decode_feats, data_samples)

            out_hw = (int(inputs.shape[-2]), int(inputs.shape[-1]))
            num_classes = int(getattr(self.decode_head, 'num_classes', 3))
            for i in missing_indices:
                recovered = self._instances_from_queries(
                    all_cls_scores=all_cls_scores,
                    all_mask_preds=all_mask_preds,
                    sample_index=i,
                    out_hw=out_hw,
                    num_classes=num_classes,
                    score_thr=0.05,
                    min_area=64,
                    max_instances=300,
                )
                if len(recovered) > 0:
                    results[i].pred_instances = recovered
                    continue
                # Last-resort fallback (kept low-confidence by design).
                pred_sem = getattr(results[i], 'pred_sem_seg', None)
                sem_data = getattr(pred_sem, 'data', None) if pred_sem is not None else None
                if torch.is_tensor(sem_data):
                    results[i].pred_instances = self._instances_from_semantic(sem_data)

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
                    query_indices = getattr(insts, 'query_indices', None)
                    if torch.is_tensor(query_indices) and query_indices.numel() > 0:
                        if pred_normals.shape[0] == 0:
                            insts.normals = torch.zeros(
                                (len(insts), 3),
                                dtype=geo_preds.dtype,
                                device=geo_preds.device)
                            continue
                        idx = query_indices.to(device=pred_normals.device, dtype=torch.long)
                        idx = torch.clamp(idx, min=0, max=max(int(pred_normals.shape[0]) - 1, 0))
                        insts.normals = pred_normals[idx]
                        continue
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

        if aux_dense is not None or aux_edge is not None:
            try:
                from mmengine.structures import PixelData
            except Exception:
                PixelData = None
            for i in range(len(results)):
                if aux_dense is not None:
                    dense_i = aux_dense[i]
                    if PixelData is not None:
                        results[i].pred_dense_normals = PixelData(data=dense_i)
                    else:
                        results[i].pred_dense_normals = dense_i
                if aux_edge is not None:
                    edge_i = aux_edge[i]
                    if PixelData is not None:
                        results[i].pred_edge_map = PixelData(data=edge_i)
                    else:
                        results[i].pred_edge_map = edge_i

        return results
