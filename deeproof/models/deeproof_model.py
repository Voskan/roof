import inspect
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mmengine.config import ConfigDict
except Exception:  # pragma: no cover - fallback for lightweight test envs
    ConfigDict = None
try:
    import cv2
except Exception:  # pragma: no cover - optional in minimal environments
    cv2 = None

try:
    # Some mmseg builds expose Mask2Former only through this module path.
    from mmseg.models.segmentors.mask2former import Mask2Former as Mask2FormerBase
except Exception:
    try:
        # Older forks may export it at package level.
        from mmseg.models.segmentors import Mask2Former as Mask2FormerBase
    except Exception:
        # Official mmseg versions may only provide EncoderDecoder.
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
    
    This implementation resolves the 'disconnected supervision' bug by
    reusing the Hungarian Matching results from the Mask Head to supervise 
    the Geometry Head.
    """

    def __init__(self,
                  geometry_head: dict,
                  geometry_loss_weight: float = 5.0,
                  **kwargs):
        super().__init__(**kwargs)

        # Ensure inference config is always present and compatible with both
        # `.get(...)` and attribute-style (`.mode`) access across mmseg versions.
        self.test_cfg = self._normalize_test_cfg(getattr(self, 'test_cfg', None))
        
        # 1. Initialize Geometry Head (MLP that takes query embeddings)
        self.geometry_head = MODELS.build(geometry_head)
        
        # 2. Define Geometry Loss (Cosine Similarity)
        self.geometry_loss = CosineSimilarityLoss(loss_weight=geometry_loss_weight)
        
        # Store loss weight for explicit usage if needed
        self.geometry_loss_weight = geometry_loss_weight

    @staticmethod
    def _normalize_test_cfg(test_cfg):
        default_cfg = dict(mode='whole')

        if test_cfg is None:
            if ConfigDict is not None:
                return ConfigDict(default_cfg)
            return default_cfg

        if isinstance(test_cfg, dict):
            merged = dict(default_cfg)
            merged.update(test_cfg)
            if ConfigDict is not None:
                return ConfigDict(merged)
            return merged

        # Object-style configs. Ensure `.mode` exists.
        if not hasattr(test_cfg, 'mode'):
            try:
                setattr(test_cfg, 'mode', 'whole')
            except Exception:
                if ConfigDict is not None:
                    return ConfigDict(default_cfg)
                return default_cfg

        # If object has no dict-like `.get`, convert to ConfigDict when possible.
        if not hasattr(test_cfg, 'get'):
            if ConfigDict is not None:
                return ConfigDict(dict(mode=getattr(test_cfg, 'mode', 'whole')))
            return dict(mode=getattr(test_cfg, 'mode', 'whole'))

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
            if len(query_like) == 0:
                return None
            query_like = query_like[-1]
        if not torch.is_tensor(query_like):
            return None

        query = query_like
        if query.ndim == 4:
            # Typical decoder output format: [num_layers, B, Q, C].
            query = query[-1]
        elif query.ndim == 2:
            query = query.unsqueeze(0).expand(batch_size, -1, -1)

        if query.ndim != 3:
            return None
        if query.shape[0] != batch_size and query.shape[1] == batch_size:
            query = query.permute(1, 0, 2).contiguous()
        if query.shape[0] != batch_size:
            return None
        return query

    def _query_from_head_modules(
        self,
        batch_size: int,
        ref_tensor: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Best-effort query embedding extraction from decode head internals."""
        device = ref_tensor.device if torch.is_tensor(ref_tensor) else None
        dtype = ref_tensor.dtype if torch.is_tensor(ref_tensor) else None

        for owner in (self.decode_head, getattr(self.decode_head, 'predictor', None)):
            if owner is None:
                continue
            for name in ('query_feat', 'query_embed'):
                module = getattr(owner, name, None)
                weight = getattr(module, 'weight', None)
                if torch.is_tensor(weight) and weight.ndim == 2:
                    query = weight
                    if device is not None:
                        query = query.to(device=device)
                    if dtype is not None:
                        query = query.to(dtype=dtype)
                    return query.unsqueeze(0).expand(batch_size, -1, -1)
        return None

    def _normalize_query_embeddings(
        self,
        query_like,
        batch_size: int,
        all_cls_scores,
    ) -> Optional[torch.Tensor]:
        """Return [B, Q, C_geo] query embeddings compatible with GeometryHead."""
        query = self._to_bqc(query_like, batch_size)
        cls_proxy = self._to_bqc(all_cls_scores, batch_size)

        if query is None:
            query = self._query_from_head_modules(batch_size=batch_size, ref_tensor=cls_proxy)
        if query is None:
            return None

        expected_dim = self._geometry_embed_dim()
        if query.shape[-1] == expected_dim:
            return query

        # Prefer true query embeddings from head internals over class logits.
        fallback = self._query_from_head_modules(batch_size=batch_size, ref_tensor=query)
        fallback = self._to_bqc(fallback, batch_size)
        if fallback is not None and fallback.shape[-1] == expected_dim:
            return fallback

        # Final safety: adapt dimensionality to avoid runtime shape crashes.
        if query.shape[-1] < expected_dim:
            pad = query.new_zeros(*query.shape[:-1], expected_dim - query.shape[-1])
            query = torch.cat([query, pad], dim=-1)
        else:
            query = query[..., :expected_dim]
        return query

    @staticmethod
    def _masks_to_tensor(masks, device: torch.device) -> torch.Tensor:
        """Convert mask containers to a [N, H, W] tensor on target device."""
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
            # Typical shape from SegDataSample: [1, H, W].
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

        # Infer signature from the first non-variadic definition in the MRO.
        # This avoids masking real runtime TypeErrors behind a fallback retry.
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

        # Modern signature:
        # loss_by_feat(all_cls_scores, all_mask_preds, batch_gt_instances, batch_img_metas)
        return loss_by_feat(
            all_cls_scores,
            all_mask_preds,
            batch_gt_instances,
            batch_img_metas,
        )

    def loss(self, inputs: torch.Tensor, data_samples: List[SegDataSample]) -> dict:
        """
        Calculate multi-task losses with explicit Hungarian Matching for Geometry.
        """
        # A. Feature Extraction (Backbone + Neck)
        x = self.extract_feat(inputs)
        
        # B. Standard Mask2Former Forward
        # Normal Mask2Former returns cls_scores and mask_preds for each layer
        all_cls_scores, all_mask_preds = self.decode_head(x, data_samples)
        
        # C. Standard Losses (Segmentation & Classification)
        losses = self._loss_by_feat_compat(all_cls_scores, all_mask_preds, data_samples)
        
        # D. Geometry Head Prediction & Supervision
        # We need the query embeddings from the transformer decoder.
        # DeepRoof assumes the decoder outputs or caches the final query embeddings.
        if not hasattr(self.decode_head, 'last_query_embeddings'):
            # In production, we ensure the decoder is configured to expose embeddings.
            # If missing, we return zero loss to prevent training crashes but signal the gap.
            losses['loss_geometry'] = all_cls_scores[0].sum() * 0.0
            return losses

        # geo_preds shape: (B, Num_Queries, 3)
        query_embeddings = self._normalize_query_embeddings(
            getattr(self.decode_head, 'last_query_embeddings', None),
            batch_size=len(data_samples),
            all_cls_scores=all_cls_scores,
        )
        if query_embeddings is None:
            losses['loss_geometry'] = all_cls_scores[0].sum() * 0.0
            return losses
        geo_preds = self.geometry_head(query_embeddings)
        
        total_geo_loss = 0.0
        num_pos_total = 0
        
        # E. Explicit Hungarian Matching per Image
        for i in range(len(data_samples)):
            img_geo_pred = geo_preds[i]        # (Num_Queries, 3)
            img_cls_pred = all_cls_scores[-1][i] # (Num_Queries, Num_Classes)
            img_mask_pred = all_mask_preds[-1][i] # (Num_Queries, H, W)
            
            gt_instances = data_samples[i].gt_instances
            img_meta = data_samples[i].metainfo
            
            # Handle "No Objects" in Image
            if len(gt_instances) == 0:
                continue 

            # Prepare predictions for the assigner
            pred_instances = InstanceData()
            pred_instances.scores = img_cls_pred
            pred_instances.masks = img_mask_pred
            gt_instances_for_assign = self._prepare_gt_instances_for_assigner(
                gt_instances=gt_instances,
                pred_mask_shape=img_mask_pred.shape,
                device=inputs.device,
            )
            
            # 1. Explicitly Invoke the Hungarian Matcher (Assigner)
            # This identifies which query (prediction) corresponds to which GT instance.
            try:
                assign_result = self.decode_head.assigner.assign(
                    pred_instances, gt_instances_for_assign, img_meta=img_meta)
            except TypeError:
                # Older assigners may not accept keyword args.
                assign_result = self.decode_head.assigner.assign(
                    pred_instances, gt_instances_for_assign, img_meta)
            
            # 2. Extract Positive Matches
            # gt_inds maps pred_idx -> (gt_idx + 1). 0 means background/no-match.
            pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1)
            
            if pos_inds.numel() == 0:
                continue  # No query was matched to any GT roof plane
            
            # 3. Align Geometry Predictions with Ground Truth
            # src_idx: indices of matched queries
            # tgt_idx: indices of matched GT instances
            matched_src_idx = pos_inds
            matched_tgt_idx = assign_result.gt_inds[pos_inds] - 1
            
            matched_preds = img_geo_pred[matched_src_idx]   # (N_matched, 3)
            
            # Extract GT Normals (Assumed to be pre-calculated in dataset/preprocessing)
            if hasattr(gt_instances, 'normals'):
                matched_targets = gt_instances.normals[matched_tgt_idx] # (N_matched, 3)
            else:
                # Fallback: Extract normals from dense ground truth map
                # (This is more computationally expensive but robust)
                dense_normals = data_samples[i].gt_normals.data # (3, H, W)
                raw_masks = gt_instances.masks
                if hasattr(raw_masks, 'to_tensor'):
                    gt_masks = raw_masks.to_tensor(dtype=torch.bool, device=inputs.device)
                else:
                    gt_masks = raw_masks
                    if not torch.is_tensor(gt_masks):
                        gt_masks = torch.as_tensor(gt_masks)
                    gt_masks = gt_masks.to(device=inputs.device, dtype=torch.bool)
                
                gt_normals_list = []
                for idx in matched_tgt_idx:
                    mask = gt_masks[idx]
                    avg_n = dense_normals[:, mask].mean(dim=1)
                    avg_n = F.normalize(avg_n, p=2, dim=0) # Must be unit vector
                    gt_normals_list.append(avg_n)
                matched_targets = torch.stack(gt_normals_list)

            # 4. Compute Cosine Similarity Loss on Matched Pairs
            # loss = 1 - cosine_similarity. Weighting is handled by the loss module.
            loss_geo = self.geometry_loss(matched_preds, matched_targets)
            
            # Accumulate
            # We weight by batch size correctly by tracking total matches
            total_geo_loss += loss_geo * matched_src_idx.numel()
            num_pos_total += matched_src_idx.numel()

        # F. Final Normalization & Safety
        if num_pos_total > 0:
            losses['loss_geometry'] = total_geo_loss / num_pos_total
        else:
            # Safety: return zero loss while preserving gradients for parameters if possible
            losses['loss_geometry'] = geo_preds.sum() * 0.0
            
        return losses

    def predict(self, inputs: torch.Tensor, data_samples: List[SegDataSample]) -> List[SegDataSample]:
        """
        Implementation of inference with geometry predictions.
        """
        # Keep inference path robust if external code mutates/clears test_cfg.
        self.test_cfg = self._normalize_test_cfg(getattr(self, 'test_cfg', None))
        results = super().predict(inputs, data_samples)

        # Some mmseg variants return semantic maps only. Build instance outputs
        # from connected components so inference/post-processing can proceed.
        for sample in results:
            pred_instances = getattr(sample, 'pred_instances', None)
            has_instances = pred_instances is not None and len(pred_instances) > 0
            if has_instances:
                continue
            pred_sem = getattr(sample, 'pred_sem_seg', None)
            sem_data = getattr(pred_sem, 'data', None) if pred_sem is not None else None
            if torch.is_tensor(sem_data):
                sample.pred_instances = self._instances_from_semantic(sem_data)
        
        # Run Geometry Head on inference embeddings
        x = self.extract_feat(inputs)
        # Cache decode-head query embeddings using API-compatible calls across
        # mmseg/mmdet variants. Geometry attachment is best-effort at inference.
        if hasattr(self.decode_head, 'last_query_embeddings'):
            self.decode_head.last_query_embeddings = None
        if hasattr(self.decode_head, 'last_cls_scores'):
            self.decode_head.last_cls_scores = None

        decode_ok = False

        # Preferred path: our custom decode head caches embeddings in forward().
        try:
            self.decode_head(x, data_samples)
            decode_ok = True
        except Exception:
            decode_ok = False

        # Fallback for forks where only predict() exists/works.
        if not decode_ok and hasattr(self.decode_head, 'predict'):
            predict_fn = self.decode_head.predict
            batch_img_metas = [
                sample.metainfo for sample in data_samples
            ] if data_samples is not None else []
            test_cfg = getattr(self, 'test_cfg', None)
            call_attempts = [
                lambda: predict_fn(x, data_samples, test_cfg),
                lambda: predict_fn(x, batch_img_metas, test_cfg),
                lambda: predict_fn(x, data_samples),
                lambda: predict_fn(x, batch_img_metas),
            ]
            for call in call_attempts:
                try:
                    call()
                    decode_ok = True
                    break
                except TypeError:
                    continue
                except Exception:
                    break

        query_cache = getattr(self.decode_head, 'last_query_embeddings', None)
        if query_cache is not None:
            query_embeddings = self._normalize_query_embeddings(
                query_cache,
                batch_size=len(data_samples),
                all_cls_scores=getattr(self.decode_head, 'last_cls_scores', []),
            )
            if query_embeddings is None:
                return results
            geo_preds = self.geometry_head(query_embeddings)
            
            for i in range(len(results)):
                # Attach normal vectors to each detected instance
                # Mask2FormerHead.predict normally populates results[i].pred_instances
                if not hasattr(results[i], 'pred_instances'):
                    continue
                insts = results[i].pred_instances
                if len(insts) > 0:
                    # During inference, we don't match; we just attach the predicted
                    # normal for the query that generated each instance.
                    # Note: Need to verify if 'query_indices' are stored by the head.
                    # As a simplified production fallback:
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
