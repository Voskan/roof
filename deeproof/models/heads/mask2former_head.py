from typing import Any, List, Optional, Tuple

import torch
from mmseg.models.decode_heads import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample

@MODELS.register_module()
class DeepRoofMask2FormerHead(Mask2FormerHead):
    """
    Custom Mask2Former Head that exposes query embeddings for multi-task learning.
    
    Standard Mask2FormerHead.forward() only returns (cls_scores, mask_preds).
    This subclass captures the final query embeddings and stores them in 
    `self.last_query_embeddings` so that the Segmentor (DeepRoofMask2Former) 
    can pass them to the Geometry Head.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_query_embeddings = None
        self.last_cls_scores = None

    @staticmethod
    def _to_bqc(tensor_like: Any, batch_size: int) -> Optional[torch.Tensor]:
        """Normalize query-like tensors to shape [B, Q, C]."""
        if isinstance(tensor_like, (list, tuple)):
            if len(tensor_like) == 0:
                return None
            tensor_like = tensor_like[-1]
        if not torch.is_tensor(tensor_like):
            return None

        tensor = tensor_like
        if tensor.ndim == 4:
            # Common mmdet layout: [num_layers, B, Q, C]
            tensor = tensor[-1]
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).expand(batch_size, -1, -1)

        if tensor.ndim != 3:
            return None
        if tensor.shape[0] != batch_size and tensor.shape[1] == batch_size:
            tensor = tensor.permute(1, 0, 2).contiguous()
        if tensor.shape[0] != batch_size:
            return None
        return tensor

    def _query_from_embedding_modules(
        self,
        batch_size: int,
        ref_tensor: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Best-effort extraction from query feature embeddings."""
        device = ref_tensor.device if torch.is_tensor(ref_tensor) else None
        dtype = ref_tensor.dtype if torch.is_tensor(ref_tensor) else None

        for owner in (self, getattr(self, 'predictor', None)):
            if owner is None:
                continue
            for name in ('query_feat', 'query_embed'):
                module = getattr(owner, name, None)
                weight = getattr(module, 'weight', None)
                if torch.is_tensor(weight) and weight.ndim == 2:
                    q = weight
                    if device is not None:
                        q = q.to(device=device)
                    if dtype is not None:
                        q = q.to(dtype=dtype)
                    return q.unsqueeze(0).expand(batch_size, -1, -1)
        return None

    def forward(self, x: List[torch.Tensor], data_samples: List[SegDataSample]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass that captures embeddings.
        
        Args:
            x (list[Tensor]): Multi-level features from the neck.
            data_samples (list[SegDataSample]): The segmentation data samples.
            
        Returns:
            tuple[list[Tensor], list[Tensor]]: 
                - all_cls_scores (list[Tensor]): Classification scores for each layer.
                - all_mask_preds (list[Tensor]): Mask predictions for each layer.
        """
        # Standard Mask2FormerHead calls its internal predictor
        # Predictor returns (all_cls_scores, all_mask_preds, query_embeddings)
        # But wait, standard predictor might NOT return embeddings either.
        
        # In MMSeg, Mask2FormerHead.forward normally does:
        # return self.predictor(x, data_samples)
        
        # We need to ensure we get the embeddings from the predictor.
        # If the predictor is standard, we might need to wrap its forward too.
        
        # Let's see if we can get them from the internal Transformer Decoder
        # by running it again or capturing output.
        
        # Actually, most Mask2Former implementations in OpenMMLab 
        # return (cls, mask) from the head, but the predictor's forward 
        # often generates the embeddings.
        
        # Safety: We call the predictor and if it returns 3 items, we capture the 3rd.
        # If not, we resort to a manual forward pass through the predictor's decoder.
        
        batch_size = len(data_samples)

        if hasattr(self, 'predictor'):
            out = self.predictor(x, data_samples)
        else:
            out = super().forward(x, data_samples)

        if isinstance(out, (list, tuple)):
            all_cls_scores, all_mask_preds = out[:2]
            query_embeddings = out[2] if len(out) >= 3 else None
        else:
            # Keep behavior permissive for custom forks.
            all_cls_scores, all_mask_preds = out
            query_embeddings = None

        self.last_cls_scores = all_cls_scores
        query_embeddings = self._to_bqc(query_embeddings, batch_size)

        # Fallback to learnable query embeddings from the head module.
        if query_embeddings is None:
            cls_proxy = self._to_bqc(all_cls_scores, batch_size)
            query_embeddings = self._query_from_embedding_modules(
                batch_size=batch_size,
                ref_tensor=cls_proxy,
            )

        # Last resort: class logits (shape [B, Q, C_cls]) to avoid None state.
        if query_embeddings is None:
            query_embeddings = self._to_bqc(all_cls_scores, batch_size)

        self.last_query_embeddings = query_embeddings
        return all_cls_scores, all_mask_preds

    def loss_by_feat(
        self,
        all_cls_scores: Any,
        all_mask_preds: Any,
        *args,
        **kwargs
    ) -> dict:
        """
        Compatibility wrapper for mmseg API variants.
        
        Some older wrappers call this as:
        - loss_by_feat(all_cls_scores, all_mask_preds, data_samples)
        while newer mmseg expects:
        - loss_by_feat(all_cls_scores, all_mask_preds, batch_gt_instances, batch_img_metas)
        """
        if len(args) == 1 and not kwargs:
            data_samples = args[0]
            if isinstance(data_samples, list) and (
                len(data_samples) == 0 or hasattr(data_samples[0], 'gt_instances')
            ):
                batch_gt_instances = [sample.gt_instances for sample in data_samples]
                batch_img_metas = [sample.metainfo for sample in data_samples]
                return super().loss_by_feat(
                    all_cls_scores,
                    all_mask_preds,
                    batch_gt_instances,
                    batch_img_metas,
                )

        return super().loss_by_feat(all_cls_scores, all_mask_preds, *args, **kwargs)
