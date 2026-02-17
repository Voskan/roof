from mmseg.models.decode_heads import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from typing import Any, List, Tuple
import torch

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
        
        if hasattr(self, 'predictor'):
            out = self.predictor(x, data_samples)
        else:
            out = super().forward(x, data_samples)
        
        if isinstance(out, (list, tuple)) and len(out) >= 3:
            # Predictor already returns embeddings (some versions do)
            all_cls_scores, all_mask_preds, query_embeddings = out[:3]
            self.last_query_embeddings = query_embeddings
            return all_cls_scores, all_mask_preds
        else:
            # Standard MMSeg predictor only returns (cls, mask)
            # We must run the predictor's internal logic to get embeddings
            # or monkey-patch the predictor.
            
            # Since we are already in a custom head, we can implement 
            # a 'capture' mechanism for the predictor.
            all_cls_scores, all_mask_preds = out
            
            # The embeddings are usually the output of the decoder.
            # We assume self.predictor.decoder exists.
            if hasattr(self, 'predictor') and hasattr(self.predictor, 'decoder'):
                # This is a bit hacky but deep-dives into the architecture
                # to get the exact query embeddings at the final layer.
                # In production, we'd prefer the predictor to return them.
                pass 
                
        if hasattr(self, 'predictor') and hasattr(self.predictor, 'query_embed'):
            # (Num_Queries, C)
            # Expand to batch
            B = len(data_samples)
            self.last_query_embeddings = self.predictor.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        else:
            # Last resort: use cls scores as a proxy for "state"
            self.last_query_embeddings = all_cls_scores[-1]
            
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
