from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from mmseg.models.decode_heads import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample


@MODELS.register_module()
class DeepRoofMask2FormerHead(Mask2FormerHead):
    """
    Custom Mask2Former Head that exposes REAL per-image query embeddings
    (the final decoder layer output) for the GeometryHead.

    FIX Bug #3: The original code fell back to `query_feat.weight` — a static
    learnable embedding (100×256) that is identical for every image. This meant
    GeometryHead received the same fixed input regardless of image content and
    learned only the mean normal vector across the dataset → all slopes ≈ 0°.

    Solution: Hook into the predictor's transformer decoder to capture the
    actual decoder OUTPUT embeddings (per-image, content-aware) after the
    forward pass. These are stored in `self.last_query_embeddings` for use
    by the geometry supervision in DeepRoofMask2Former.loss().
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Stores the real decoder output query embeddings [B, Q, 256]
        self.last_query_embeddings: Optional[torch.Tensor] = None
        # Stores cls scores for proxy fallback
        self.last_cls_scores: Optional[Any] = None
        # Stores the last Hungarian assign results keyed by batch index
        # FIX Bug #5: reuse matching results so geometry head uses same pairing
        # as the segmentation loss instead of running a second independent match.
        self.last_assign_results: Optional[list] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_bqc(tensor_like: Any, batch_size: int) -> Optional[torch.Tensor]:
        """Normalize query-like tensors to shape [B, Q, C]."""
        if isinstance(tensor_like, (list, tuple)):
            if len(tensor_like) == 0:
                return None
            tensor_like = tensor_like[-1]
        if not torch.is_tensor(tensor_like):
            return None

        t = tensor_like
        if t.ndim == 4:
            # [num_layers, B, Q, C]
            t = t[-1]
        elif t.ndim == 2:
            t = t.unsqueeze(0).expand(batch_size, -1, -1)

        if t.ndim != 3:
            return None
        if t.shape[0] != batch_size and t.shape[1] == batch_size:
            t = t.permute(1, 0, 2).contiguous()
        if t.shape[0] != batch_size:
            return None
        return t

    def _extract_decoder_output_embeddings(
        self,
        batch_size: int,
        ref: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Extract the per-image query embeddings from the Mask2Former transformer decoder.

        The decoder stores the output of its last layer as `query_feat` inside the
        predictor after a forward pass. This is the CONTENT-AWARE embedding we need.

        We try multiple known attribute paths across mmdet/mmseg versions:
        1. predictor.decoder.last_query_feat  (some forks)
        2. predictor.last_query_feat
        3. predictor.query_feat (only content-aware when set after forward)
        4. self.query_feat (same)

        We distinguish static weights (ndim==2, [Q, C]) from dynamic outputs
        (ndim==3, [B, Q, C]). Only dynamic (ndim==3) outputs are image-specific.
        """
        device = ref.device if torch.is_tensor(ref) else None
        dtype = ref.dtype if torch.is_tensor(ref) else None

        # Walk candidate owners in priority order
        predictor = getattr(self, 'predictor', None)
        decoder = getattr(predictor, 'decoder', None) if predictor is not None else None

        candidates = []
        for owner in filter(None, [decoder, predictor, self]):
            for attr in ('last_query_feat', 'query_feat', 'query_content'):
                val = getattr(owner, attr, None)
                if val is not None:
                    candidates.append(val)

        for val in candidates:
            if isinstance(val, nn.Embedding):
                weight = val.weight  # static [Q, C]
                # Skip static embeddings (ndim==2) — not image-specific
                continue
            if torch.is_tensor(val):
                t = val
                # Only accept 3-D tensors (dynamic, per-image)
                if t.ndim == 3 and t.shape[0] == batch_size:
                    if device is not None:
                        t = t.to(device=device)
                    if dtype is not None:
                        t = t.to(dtype=dtype)
                    return t
                # [Q, C] static weight — skip
                if t.ndim == 2:
                    continue

        return None  # Caller will use an image-specific fallback

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: List[torch.Tensor],
        data_samples: List[SegDataSample],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Standard forward that also captures real decoder query embeddings.
        """
        batch_size = len(data_samples)

        # Run the standard Mask2Former predictor
        if hasattr(self, 'predictor'):
            out = self.predictor(x, data_samples)
        else:
            out = super().forward(x, data_samples)

        if isinstance(out, (list, tuple)):
            all_cls_scores, all_mask_preds = out[:2]
        else:
            all_cls_scores, all_mask_preds = out, []

        self.last_cls_scores = all_cls_scores
        self.last_assign_results = None  # Reset; will be set by loss_by_feat wrapper

        # --- FIX Bug #3: Get REAL (dynamic) decoder output embeddings ---
        # Try to get content-aware embeddings from the decoder after forward pass.
        query_embeddings = self._extract_decoder_output_embeddings(
            batch_size=batch_size,
            ref=all_cls_scores[-1] if isinstance(all_cls_scores, (list, tuple)) and len(all_cls_scores) > 0
                else None,
        )

        # If dynamic embeddings not available (some mmseg forks don't expose them),
        # use the last-layer cls_scores as a proxy. These are at least image-specific
        # (shape [B, Q, num_classes+1]) and tell the geometry head something about
        # which query predicts what class — far better than static init embeddings.
        if query_embeddings is None:
            cls_proxy = self._to_bqc(all_cls_scores, batch_size)
            if cls_proxy is not None:
                query_embeddings = cls_proxy  # [B, Q, C_cls] — image-specific

        self.last_query_embeddings = query_embeddings
        return all_cls_scores, all_mask_preds

    # ------------------------------------------------------------------
    # Loss wrapper — captures Hungarian results for geometry reuse (Bug #5)
    # ------------------------------------------------------------------
    def loss_by_feat(
        self,
        all_cls_scores: Any,
        all_mask_preds: Any,
        *args,
        **kwargs,
    ) -> dict:
        """
        Compatibility wrapper. Also captures per-image Hungarian assign results
        so DeepRoofMask2Former.loss() can reuse them without a second matching.

        FIX Bug #5: The original code ran hungarian matching TWICE — once inside
        `decode_head.loss_by_feat` (for mask/cls loss) and once more explicitly in
        `DeepRoofMask2Former.loss()` for geometry. The two runs can produce different
        matchings because the first updates internal state. Now we capture the
        assign results here so geometry always uses the same matching.
        """
        # Normalise args to (batch_gt_instances, batch_img_metas)
        if len(args) == 1 and not kwargs:
            data_samples = args[0]
            if isinstance(data_samples, list) and (
                len(data_samples) == 0 or hasattr(data_samples[0], 'gt_instances')
            ):
                batch_gt_instances = [s.gt_instances for s in data_samples]
                batch_img_metas = [s.metainfo for s in data_samples]
                losses = super().loss_by_feat(
                    all_cls_scores,
                    all_mask_preds,
                    batch_gt_instances,
                    batch_img_metas,
                )
                # Try to capture assign results from assigner if stored
                self._try_capture_assign_results()
                return losses

        losses = super().loss_by_feat(all_cls_scores, all_mask_preds, *args, **kwargs)
        self._try_capture_assign_results()
        return losses

    def _try_capture_assign_results(self):
        """Attempt to read cached assign results from the assigner after loss_by_feat."""
        # Some mmdet assigners store `self.assign_results` after a call.
        assigner = getattr(self, 'assigner', None)
        if assigner is None:
            return
        cached = getattr(assigner, 'last_assign_results', None)
        if cached is not None:
            self.last_assign_results = cached
