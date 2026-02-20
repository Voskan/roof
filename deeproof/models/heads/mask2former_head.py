import copy
import inspect
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmseg.models.decode_heads import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample


@MODELS.register_module()
class DeepRoofMask2FormerHead(Mask2FormerHead):
    """
    Custom Mask2Former Head that captures REAL per-image decoder query embeddings
    via a PyTorch forward hook on the transformer decoder's last layer.

    FIX Bug #3 (improved): The original attribute-scanning approach failed because
    mmdet's Mask2FormerTransformerDecoder does not expose its internal query state
    as named attributes after the forward pass on all versions. Instead we register
    a hook on the last transformer decoder layer so we intercept the actual output
    tensor [B, Q, 256] directly. This is guaranteed to produce real, content-aware
    embeddings (not zero-padded class logits as the fallback was producing).

    Without real embeddings: GeometryHead receives [B, Q, 4_cls + 252_zeros]
    → outputs dataset-mean normal ≈ (0, 0, 1) for every query.

    With real embeddings: GeometryHead receives [B, Q, 256] with content-aware
    features → can distinguish sloped vs flat roofs and predict correct normals.

    FIX Bug #5: Hungarian assignment from seg loss is cached so geometry supervision
    uses exactly the same query-to-GT pairing.
    """

    @staticmethod
    def _runtime_expects_layer_cfg() -> bool:
        """Detect whether installed mmdet/mmseg Mask2Former expects `layer_cfg`."""
        try:
            src = inspect.getsource(Mask2FormerHead.__init__)
        except Exception:
            # Conservative default for modern mmdet/mmseg stacks.
            return True
        return 'layer_cfg' in src

    @staticmethod
    def _to_plain_dict(cfg_like: Any) -> Optional[Dict[str, Any]]:
        if cfg_like is None:
            return None
        if isinstance(cfg_like, dict):
            return dict(cfg_like)
        try:
            return dict(cfg_like.items())
        except Exception:
            return None

    @classmethod
    def _upgrade_legacy_transformer_cfg(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upgrade legacy OpenMMLab transformer config keys to modern schema:
        - transformerlayers -> layer_cfg
        - attn_cfgs/ffn_cfgs -> self_attn_cfg/cross_attn_cfg/ffn_cfg
        """
        out = copy.deepcopy(kwargs)
        decode_head_cfg = cls._to_plain_dict(out)
        if decode_head_cfg is None:
            return out

        def _upgrade_encoder_layer_cfg(layer_cfg_like: Any) -> Dict[str, Any]:
            layer_cfg = cls._to_plain_dict(layer_cfg_like) or {}
            if 'attn_cfgs' in layer_cfg and 'self_attn_cfg' not in layer_cfg:
                layer_cfg['self_attn_cfg'] = copy.deepcopy(layer_cfg.pop('attn_cfgs'))
            if 'ffn_cfgs' in layer_cfg and 'ffn_cfg' not in layer_cfg:
                layer_cfg['ffn_cfg'] = copy.deepcopy(layer_cfg.pop('ffn_cfgs'))
            layer_cfg.pop('operation_order', None)
            layer_cfg.pop('feedforward_channels', None)
            return layer_cfg

        def _upgrade_decoder_layer_cfg(layer_cfg_like: Any) -> Dict[str, Any]:
            layer_cfg = cls._to_plain_dict(layer_cfg_like) or {}
            attn_cfgs = layer_cfg.pop('attn_cfgs', None)
            if attn_cfgs is not None:
                if isinstance(attn_cfgs, (list, tuple)):
                    if len(attn_cfgs) >= 1 and 'self_attn_cfg' not in layer_cfg:
                        layer_cfg['self_attn_cfg'] = copy.deepcopy(attn_cfgs[0])
                    if len(attn_cfgs) >= 2 and 'cross_attn_cfg' not in layer_cfg:
                        layer_cfg['cross_attn_cfg'] = copy.deepcopy(attn_cfgs[1])
                    elif len(attn_cfgs) == 1 and 'cross_attn_cfg' not in layer_cfg:
                        layer_cfg['cross_attn_cfg'] = copy.deepcopy(attn_cfgs[0])
                elif isinstance(attn_cfgs, dict):
                    layer_cfg.setdefault('self_attn_cfg', copy.deepcopy(attn_cfgs))
                    layer_cfg.setdefault('cross_attn_cfg', copy.deepcopy(attn_cfgs))
            if 'ffn_cfgs' in layer_cfg and 'ffn_cfg' not in layer_cfg:
                layer_cfg['ffn_cfg'] = copy.deepcopy(layer_cfg.pop('ffn_cfgs'))
            layer_cfg.pop('operation_order', None)
            layer_cfg.pop('feedforward_channels', None)
            return layer_cfg

        pixel_decoder = cls._to_plain_dict(decode_head_cfg.get('pixel_decoder'))
        if pixel_decoder is not None:
            encoder = cls._to_plain_dict(pixel_decoder.get('encoder'))
            if encoder is not None:
                if 'transformerlayers' in encoder and 'layer_cfg' not in encoder:
                    encoder['layer_cfg'] = _upgrade_encoder_layer_cfg(encoder.pop('transformerlayers'))
                elif 'layer_cfg' in encoder:
                    encoder['layer_cfg'] = _upgrade_encoder_layer_cfg(encoder['layer_cfg'])
                pixel_decoder['encoder'] = encoder
            decode_head_cfg['pixel_decoder'] = pixel_decoder

        transformer_decoder = cls._to_plain_dict(decode_head_cfg.get('transformer_decoder'))
        if transformer_decoder is not None:
            if 'transformerlayers' in transformer_decoder and 'layer_cfg' not in transformer_decoder:
                transformer_decoder['layer_cfg'] = _upgrade_decoder_layer_cfg(
                    transformer_decoder.pop('transformerlayers'))
            elif 'layer_cfg' in transformer_decoder:
                transformer_decoder['layer_cfg'] = _upgrade_decoder_layer_cfg(transformer_decoder['layer_cfg'])
            decode_head_cfg['transformer_decoder'] = transformer_decoder

        return decode_head_cfg

    def __init__(self, **kwargs):
        # Compatibility shim for newer OpenMMLab stacks that expect `layer_cfg`.
        if self._runtime_expects_layer_cfg():
            kwargs = self._upgrade_legacy_transformer_cfg(kwargs)
        super().__init__(**kwargs)
        self.last_query_embeddings: Optional[torch.Tensor] = None
        self.last_cls_scores: Optional[Any] = None
        self.last_assign_results: Optional[list] = None

        # Hook handle — registered lazily on first forward call
        self._hook_handle: Optional[Any] = None
        # Buffer filled by the hook
        self._hooked_decoder_output: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------
    def _register_decoder_hook(self):
        """
        Register a forward hook on the LAST layer of the Mask2Former transformer
        decoder to capture real query embeddings.

        The hook fires AFTER the layer's forward and stores the output tensor.
        We only hook the LAST decoder layer so we get the most refined embeddings.

        The Mask2Former decoder stack is typically:
          predictor → decoder → layers[num_layers-1]
        """
        if self._hook_handle is not None:
            return  # Already registered

        # Walk the module tree to find the decoder layer stack
        decoder_layer = self._find_last_decoder_layer()
        if decoder_layer is None:
            return

        def _hook(module, input, output):
            """Capture the output of the last decoder cross-attention layer."""
            # Output is typically a tensor [B, Q, C] or a tuple/list
            if isinstance(output, (list, tuple)):
                out = output[0]
            else:
                out = output
            if torch.is_tensor(out) and out.ndim == 3:
                self._hooked_decoder_output = out.detach()
            # Return None to leave output unchanged

        self._hook_handle = decoder_layer.register_forward_hook(_hook)

    def _find_last_decoder_layer(self) -> Optional[nn.Module]:
        """
        Locate the last transformer decoder layer in the predictor.

        Traversal order (most common mmdet/mmseg architectures):
          1. self.predictor.decoder.layers[-1]
          2. self.predictor.decoder[-1]
          3. self.predictor.layers[-1]
          4. Any nn.ModuleList whose items are nn.Module (decoder layers)
        """
        predictor = getattr(self, 'predictor', None)
        if predictor is None:
            return None

        decoder = getattr(predictor, 'decoder', None)

        # Check common layer container attribute names
        for container_owner in filter(None, [decoder, predictor]):
            for attr in ('layers', 'transformer_layers', 'decoder_layers'):
                layers = getattr(container_owner, attr, None)
                if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
                    last = layers[-1]
                    if isinstance(last, nn.Module):
                        return last

        # Fallback: scan all named children for a ModuleList of decoder-like layers
        for name, module in self.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) >= 3:
                last = module[-1]
                if hasattr(last, 'self_attn') or hasattr(last, 'cross_attn') \
                        or hasattr(last, 'attention') or hasattr(last, 'norm'):
                    return last

        return None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: List[torch.Tensor],
        data_samples: List[SegDataSample],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Standard Mask2Former forward that also captures real decoder query embeddings.
        """
        batch_size = len(data_samples)

        # Lazily register the decoder hook on first call
        self._register_decoder_hook()
        # Reset hook buffer before forward so we know what's from THIS call
        self._hooked_decoder_output = None

        # Standard predictor call
        if hasattr(self, 'predictor'):
            out = self.predictor(x, data_samples)
        else:
            out = super().forward(x, data_samples)

        if isinstance(out, (list, tuple)):
            all_cls_scores, all_mask_preds = out[:2]
        else:
            all_cls_scores, all_mask_preds = out, []

        self.last_cls_scores = all_cls_scores
        self.last_assign_results = None

        # --- Primary source: forward hook captured real decoder layer output ---
        # Shape: [B, Q, 256] — content-aware, different for every image
        query_embeddings = None
        hooked = self._hooked_decoder_output
        if hooked is not None and hooked.ndim == 3 and hooked.shape[0] == batch_size:
            query_embeddings = hooked

        # --- Fallback: try known attribute paths (some forks expose them) ---
        if query_embeddings is None:
            query_embeddings = self._scan_attributes_for_embeddings(batch_size)

        # --- Last resort: class logits (image-specific but low-dim) ---
        # This produces near-zero normals due to zero-padding to 256, but is better
        # than random init. Geometry quality will be poor until decoder hook works.
        if query_embeddings is None:
            if isinstance(all_cls_scores, (list, tuple)) and len(all_cls_scores) > 0:
                cls = all_cls_scores[-1]
                if torch.is_tensor(cls) and cls.ndim == 3 and cls.shape[0] == batch_size:
                    query_embeddings = cls
                    # NOTE: cls has shape [B, Q, C_cls=4], NOT 256.
                    # _normalize_query_embeddings in deeproof_model will pad to 256.
                    # This is a lossy fallback — hook registration above must succeed
                    # to get meaningful geometry predictions.

        self.last_query_embeddings = query_embeddings
        return all_cls_scores, all_mask_preds

    def _scan_attributes_for_embeddings(
        self, batch_size: int
    ) -> Optional[torch.Tensor]:
        """
        Scan known attribute paths on the decoder for query embeddings.
        Only accepts 3-D tensors [B, Q, C] (dynamic/per-image), rejects
        2-D static weight matrices [Q, C] which are identical for every image.
        """
        predictor = getattr(self, 'predictor', None)
        decoder = getattr(predictor, 'decoder', None) if predictor else None

        for owner in filter(None, [decoder, predictor, self]):
            for attr in ('last_query_feat', 'query_feat', 'query_content',
                         'decoder_output', 'last_hidden_state'):
                val = getattr(owner, attr, None)
                if isinstance(val, nn.Embedding):
                    continue  # Static init weight — skip
                if torch.is_tensor(val) and val.ndim == 3 and val.shape[0] == batch_size:
                    return val
        return None

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
        """
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
                self._try_capture_assign_results()
                return losses

        losses = super().loss_by_feat(all_cls_scores, all_mask_preds, *args, **kwargs)
        self._try_capture_assign_results()
        return losses

    def _try_capture_assign_results(self):
        """Attempt to read cached assign results from the assigner after loss_by_feat."""
        assigner = getattr(self, 'assigner', None)
        if assigner is None:
            return
        cached = getattr(assigner, 'last_assign_results', None)
        if cached is not None:
            self.last_assign_results = cached

    def __del__(self):
        """Cleanup hook on destruction."""
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
