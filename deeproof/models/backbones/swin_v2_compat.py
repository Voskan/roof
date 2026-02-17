import inspect

from mmseg.registry import MODELS


def _build_swin_v2_alias():
    """Register a SwinTransformerV2 compatibility alias when needed."""
    if MODELS.get('SwinTransformerV2') is not None:
        # Native implementation already exists in this environment.
        return

    try:
        from mmseg.models.backbones import SwinTransformer as _SwinTransformer
    except Exception as exc:
        raise ImportError(
            'SwinTransformerV2 is unavailable and fallback SwinTransformer '
            'could not be imported from mmseg.models.backbones.'
        ) from exc

    @MODELS.register_module(name='SwinTransformerV2')
    class SwinTransformerV2(_SwinTransformer):
        """Compatibility wrapper for configs written against Swin V2 naming.

        Some MMSeg versions expose only ``SwinTransformer``. This adapter
        preserves old config names while dropping V2-only kwargs that are not
        supported by the base implementation.
        """

        def __init__(
            self,
            pretrained_window_sizes=None,
            convert_weights=False,
            **kwargs,
        ):
            # Kept for config compatibility/documentation parity.
            self.pretrained_window_sizes = pretrained_window_sizes
            self.convert_weights = convert_weights

            sig = inspect.signature(_SwinTransformer.__init__)
            accepts_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if not accepts_var_kwargs:
                valid_keys = set(sig.parameters.keys()) - {'self'}
                kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

            super().__init__(**kwargs)


_build_swin_v2_alias()
