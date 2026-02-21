import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class RoofEdgeHead(BaseModule):
    """
    Structural edge head for ridge/valley/eave proxy heatmap prediction.
    """

    def __init__(
        self,
        in_channels: int = 192,
        hidden_channels: int = 96,
        feat_index: int = 0,
        num_layers: int = 2,
        out_channels: int = 1,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.feat_index = int(feat_index)
        layers = []
        ch = int(in_channels)
        for _ in range(max(int(num_layers), 1)):
            layers.append(nn.Conv2d(ch, int(hidden_channels), kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(int(hidden_channels)))
            layers.append(nn.ReLU(inplace=True))
            ch = int(hidden_channels)
        self.stem = nn.Sequential(*layers)
        self.out_conv = nn.Conv2d(ch, int(out_channels), kernel_size=1)

    def forward(self, feats, output_size=None):
        if isinstance(feats, (list, tuple)):
            feat = feats[self.feat_index]
        else:
            feat = feats
        x = self.stem(feat)
        x = self.out_conv(x)
        if output_size is not None and tuple(x.shape[-2:]) != tuple(output_size):
            x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        return x
