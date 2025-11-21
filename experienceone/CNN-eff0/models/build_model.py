from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


def build_efficientnet_b0(num_classes: int, pretrained: bool = True, drop_rate: float = 0.2):
    try:
        if pretrained:
            w = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            net = models.efficientnet_b0(weights=w)
        else:
            net = models.efficientnet_b0(weights=None)
    except Exception:
        # fallback for older/newer torchvision api
        net = models.efficientnet_b0(pretrained=pretrained)

    net.classifier[0] = nn.Dropout(p=drop_rate, inplace=True)
    in_f = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_f, num_classes)
    return net


def build_model(arch: str, num_classes: int, pretrained: bool, drop_rate: float):
    arch = (arch or '').lower()
    if arch == 'efficientnet_b0' or arch == 'efficientnet-b0' or arch == 'effb0':
        return build_efficientnet_b0(num_classes, pretrained, drop_rate)
    raise ValueError(f"Unsupported arch: {arch}")
