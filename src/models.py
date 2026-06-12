from __future__ import annotations

import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet18_Weights, convnext_tiny, resnet18


def build_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_convnext_tiny(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def build_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    name = model_name.lower()
    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    if name == "convnext_tiny":
        return build_convnext_tiny(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model: {model_name}")
