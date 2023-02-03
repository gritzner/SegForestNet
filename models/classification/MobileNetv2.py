# https://arxiv.org/abs/1801.04381
import core
import models
import torch.nn as nn


class MobileNetv2(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        
        self.backbone = core.create_object(models.backbones, config.backbone, **params.__dict__)
        self.model_name ("MobileNetv2", self.backbone.model_name)
        self.classifier = nn.Sequential(
            nn.Conv2d(self.backbone.num_features[-1], 1280, 1),
            self.backbone.normalization(1280),
            self.backbone.activation(1280),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, params.num_classes, 1),
            nn.Flatten(),
        ).to(core.device)
    
    def forward(self, x):
        return self.classifier(self.backbone(x)[-1])
