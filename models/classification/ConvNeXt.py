# https://arxiv.org/abs/2201.03545
import core
import models
import torch.nn as nn


class ConvNeXt(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        
        self.backbone = core.create_object(models.backbones, config.backbone, **params.__dict__)
        self.model_name ("ConvNeXt", self.backbone.model_name)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            self.backbone.normalization(self.backbone.num_features[-1]),
            nn.Conv2d(self.backbone.num_features[-1], params.num_classes, 1),
            nn.Flatten(),
        ).to(core.device)

    def forward(self, x, intermediate_results=False):
        return self.classifier(self.backbone(x)[-1])
