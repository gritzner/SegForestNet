# https://arxiv.org/abs/1610.02357
import core
import models
import torch.nn as nn


class Xception(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        
        self.backbone = core.create_object(models.backbones, config.backbone, **params.__dict__)
        self.model_name ("Xception", self.backbone.model_name)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.backbone.num_features[-1], params.num_classes, 1),
            nn.Flatten(),
        ).to(core.device)
    
    def forward(self, x):
        return self.classifier(self.backbone(x)[-1])
