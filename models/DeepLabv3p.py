# https://arxiv.org/abs/1802.02611
import core
import torch
import torch.nn as nn
import models
import types


class AtrousSpatialPyramidPooling(nn.ModuleList):
    def __init__(self, config, input_features, feature_space_size, backbone):
        super().__init__()
        # global average pooling
        self.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_features, 256, 1),
            backbone.normalization(256),
            backbone.activation(256),
            nn.Upsample(feature_space_size)
        ))
        # 1x1 conv
        self.append(nn.Sequential(
            nn.Conv2d(input_features, 256, 1),
            backbone.normalization(256),
            backbone.activation(256)
        ))
        # atrous conv
        for d in config.aspp_dilation_rates:
            self.append(nn.Sequential(
                backbone.padding(d),
                nn.Conv2d(input_features, input_features, 3, dilation=d, groups=input_features),
                nn.Conv2d(input_features, 256, 1),
                backbone.normalization(256),
                backbone.activation(256)
            ))
        
    def forward(self, x):
        results = []
        for layer in self:
            results.append(layer(x))
        return torch.cat(results, 1)
    
    
class DeepLabv3pEncoder(nn.Module):
    def __init__(self, config, params):
        super().__init__()
    
        self.backbone = models.XceptionFeatureExtractor(config, params.input_shape[0])
            
        self.skip_connection = nn.Sequential(
            nn.Conv2d(self.backbone.num_features[1], 48, 1),
            self.backbone.normalization(48),
            self.backbone.activation(48),
        ).to(core.device)
        
        self.aspp = nn.Sequential(
            AtrousSpatialPyramidPooling(config, self.backbone.num_features[4], params.input_shape[1] // 2**self.backbone.downsampling, self.backbone),
            nn.Conv2d(256*(2+len(config.aspp_dilation_rates)), 256, 1),
            self.backbone.normalization(256),
            self.backbone.activation(256),
        ).to(core.device)
        
    def forward(self, x):
        y = self.backbone(x)
        return self.skip_connection(y[1]), self.aspp(y[4])
    
            
class DeepLabv3p(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        if params.input_shape[1] != params.input_shape[2]:
            raise RuntimeError("input shape for DeepLabv3p must be square")
            
        self.encoder = DeepLabv3pEncoder(config, params)
        self.model_name = "DeepLabv3p"
        self.upsample = nn.Upsample(scale_factor=2**(self.encoder.backbone.downsampling-2), mode="bilinear", align_corners=False)
        
        self.final = nn.Sequential(
            self.encoder.backbone.padding(1),
            nn.Conv2d(304, 304, 3, groups=304),
            nn.Conv2d(304, 256, 1),
            self.encoder.backbone.normalization(256),
            self.encoder.backbone.activation(256),
            self.encoder.backbone.padding(1),
            nn.Conv2d(256, 256, 3, groups=256),
            nn.Conv2d(256, 256, 1),
            self.encoder.backbone.normalization(256),
            self.encoder.backbone.activation(256),
            nn.Conv2d(256, params.num_classes, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        ).to(core.device)
        
    def forward(self, x, yt, ce_loss_func, weight, ignore_index):
        yp = self.encoder(x)
        yp = torch.cat((yp[0], self.upsample(yp[1])), 1)
        yp = self.final(yp)
        return yp, ce_loss_func(yp, yt, weight=weight, ignore_index=ignore_index)
