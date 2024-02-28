# https://arxiv.org/abs/2011.09766
# https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Foreground-Aware_Relation_Network_for_Geospatial_Object_Segmentation_in_High_Spatial_CVPR_2020_paper.pdf
import core
import torch
import torch.nn as nn
import functools
from .FCN import FCNEncoder


class FarSeg(nn.Module):
    def __init__(self, config, params):
        super().__init__()

        if params.input_shape[1] != params.input_shape[2]:
            raise RuntimeError("input shape for FarSeg must be square")
        config.num_features = getattr(config, "num_features", 256)
    
        self.encoder = FCNEncoder(config, params)
        self.model_name = "FarSeg"
        
        for i, input_features in enumerate(self.encoder.num_features[-4:]):
            setattr(self, f"p{i}a", nn.Conv2d(input_features, config.num_features, 1).to(core.device))
            setattr(
                self,
                f"p{i}b",
                nn.Sequential(
                    self.encoder.padding(1),
                    nn.Conv2d(config.num_features, config.num_features, 3),
                ).to(core.device)
            )
            setattr(
                self,
                f"p{i}c",
                nn.Sequential(
                    nn.Conv2d(config.num_features, config.num_features, 1),
                    self.encoder.normalization(config.num_features),
                    self.encoder.activation(config.num_features)
                ).to(core.device)
            )
            setattr(
                self,
                f"p{i}d",
                nn.Sequential(
                    nn.Conv2d(config.num_features, config.num_features, 1),
                    self.encoder.normalization(config.num_features),
                    self.encoder.activation(config.num_features)
                ).to(core.device)
            )
            decoder = [
                self.encoder.padding(1),
                nn.Conv2d(config.num_features, config.num_features, 3),
                self.encoder.normalization(config.num_features),
                self.encoder.activation(config.num_features)
            ]
            for j in range(i):
                decoder.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
                if j+1 < i:
                    decoder.extend([
                        self.encoder.padding(1),
                        nn.Conv2d(config.num_features, config.num_features, 3),
                        self.encoder.normalization(config.num_features),
                        self.encoder.activation(config.num_features)
                    ])
            setattr(self, f"p{i}e", nn.Sequential(*decoder).to(core.device))
            
        self.scene_embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.encoder.num_features[-1], config.num_features, 1)
        ).to(core.device)
        
        self.classifier = nn.Conv2d(config.num_features, params.num_classes, 1).to(core.device)
        
    def forward(self, x, yt, ce_loss_func, weight, ignore_index):
        ys = self.encoder(x)[-4:]        
        scene_embedding = self.scene_embedding(ys[-1])
        
        ys = [getattr(self,f"p{i}a")(y) for i, y in enumerate(ys)]
        ys = [y+nn.functional.interpolate(ys[i+1],scale_factor=2,mode="nearest") if i<3 else y for i, y in enumerate(ys)]
        ys = [getattr(self,f"p{i}b")(y) for i, y in enumerate(ys)]
        
        r = [getattr(self,f"p{i}c")(y)*scene_embedding for i, y in enumerate(ys)]
        r = [torch.sigmoid(r.sum(1).unsqueeze(1)) for r in r]
        ys = [getattr(self,f"p{i}d")(y)*r[i] for i, y in enumerate(ys)]
        
        ys = [getattr(self,f"p{i}e")(y) for i, y in enumerate(ys)]
        y = functools.reduce(lambda a,b: a+b, ys)
        
        y = self.classifier(y)
        y = nn.functional.interpolate(y, scale_factor=4, mode="bilinear", align_corners=False)
        
        return y, ce_loss_func(y, yt, weight=weight, ignore_index=ignore_index)
    