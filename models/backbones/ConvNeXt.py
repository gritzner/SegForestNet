# https://arxiv.org/abs/2201.03545
import core
import torch
import torch.nn as nn
import utils


class ConvNeXtBlock(nn.Module):
    def __init__(self, features, padding, normalization):
        super().__init__()
        self.block = nn.Sequential(
            padding(3),
            nn.Conv2d(features, features, 7, groups=features),
            normalization(features),
            nn.Conv2d(features, 4*features, 1),
            nn.GELU(),
            nn.Conv2d(4*features, features, 1)
        )
        
    def forward(self, x):
        return x + self.block(x)


class ConvNeXt(nn.ModuleList):
    def __init__(self, config, params):
        super().__init__()
        
        assert 2 <= config.downsampling <= 5
        self.model_name = "ConvNeXt"
        self.downsampling = config.downsampling
        self.num_features = tuple([config.features[0], *config.features])
        self.activation = lambda _: nn.GELU()
        self.padding = getattr(nn, getattr(config, "padding", "ZeroPad2d"))
        self.normalization = utils.norm_wrapper("PixelNorm2d") # paper mentions LayerNormNd but the reference code effectively implements PixelNorm2d
        
        self.append(nn.Sequential(
            nn.Conv2d(params.input_shape[0], config.features[0], 4, stride=4),
            self.normalization(config.features[0])            
        ).to(core.device))
        
        self.append(nn.Sequential(
            *[ConvNeXtBlock(config.features[0], self.padding, self.normalization) for _ in range(config.blocks[0])]
        ).to(core.device))

        for i in range(1, 4):
            k = 2 if config.downsampling > i+1 else 1
            self.append(nn.Sequential(
                self.normalization(config.features[i-1]),
                nn.Conv2d(config.features[i-1], config.features[i], k, stride=k),
                *[ConvNeXtBlock(config.features[i], self.padding, self.normalization) for _ in range(config.blocks[i])]
            ).to(core.device))
    
    def forward(self, x):
        y = [x,]
        for layer in self:
            y.append(layer(y[-1]))
        return tuple(y[1:])
    
    
def ConvNeXt_XXT(config, params):
    config.features = [24*(2**i) for i in range(4)]
    config.blocks = [3, 3, 9, 3]
    return ConvNeXt(config, params)

def ConvNeXt_XT(config, params):
    config.features = [64*(2**i) for i in range(4)]
    config.blocks = [3, 3, 9, 3]
    return ConvNeXt(config, params)

def ConvNeXt_T(config, params):
    config.features = [96*(2**i) for i in range(4)]
    config.blocks = [3, 3, 9, 3]
    return ConvNeXt(config, params)

def ConvNeXt_S(config, params):
    config.features = [96*(2**i) for i in range(4)]
    config.blocks = [3, 3, 27, 3]
    return ConvNeXt(config, params)

def ConvNeXt_B(config, params):
    config.features = [128*(2**i) for i in range(4)]
    config.blocks = [3, 3, 27, 3]
    return ConvNeXt(config, params)

def ConvNeXt_L(config, params):
    config.features = [192*(2**i) for i in range(4)]
    config.blocks = [3, 3, 27, 3]
    return ConvNeXt(config, params)

def ConvNeXt_XL(config, params):
    config.features = [256*(2**i) for i in range(4)]
    config.blocks = [3, 3, 27, 3]
    return ConvNeXt(config, params)
