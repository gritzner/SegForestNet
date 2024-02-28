# https://arxiv.org/abs/1505.04597
import core
import torch
import torch.nn as nn
import utils


class ConvBlock(nn.Module):
    def __init__(self, parent, channels, pooling=False, upsampling=False):
        super().__init__()
        
        if pooling:
            block = [nn.MaxPool2d(2)]
        else:
            block = []
        
        block.extend([
            parent.padding(1),
            nn.Conv2d(channels[0], channels[1], kernel_size=3),
            parent.normalization(channels[1]),
            parent.activation(channels[1]),
            parent.padding(1),
            nn.Conv2d(channels[1], channels[2], kernel_size=3),
            parent.normalization(channels[2]),
            parent.activation(channels[2])
        ])
        
        if upsampling:
            block.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return self.conv_block(x)


class UNetEncoder(nn.Module):
    def __init__(self, config, params, num_features):
        super().__init__()

        self.activation = utils.relu_wrapper(getattr(config,"relu_type","LeakyReLU")) # default from paper: ReLU
        self.padding = getattr(nn, getattr(config, "padding", "ZeroPad2d"))
        self.normalization = utils.norm_wrapper(getattr(config,"norm_type","BatchNorm2d")) # default from paper: BatchNorm2d
        self.downsampling = 4
        
        backbone = [
            ConvBlock(self, [params.input_shape[0], num_features[0], num_features[0]]),
            ConvBlock(self, [num_features[0], num_features[1], num_features[1]], pooling=True),
            ConvBlock(self, [num_features[1], num_features[2], num_features[2]], pooling=True),
            ConvBlock(self, [num_features[2], num_features[3], num_features[3]], pooling=True),
            ConvBlock(self, [num_features[3], num_features[4], num_features[4]], pooling=True)
        ]
        self.backbone = nn.Sequential(*backbone).to(core.device)

    def forward(self, x):
        y = [x]
        for block in self.backbone:
            y.append(block(y[-1]))
        return tuple(y[1:])
    

class UNet(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        
        if params.input_shape[1] != params.input_shape[2]:
            raise RuntimeError("input shape for UNet must be square")

        self.model_name = "UNet"
        num_features = (64, 128, 256, 512, 1024)
        if getattr(config, "small", False):
            num_features = tuple([i//2 for i in num_features])

        self.encoder = UNetEncoder(config, params, num_features)
        self.decoder = [
            ConvBlock(self.encoder, [sum(num_features[3:5]), num_features[3], num_features[3]], upsampling=True),
            ConvBlock(self.encoder, [sum(num_features[2:4]), num_features[2], num_features[2]], upsampling=True),
            ConvBlock(self.encoder, [sum(num_features[1:3]), num_features[1], num_features[1]], upsampling=True),
            ConvBlock(self.encoder, [sum(num_features[0:2]), num_features[0], num_features[0]]),
            nn.Conv2d(num_features[0], params.num_classes, kernel_size=1)
        ]
        self.decoder = nn.Sequential(*self.decoder).to(core.device)
            
    def forward(self, x, yt, ce_loss_func, weight, ignore_index):
        y0 = self.encoder(x)
        y1 = nn.functional.interpolate(y0[-1], scale_factor=2, mode="bilinear", align_corners=False)
        for i, block in enumerate(self.decoder):
            if i < 4:
                y1 = torch.cat([y0[3-i], y1], dim=1)
            y1 = block(y1)
        return y1, ce_loss_func(y1, yt, weight=weight, ignore_index=ignore_index)
