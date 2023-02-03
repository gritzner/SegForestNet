# https://arxiv.org/abs/1801.04381
import core
import torch
import torch.nn as nn
import utils


class InvBottleneck(nn.ModuleList):
    def __init__(self, prev_filters, padding, normalization, activation, t, c, n, s, initial_dilation=1, dilation=1, involution=False):
        super().__init__()
        for sub_index in range(n):
            _c0 = prev_filters if sub_index == 0 else c
            _c1 = t * _c0
            _s = s if sub_index == 0 else 1
            _d = initial_dilation if sub_index == 0 else dilation
            self.append(nn.Sequential(
                nn.Conv2d(_c0, _c1, 1),
                normalization(_c1),
                activation(_c1),
                padding(_d),
                utils.Involution(_c1, _c1, 3, stride=_s, dilation=_d) if involution else nn.Conv2d(_c1, _c1, 3, stride=_s, dilation=_d, groups=_c1),
                normalization(_c1),
                activation(_c1),
                nn.Conv2d(_c1, c, 1),
                normalization(c)
            ))
    
    def forward(self, x):
        for sub_index, layer in enumerate(self):
            x = layer(x) if sub_index == 0 else layer(x) + x
        return x


class MobileNetv2(nn.ModuleList):
    def __init__(self, config, params):
        super().__init__()
        
        if not (3 <= config.downsampling <= 5):
            raise RuntimeError(f"unsupported downsampling setting '{config.downsampling}'")
        self.model_name = "MobileNetv2"
        self.downsampling = config.downsampling
        self.num_features = (16, 24, 32, 96, 320)
        self.activation = utils.relu_wrapper(getattr(config,"relu_type","LeakyReLU")) # default from paper: ReLU6
        self.padding = getattr(nn, getattr(config, "padding", "ZeroPad2d"))
        involution = getattr(config, "involution", False)
        dilation_rates = (2,4) if getattr(config, "use_dilated_convs", False) else (1,1)
        self.normalization = utils.norm_wrapper(getattr(config,"norm_type","BatchNorm2d")) # default from paper: BatchNorm2d
        
        self.append(nn.Sequential(
            self.padding(1),
            nn.Conv2d(params.input_shape[0], 32, 3, stride=2),
            self.normalization(32),
            self.activation(32),
            InvBottleneck(32, self.padding, self.normalization, self.activation, 1, 16, 1, 1, involution=involution),
        ).to(core.device))
        
        self.append(nn.Sequential(
            InvBottleneck(16, self.padding, self.normalization, self.activation, 6, 24, 2, 2, involution=involution),
        ).to(core.device))
        
        self.append(nn.Sequential(
            InvBottleneck(24, self.padding, self.normalization, self.activation, 6, 32, 3, 2, involution=involution),
        ).to(core.device))
        
        if 4 <= config.downsampling <= 5:
            self.append(nn.Sequential(
                InvBottleneck(32, self.padding, self.normalization, self.activation, 6, 64, 4, 2, involution=involution),
                InvBottleneck(64, self.padding, self.normalization, self.activation, 6, 96, 3, 1, involution=involution),
            ).to(core.device))
            if config.downsampling == 5:
                self.append(nn.Sequential(
                    InvBottleneck(96, self.padding, self.normalization, self.activation, 6, 160, 3, 2, involution=involution),
                    InvBottleneck(160, self.padding, self.normalization, self.activation, 6, 320, 1, 1, involution=involution),
                ).to(core.device))
            else:
                self.append(nn.Sequential(
                    InvBottleneck(96, self.padding, self.normalization, self.activation, 6, 160, 3, 1, dilation=dilation_rates[0], involution=involution),
                    InvBottleneck(160, self.padding, self.normalization, self.activation, 6, 320, 1, 1, initial_dilation=dilation_rates[0], involution=involution),
                ).to(core.device))
        else:
            self.append(nn.Sequential(
                InvBottleneck(32, self.padding, self.normalization, self.activation, 6, 64, 4, 1, dilation=dilation_rates[0], involution=involution),
                InvBottleneck(64, self.padding, self.normalization, self.activation, 6, 96, 3, 1, initial_dilation=dilation_rates[0], dilation=dilation_rates[0], involution=involution),
            ).to(core.device))
            self.append(nn.Sequential(
                InvBottleneck(96, self.padding, self.normalization, self.activation, 6, 160, 3, 1, initial_dilation=dilation_rates[0], dilation=dilation_rates[1], involution=involution),
                InvBottleneck(160, self.padding, self.normalization, self.activation, 6, 320, 1, 1, initial_dilation=dilation_rates[1], involution=involution),
            ).to(core.device))
        
    def forward(self, x):
        y = [x,]
        for layer in self:
            y.append(layer(y[-1]))
        return tuple(y[1:])
