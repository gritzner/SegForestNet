# https://arxiv.org/abs/1610.02357
import core
import torch
import torch.nn as nn
import utils


class EntryFlow(nn.ModuleList):
    def __init__(self, parent, in_filters, out_filters, additional_activation=True, stride=2, involution=False):
        super().__init__()
        block = [parent.activation(in_filters)] if additional_activation else []
        block.extend([
            parent.padding(1),
            utils.Involution(in_filters, in_filters, 3) if involution else nn.Conv2d(in_filters, in_filters, 3, groups=in_filters),
            nn.Conv2d(in_filters, out_filters, 1),
            parent.normalization(out_filters),
            parent.activation(out_filters),
            parent.padding(1),
            utils.Involution(out_filters, out_filters, 3) if involution else nn.Conv2d(out_filters, out_filters, 3, groups=out_filters),
            nn.Conv2d(out_filters, out_filters, 1),
            parent.normalization(out_filters),
            parent.padding(1),
            nn.MaxPool2d(3, stride=stride)
        ])
        self.append(nn.Sequential(*block))
        self.append(nn.Sequential(
            nn.Conv2d(in_filters, out_filters, 1, stride=stride),
            parent.normalization(out_filters)
        ))
    
    def forward(self, x):
        return self[0](x) + self[1](x)


class MiddleFlow(nn.ModuleList):
    def __init__(self, parent, n, dilation, involution=False):
        super().__init__()
        for _ in range(n):
            self.append(nn.Sequential(
                parent.activation(728),
                parent.padding(dilation),
                utils.Involution(728, 728, 3, dilation=dilation) if involution else nn.Conv2d(728, 728, 3, dilation=dilation, groups=728),
                nn.Conv2d(728, 728, 1),
                parent.normalization(728),
                parent.activation(728),
                parent.padding(dilation),
                utils.Involution(728, 728, 3, dilation=dilation) if involution else nn.Conv2d(728, 728, 3, dilation=dilation, groups=728),
                nn.Conv2d(728, 728, 1),
                parent.normalization(728),
                parent.activation(728),
                parent.padding(dilation),
                utils.Involution(728, 728, 3, dilation=dilation) if involution else nn.Conv2d(728, 728, 3, dilation=dilation, groups=728),
                nn.Conv2d(728, 728, 1),
                parent.normalization(728),
            ))
    
    def forward(self, x):
        for layer in self:
            x = layer(x) + x
        return x
    
    
class ExitFlow(nn.ModuleList):
    def __init__(self, parent, stride, dilation, involution=False):
        super().__init__()
        self.append(nn.Sequential(
            parent.activation(728),
            parent.padding(dilation),
            utils.Involution(728, 728, 3, dilation=dilation) if involution else nn.Conv2d(728, 728, 3, dilation=dilation, groups=728),
            nn.Conv2d(728, 728, 1),
            parent.normalization(728),
            parent.activation(728),
            parent.padding(dilation),
            utils.Involution(728, 728, 3, dilation=dilation) if involution else nn.Conv2d(728, 728, 3, dilation=dilation, groups=728),
            nn.Conv2d(728, 1024, 1),
            parent.normalization(1024),
            parent.padding(1),
            nn.MaxPool2d(3, stride=stride),
        ))
        self.append(nn.Sequential(
            nn.Conv2d(728, 1024, 1, stride=stride),
            parent.normalization(1024),
        ))
    
    def forward(self, x):
        return self[0](x) + self[1](x)


class Xception(nn.ModuleList):
    def __init__(self, config, params):
        super().__init__()
        
        if not (3 <= config.downsampling <= 5):
            raise RuntimeError(f"unsupported downsampling setting '{config.downsampling}'")
        self.model_name = "Xception"
        self.downsampling = config.downsampling
        self.num_features = (32, 128, 256, 728, 2048)
        self.activation = utils.relu_wrapper(getattr(config,"relu_type","LeakyReLU")) # default from paper: ReLU
        self.padding = getattr(nn, getattr(config, "padding", "ZeroPad2d"))
        involution = getattr(config, "involution", False)
        use_dilation = getattr(config, "use_dilated_convs", False)
        self.normalization = utils.norm_wrapper(getattr(config,"norm_type","BatchNorm2d")) # default from paper: BatchNorm2d
        
        self.append(nn.Sequential(
            self.padding(1),
            nn.Conv2d(params.input_shape[0], 32, 3, stride=2),
            self.normalization(32),
            self.activation(32),
            self.padding(1),
            nn.Conv2d(32, 32, 3),
            self.normalization(32),
            self.activation(32),
        ).to(core.device))
        
        self.append(nn.Sequential(
            EntryFlow(self, 32, 128, False, involution=involution),
        ).to(core.device))
        
        self.append(nn.Sequential(
            EntryFlow(self, 128, 256, involution=involution),
        ).to(core.device))
        
        dilation = 2 if use_dilation and config.downsampling == 3 else 1
        self.append(nn.Sequential(
            EntryFlow(self, 256, 728, stride=2 if config.downsampling > 3 else 1, involution=involution),
            MiddleFlow(self, 8, dilation, involution=involution),
        ).to(core.device))
        
        dilation *= 2 if use_dilation and config.downsampling < 5 else 1
        self.append(nn.Sequential(
            ExitFlow(self, 2 if config.downsampling == 5 else 1, dilation, involution=involution),
            self.padding(dilation),
            utils.Involution(1024, 1024, 3, dilation=dilation) if involution else nn.Conv2d(1024, 1024, 3, dilation=dilation, groups=1024),
            nn.Conv2d(1024, 1536, 1),
            self.normalization(1536),
            self.activation(1536),
            self.padding(dilation),
            utils.Involution(1536, 1536, 3, dilation=dilation) if involution else nn.Conv2d(1536, 1536, 3, dilation=dilation, groups=1536),
            nn.Conv2d(1536, 2048, 1),
            self.normalization(2048),
            self.activation(2048),
        ).to(core.device))
    
    def forward(self, x):
        y = [x,]
        for layer in self:
            y.append(layer(y[-1]))
        return tuple(y[1:])
