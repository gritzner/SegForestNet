# https://arxiv.org/abs/1610.02357
import core
import torch
import torch.nn as nn
import utils


class EntryFlow(nn.ModuleList):
    def __init__(self, parent, in_filters, out_filters, additional_activation=True, stride=2):
        super().__init__()
        block = [parent.activation(in_filters)] if additional_activation else []
        block.extend([
            parent.padding(1),
            nn.Conv2d(in_filters, in_filters, 3, groups=in_filters),
            nn.Conv2d(in_filters, out_filters, 1),
            parent.normalization(out_filters),
            parent.activation(out_filters),
            parent.padding(1),
            nn.Conv2d(out_filters, out_filters, 3, groups=out_filters),
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
    def __init__(self, parent, n):
        super().__init__()
        for _ in range(n):
            self.append(nn.Sequential(
                parent.activation(728),
                parent.padding(1),
                nn.Conv2d(728, 728, 3, groups=728),
                nn.Conv2d(728, 728, 1),
                parent.normalization(728),
                parent.activation(728),
                parent.padding(1),
                nn.Conv2d(728, 728, 3, groups=728),
                nn.Conv2d(728, 728, 1),
                parent.normalization(728),
                parent.activation(728),
                parent.padding(1),
                nn.Conv2d(728, 728, 3, groups=728),
                nn.Conv2d(728, 728, 1),
                parent.normalization(728),
            ))
    
    def forward(self, x):
        for layer in self:
            x = layer(x) + x
        return x
    
    
class ExitFlow(nn.ModuleList):
    def __init__(self, parent, stride):
        super().__init__()
        self.append(nn.Sequential(
            parent.activation(728),
            parent.padding(1),
            nn.Conv2d(728, 728, 3, groups=728),
            nn.Conv2d(728, 728, 1),
            parent.normalization(728),
            parent.activation(728),
            parent.padding(1),
            nn.Conv2d(728, 728, 3, groups=728),
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


class XceptionFeatureExtractor(nn.ModuleList):
    def __init__(self, config, num_input_channels):
        super().__init__()
        
        if not (3 <= config.downsampling <= 5):
            raise RuntimeError(f"unsupported downsampling setting '{config.downsampling}'")
        self.model_name = "XceptionFeatureExtractor"
        self.downsampling = config.downsampling
        self.num_features = (64, 128, 256, 728, 2048)
        self.activation = utils.relu_wrapper(getattr(config,"relu_type","LeakyReLU")) # default from paper: ReLU
        self.padding = getattr(nn, getattr(config, "padding", "ZeroPad2d"))
        self.normalization = utils.norm_wrapper(getattr(config,"norm_type","BatchNorm2d")) # default from paper: BatchNorm2d
        
        self.append(nn.Sequential(
            self.padding(1),
            nn.Conv2d(num_input_channels, 32, 3, stride=2),
            self.normalization(32),
            self.activation(32),
            self.padding(1),
            nn.Conv2d(32, 64, 3),
            self.normalization(64),
            self.activation(64),
        ).to(core.device))
        
        self.append(nn.Sequential(
            EntryFlow(self, 64, 128, False),
        ).to(core.device))
        
        self.append(nn.Sequential(
            EntryFlow(self, 128, 256),
        ).to(core.device))
        
        self.append(nn.Sequential(
            EntryFlow(self, 256, 728, stride=2 if config.downsampling > 3 else 1),
            MiddleFlow(self, 8),
        ).to(core.device))
        
        self.append(nn.Sequential(
            ExitFlow(self, 2 if config.downsampling == 5 else 1),
            self.padding(1),
            nn.Conv2d(1024, 1024, 3, groups=1024),
            nn.Conv2d(1024, 1536, 1),
            self.normalization(1536),
            self.activation(1536),
            self.padding(1),
            nn.Conv2d(1536, 1536, 3, groups=1536),
            nn.Conv2d(1536, 2048, 1),
            self.normalization(2048),
            self.activation(2048),
        ).to(core.device))
    
        if getattr(config, "pretrained_encoder", False):
            print("using pretrained model weights")
            with core.open(core.user.model_weights_paths["Xception"], "rb") as f:
                weights = torch.load(f, map_location=core.device)
            import bz2
            import json
            with bz2.open("models/xception.json.bz2", "r") as f:
                mapping = json.load(f)
            new_weights = {k: (torch.zeros_like(v) if "bias" in k else v.clone()) for k, v in self.state_dict().items()}
            for k, v in mapping.items():
                new_weights[k] = weights[v]
            self.load_state_dict(new_weights)
        
    def forward(self, x):
        y = [x,]
        for layer in self:
            y.append(layer(y[-1]))
        return tuple(y[1:])
