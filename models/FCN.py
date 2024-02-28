# https://arxiv.org/abs/1411.4038
import core
import torch.nn as nn
import models


def FCNEncoder(config, params):
    encoder = models.XceptionFeatureExtractor(config, params.input_shape[0])
    assert encoder.downsampling == 5
    return encoder.to(core.device)


class FCN(nn.Module):
    def __init__(self, config, params):
        super().__init__()

        if params.input_shape[1] != params.input_shape[2]:
            raise RuntimeError("input shape for FCN must be square")
    
        self.encoder = FCNEncoder(config, params)
        self.model_name = "FCN"
            
        self.score32 = nn.Conv2d(self.encoder.num_features[-1], params.num_classes, 1).to(core.device)
        self.upsample32 = nn.ConvTranspose2d(params.num_classes, params.num_classes, 4, stride=2, bias=False).to(core.device)
        self.score16 = nn.Conv2d(self.encoder.num_features[-2], params.num_classes, 1).to(core.device)
        self.upsample16 = nn.ConvTranspose2d(params.num_classes, params.num_classes, 4, stride=2, bias=False).to(core.device)
        self.score8 = nn.Conv2d(self.encoder.num_features[-3], params.num_classes, 1).to(core.device)
        self.upsample8 = nn.ConvTranspose2d(params.num_classes, params.num_classes, 16, stride=8, bias=False).to(core.device)
                        
    def forward(self, x, yt, ce_loss_func, weight, ignore_index):
        y = self.encoder(x)
        y0 = self.upsample32(self.score32(y[-1]))
        y1 = self.upsample16(self.score16(y[-2]) + y0[:,:,1:-1,1:-1])
        y2 = self.upsample8(self.score8(y[-3]) + y1[:,:,1:-1,1:-1])
        yp = y2[:,:,4:-4,4:-4]
        return yp, ce_loss_func(yp, yt, weight=weight, ignore_index=ignore_index)
