# https://arxiv.org/abs/1904.05730
# https://openaccess.thecvf.com/content_CVPR_2019/papers/Mou_A_Relation-Augmented_Fully_Convolutional_Network_for_Semantic_Segmentation_in_Aerial_CVPR_2019_paper.pdf
import core
import torch
import torch.nn as nn
from .FCN import FCNEncoder


class RelationModule(nn.Module):
    def __init__(self, in_features, out_features, hw, upsampling_factor):
        super().__init__()
        
        self.channel_u = nn.Conv2d(in_features, in_features, 1)
        self.channel_v = nn.Conv2d(in_features, in_features, 1)
        
        self.spatial_u = nn.Conv2d(in_features, in_features, 1)
        self.spatial_v = nn.Conv2d(in_features, in_features, 1)
        
        self.score = nn.Conv2d(in_features+hw, out_features, 1)
        self.upsample = nn.ConvTranspose2d(out_features, out_features, 2*upsampling_factor, stride=upsampling_factor, bias=False)
            
    def forward(self, x, y=None):
        
        # channel relations
        c = nn.functional.adaptive_avg_pool2d(x, 1)
        u = self.channel_u(c).squeeze().unsqueeze(2)
        v = self.channel_v(c).squeeze().unsqueeze(1)
        c = torch.bmm(u, v)
        c = nn.functional.softmax(c, dim=1)
        
        z = x.reshape(x.shape[0], x.shape[1], -1)
        x = torch.bmm(c, z).reshape(*x.shape)
        
        # spatial relations
        u = self.channel_u(x).reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        v = self.channel_v(x).reshape(x.shape[0], x.shape[1], -1)
        z = torch.bmm(u, v).reshape(x.shape[0], -1, x.shape[2], x.shape[3])
        x = torch.cat((x, z), dim=1)
        
        # FCN decoder
        return self.upsample(self.score(x) + y[:,:,1:-1,1:-1]) if isinstance(y, torch.Tensor) else self.upsample(self.score(x))
    
    
class RAFCN(nn.Module):
    def __init__(self, config, params):
        super().__init__()

        if params.input_shape[1] != params.input_shape[2]:
            raise RuntimeError("input shape for RAFCN must be square")
    
        self.encoder = FCNEncoder(config, params)
        self.model_name = "RAFCN"
            
        hw = (params.input_shape[1]//32)**2
        self.output32 = RelationModule(self.encoder.num_features[-1], params.num_classes, hw, 2).to(core.device)
        hw = (params.input_shape[1]//16)**2
        self.output16 = RelationModule(self.encoder.num_features[-2], params.num_classes, hw, 2).to(core.device)
        hw = (params.input_shape[1]//8)**2
        self.output8 = RelationModule(self.encoder.num_features[-3], params.num_classes, hw, 8).to(core.device)
                        
    def forward(self, x, yt, ce_loss_func, weight, ignore_index):
        y = self.encoder(x)
        y = self.output8(
            y[-3],
            self.output16(
                y[-2],
                self.output32(y[-1])
            )
        )
        yp = y[:,:,4:-4,4:-4]
        return yp, ce_loss_func(yp, yt, weight=weight, ignore_index=ignore_index)
