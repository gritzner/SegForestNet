# https://arxiv.org/abs/1807.10165
import core
import torch
import torch.nn as nn
import utils
from .UNet import ConvBlock, UNetEncoder


class UNetpp(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        
        if params.input_shape[1] != params.input_shape[2]:
            raise RuntimeError("input shape for UNet must be square")

        self.model_name = ("UNetpp", "UNetpp")
        self.deep_supervision = getattr(config, "deep_supervision", True)

        self.encoder = UNetEncoder(config, params, (32, 64, 128, 256, 512))

        self.x31 = ConvBlock(self.encoder, [768, 256, 256]).to(core.device)
        
        self.x21 = ConvBlock(self.encoder, [384, 128, 128]).to(core.device)
        self.x22 = ConvBlock(self.encoder, [512, 128, 128]).to(core.device)
        
        self.x11 = ConvBlock(self.encoder, [192, 64, 64]).to(core.device)
        self.x12 = ConvBlock(self.encoder, [256, 64, 64]).to(core.device)
        self.x13 = ConvBlock(self.encoder, [320, 64, 64]).to(core.device)
        
        self.x01 = ConvBlock(self.encoder, [96, 32, 32]).to(core.device)
        self.x02 = ConvBlock(self.encoder, [128, 32, 32]).to(core.device)
        self.x03 = ConvBlock(self.encoder, [160, 32, 32]).to(core.device)
        self.x04 = ConvBlock(self.encoder, [192, 32, 32]).to(core.device)
        
        if self.deep_supervision:
            self.classify1 = nn.Conv2d(32, params.num_classes, kernel_size=1).to(core.device)
            self.classify2 = nn.Conv2d(32, params.num_classes, kernel_size=1).to(core.device)
            self.classify3 = nn.Conv2d(32, params.num_classes, kernel_size=1).to(core.device)
        self.classify4 = nn.Conv2d(32, params.num_classes, kernel_size=1).to(core.device)
            
    def forward(self, x, encoder=None):
        if not encoder:
            encoder = self.encoder
        x00, x10, x20, x30, x40 = encoder(x)
        
        
        x31 = nn.functional.interpolate(x40, scale_factor=2, mode="bilinear", align_corners=False)
        x31 = self.x31(torch.cat((x30, x31), dim=1))
        
        x21 = nn.functional.interpolate(x30, scale_factor=2, mode="bilinear", align_corners=False)
        x21 = self.x21(torch.cat((x20, x21), dim=1))
        
        x22 = nn.functional.interpolate(x31, scale_factor=2, mode="bilinear", align_corners=False)
        x22 = self.x22(torch.cat((x20, x21, x22), dim=1))
        
        x11 = nn.functional.interpolate(x20, scale_factor=2, mode="bilinear", align_corners=False)
        x11 = self.x11(torch.cat((x10, x11), dim=1))

        x12 = nn.functional.interpolate(x21, scale_factor=2, mode="bilinear", align_corners=False)
        x12 = self.x12(torch.cat((x10, x11, x12), dim=1))
        
        x13 = nn.functional.interpolate(x22, scale_factor=2, mode="bilinear", align_corners=False)
        x13 = self.x13(torch.cat((x10, x11, x12, x13), dim=1))
        
        x01 = nn.functional.interpolate(x10, scale_factor=2, mode="bilinear", align_corners=False)
        x01 = self.x01(torch.cat((x00, x01), dim=1))
        
        x02 = nn.functional.interpolate(x11, scale_factor=2, mode="bilinear", align_corners=False)
        x02 = self.x02(torch.cat((x00, x01, x02), dim=1))
        
        x03 = nn.functional.interpolate(x12, scale_factor=2, mode="bilinear", align_corners=False)
        x03 = self.x03(torch.cat((x00, x01, x02, x03), dim=1))
        
        x04 = nn.functional.interpolate(x13, scale_factor=2, mode="bilinear", align_corners=False)
        x04 = self.x04(torch.cat((x00, x01, x02, x03, x04), dim=1))
        
        if self.deep_supervision:
            return self.classify1(x01) + self.classify2(x02) + self.classify3(x03) + self.classify4(x04)
        
        return self.classify4(x04)
