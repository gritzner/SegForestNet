# https://openaccess.thecvf.com/content/CVPR2021/html/Li_PointFlow_Flowing_Semantics_Through_Points_for_Aerial_Image_Segmentation_CVPR_2021_paper.html
import core
import torch
import torch.nn as nn
import functools
from .FCN import FCNEncoder


class PPModule(nn.Module):
    def __init__(self, config, params, encoder):
        super().__init__()
        
        feature_map_size = params.input_shape[1] // 2**encoder.downsampling
        
        for size in (1, 2, 3, 6):
            setattr(
                self, f"conv{size}",
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(encoder.num_features[-1], config.num_features, 1),
                    encoder.normalization(config.num_features),
                    nn.Upsample(size=feature_map_size, mode="bilinear", align_corners=False)
                )
            )
        self.conv = nn.Sequential(
            nn.Conv2d(encoder.num_features[-1] + 4*config.num_features, config.num_features, 1),
            encoder.normalization(config.num_features),
            encoder.activation(config.num_features)
        )
    
    def forward(self, x):
        y = [self.conv1(x), self.conv2(x), self.conv3(x), self.conv6(x), x]
        y = torch.cat(y, dim=1)
        return self.conv(y)


###################################################################################################################
### the code below has been copied from https://github.com/lxtGH/PFSegNets/blob/master/network/nn/point_flow.py ###
### minor changes were made, e.g., changes necessary to work with the imports of this file                      ###
### commit which was copied: 74e7545 from June 28th, 2021                                                       ###
###################################################################################################################

def point_sample(input, point_coords):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = nn.functional.grid_sample(input, 2.0 * point_coords - 1.0, align_corners=True)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.
    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.
    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (torch.div(point_indices, W, rounding_mode="trunc")).to(torch.float) * h_step
    
    return point_indices, point_coords


class PointMatcher(nn.Module):
    """
        Simple Point Matcher
    """
    def __init__(self, dim, padding):
        super(PointMatcher, self).__init__()
        self.match_conv = nn.Sequential(padding(1), nn.Conv2d(dim*2, 1, 3))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_high, x_low = x
        x_low = nn.functional.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        certainty = self.match_conv(torch.cat([x_high, x_low], dim=1))
        return self.sigmoid(certainty)


class PointFlowModuleWithMaxAvgpool(nn.Module):
    def __init__(self, config, encoder):
        super(PointFlowModuleWithMaxAvgpool, self).__init__()
        self.dim = config.reduced_num_features
        self.point_matcher = PointMatcher(config.reduced_num_features, encoder.padding)
        self.down_h = nn.Conv2d(config.num_features, config.reduced_num_features, 1)
        self.down_l = nn.Conv2d(config.num_features, config.reduced_num_features, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.maxpool_size = config.maxpool_size
        self.avgpool_size = config.avgpool_size
        self.edge_points = config.edge_points
        self.max_pool = nn.AdaptiveMaxPool2d((config.maxpool_size, config.maxpool_size), return_indices=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((config.avgpool_size, config.avgpool_size))
        self.edge_final = nn.Sequential(
            encoder.padding(1),
            nn.Conv2d(in_channels=config.num_features, out_channels=config.num_features, kernel_size=3, bias=False),
            encoder.normalization(config.num_features),
            encoder.activation(config.num_features),
            encoder.padding(1),
            nn.Conv2d(in_channels=config.num_features, out_channels=1, kernel_size=3, bias=False)
        )

    def forward(self, x):
        x_high, x_low = x
        stride_ratio = x_low.shape[2] / x_high.shape[2]
        x_high_embed = self.down_h(x_high)
        x_low_embed = self.down_l(x_low)
        N, C, H, W = x_low.shape
        N_h, C_h, H_h, W_h = x_high.shape

        certainty_map = self.point_matcher([x_high_embed, x_low_embed])
        avgpool_grid = self.avg_pool(certainty_map)
        _, _, map_h, map_w = certainty_map.size()
        avgpool_grid = nn.functional.interpolate(avgpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)

        # edge part
        x_high_edge = x_high - x_high * avgpool_grid
        edge_pred = self.edge_final(x_high_edge)
        point_indices, point_coords = get_uncertain_point_coords_on_grid(edge_pred, num_points=self.edge_points)
        sample_x = point_indices % W_h * stride_ratio
        sample_y = torch.div(point_indices, W_h, rounding_mode="trunc") * stride_ratio
        low_edge_indices = sample_x + sample_y * W
        low_edge_indices = low_edge_indices.unsqueeze(1).expand(-1, C, -1).long()
        high_edge_feat = point_sample(x_high, point_coords)
        low_edge_feat = point_sample(x_low, point_coords)
        affinity_edge = torch.bmm(high_edge_feat.transpose(2, 1), low_edge_feat).transpose(2, 1)
        affinity = self.softmax(affinity_edge)
        high_edge_feat = torch.bmm(affinity, high_edge_feat.transpose(2, 1)).transpose(2, 1)
        fusion_edge_feat = high_edge_feat + low_edge_feat

        # residual part
        maxpool_grid, maxpool_indices = self.max_pool(certainty_map)
        maxpool_indices = maxpool_indices.expand(-1, C, -1, -1)
        maxpool_grid = nn.functional.interpolate(maxpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)
        x_indices = maxpool_indices % W_h * stride_ratio
        y_indices = torch.div(maxpool_indices, W_h, rounding_mode="trunc") * stride_ratio
        low_indices = x_indices + y_indices * W
        low_indices = low_indices.long()
        x_high = x_high + maxpool_grid*x_high
        flattened_high = x_high.flatten(start_dim=2)
        high_features = flattened_high.gather(dim=2, index=maxpool_indices.flatten(start_dim=2)).view_as(maxpool_indices)
        flattened_low = x_low.flatten(start_dim=2)
        low_features = flattened_low.gather(dim=2, index=low_indices.flatten(start_dim=2)).view_as(low_indices)
        feat_n, feat_c, feat_h, feat_w = high_features.shape
        high_features = high_features.view(feat_n, -1, feat_h*feat_w)
        low_features = low_features.view(feat_n, -1, feat_h*feat_w)
        affinity = torch.bmm(high_features.transpose(2, 1), low_features).transpose(2, 1)
        affinity = self.softmax(affinity)  # b, n, n
        high_features = torch.bmm(affinity, high_features.transpose(2, 1)).transpose(2, 1)
        fusion_feature = high_features + low_features
        mp_b, mp_c, mp_h, mp_w = low_indices.shape
        low_indices = low_indices.view(mp_b, mp_c, -1)

        final_features = x_low.reshape(N, C, H*W).scatter(2, low_edge_indices, fusion_edge_feat)
        final_features = final_features.scatter(2, low_indices, fusion_feature).view(N, C, H, W)

        return final_features, edge_pred

###################################################################################################################
### end of code copied from https://github.com/lxtGH/PFSegNets/blob/master/network/nn/point_flow.py             ###
###################################################################################################################


class PFNet(nn.Module):
    def __init__(self, config, params):
        super().__init__()

        if params.input_shape[1] != params.input_shape[2]:
            raise RuntimeError("input shape for PFNet must be square")
        
        defaults = {
            "num_features": 256, "reduced_num_features": 64, "maxpool_size": 8, "avgpool_size": 8, "edge_points": 128
        }
        for k, v in defaults.items():
            setattr(config, k, getattr(config, k, v))
    
        self.encoder = FCNEncoder(config, params)
        self.model_name = ("PFNet", self.encoder.model_name)
        self.ppm = PPModule(config, params, self.encoder).to(core.device)
            
        for i in range(1, 4):
            setattr(
                self, f"conv{i}a",
                nn.Sequential(
                    nn.Conv2d(self.encoder.num_features[i], config.num_features, 1),
                    self.encoder.normalization(config.num_features),
                    self.encoder.activation(config.num_features)
                ).to(core.device)
            )
            setattr(self, f"pfmodule{i}", PointFlowModuleWithMaxAvgpool(config, self.encoder).to(core.device))
            setattr(
                self, f"conv{i}b",
                nn.Sequential(
                    nn.Conv2d(config.num_features, config.num_features, 1),
                    self.encoder.normalization(config.num_features),
                    self.encoder.activation(config.num_features)
                ).to(core.device)
            )
            
        self.classifier = nn.Sequential(
            nn.Conv2d(4*config.num_features, config.num_features, 1),
            self.encoder.normalization(config.num_features),
            self.encoder.activation(config.num_features),
            nn.Conv2d(config.num_features, params.num_classes, 1),
        ).to(core.device)        
                        
    def forward(self, x, encoder=None):
        if not encoder:
            encoder = self.encoder
        y = encoder(x)
        
        z = [self.ppm(y[-1])]
        self.edge_maps = []
        for i in reversed(range(1, 4)):
            feature_map = getattr(self, f"conv{i}a")(y[i])
            res_feature_map, edge_map = getattr(self, f"pfmodule{i}")((z[-1], feature_map))
            feature_map = getattr(self, f"conv{i}b")(feature_map + res_feature_map)
            z.append(feature_map)
            self.edge_maps.append(torch.sigmoid(edge_map))
            
        for i, feature_map in enumerate(z[:-1]):
            z[i] = nn.functional.interpolate(feature_map, scale_factor=2**(3-i), mode="bilinear", align_corners=False)
        z = self.classifier(torch.cat(z, dim=1))
        return nn.functional.interpolate(z, scale_factor=4, mode="bilinear", align_corners=False)
    
    def get_loss_function(self, class_weights, ignore_class):
        class PFNetLoss():
            def __init__(self, model):
                self.model = model
            
            def __call__(self, yp, yt):
                loss = [nn.functional.cross_entropy(yp, yt, weight=class_weights, ignore_index=ignore_class)]
                
                true_edge_map = torch.zeros_like(yt).detach().float()
                true_edge_map[:,1:-1,1:-1] = yt[:,1:-1,1:-1]
                true_edge_map[:,1:-1,1:-1] -= nn.functional.avg_pool2d(yt.float().unsqueeze(1), 3, 1).squeeze(1)
                i = torch.abs(true_edge_map) > 10**-6
                true_edge_map[i] = 1
                true_edge_map[torch.logical_not(i)] = 0
                
                for output_stride, edge_map in zip((32, 16, 8), self.model.edge_maps):
                    target_edge_map = nn.functional.avg_pool2d(true_edge_map, output_stride)
                    i = target_edge_map > .25
                    target_edge_map[i] = 1
                    target_edge_map[torch.logical_not(i)] = 0
                    loss.append(nn.functional.binary_cross_entropy(edge_map.squeeze(1), target_edge_map))
                    
                return functools.reduce(lambda a,b: a+b, loss)
            
        return PFNetLoss(self)
