import core
import numpy as np
import torch
import torch.nn as nn
import models
import functools
import copy
from .SegForestTree import *


class SegForestNetEncoder(nn.Module):
    def __init__(self, config, input_shape, num_output_features):
        super().__init__()
        
        config.features.skips = getattr(config.features, "skips", ())
        
        self.backbone = core.create_object(models.backbones, config.backbone, input_shape=input_shape, num_classes=1)
        self.downsampling = self.backbone.downsampling
        self.downsampling_factor = 2**self.backbone.downsampling
        self.feature_size = input_shape[-1] // self.downsampling_factor
        self.activation = self.backbone.activation
        self.padding = self.backbone.padding
        self.normalization = self.backbone.normalization
        self.encoder_loss = 0
        
        extra_input_features = 0
        for i, skip in enumerate(config.features.skips):
            features_in = self.backbone.num_features[skip.source]
            features_out = round(skip.factor * features_in)
            extra_input_features += features_out
            kernel_size = 2**(self.downsampling-(skip.source+1))
            assert kernel_size > 1
            setattr(
                self, f"skip{i}",
                nn.Sequential(
                    nn.Conv2d(features_in, features_out, kernel_size, stride=kernel_size),
                    self.backbone.normalization(features_out),
                    self.backbone.activation(features_out)
                ).to(core.device)
            )
        self.skips = tuple([skip.source for skip in config.features.skips])
        
        encoder_suffix = [self.backbone.padding(config.features.context)] if config.features.context > 0 else []
        encoder_suffix.extend([
            nn.Conv2d(self.backbone.num_features[-1]+extra_input_features, num_output_features, 1+2*config.features.context),
            self.backbone.normalization(num_output_features),
            self.backbone.activation(num_output_features)
        ])
        self.encoder_suffix = nn.Sequential(*encoder_suffix).to(core.device)
    
    def forward(self, x):
        y = self.backbone(x)
        
        z = [getattr(self, f"skip{i}")(y[j]) for i, j in enumerate(self.skips)]
        z.append(y[-1])
        z = torch.cat(z, dim=1)
        
        return self.encoder_suffix(z)


class SegForestNetVariationalEncoder(SegForestNetEncoder):
    def __init__(self, config, input_shape, num_output_features):
        super().__init__(config, input_shape, config.features.variational)
        self.conv_mu = nn.Conv2d(config.features.variational, num_output_features, 1).to(core.device)
        self.conv_sigma = nn.Conv2d(config.features.variational, num_output_features, 1).to(core.device)
    
    def forward(self, x):
        y = super().forward(x)
        mu, sigma = self.conv_mu(y), (0.5*self.conv_sigma(y)).exp()
        self.encoder_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
        
        if self.training:
            z = torch.randn_like(sigma)
        else:
            z = torch.zeros_like(sigma)
            
        return mu + z*sigma


class SegForestNet(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        if params.input_shape[1] != params.input_shape[2]:
            raise RuntimeError("input shape for SegForestNet must be square")
        self.loss_config = config.loss
        params.extra_payload = getattr(params, "extra_payload", 0)
            
        self.class_to_tree_map = [-1] * params.num_classes
        num_encoder_output_features = 0
        for i, tree in enumerate(config.trees):
            if not hasattr(tree, "outputs"):
                assert len(config.trees) == 1
                tree.outputs = list(range(params.num_classes))
            if type(tree.outputs) != np.ndarray:
                for output in tree.outputs:
                    assert type(output) == int
                    assert 0 <= output < params.num_classes
                tree.outputs = np.asarray(tree.outputs, dtype=np.int32)
            for output in tree.outputs:
                assert self.class_to_tree_map[output] == -1
                self.class_to_tree_map[output] = i
            num_encoder_output_features += tree.num_features.shape + tree.num_features.content
        self.class_to_tree_map = tuple(self.class_to_tree_map)
        assert -1 not in self.class_to_tree_map

        if config.features.variational > 0:
            self.encoder = SegForestNetVariationalEncoder(config, params.input_shape, num_encoder_output_features)
        else:
            self.encoder = SegForestNetEncoder(config, params.input_shape, num_encoder_output_features)
        self.model_name = ("SegForestNet", self.encoder.backbone.model_name)
        
        self.output_shape = (params.num_classes, *params.input_shape[1:])
        self.trees = tuple([PartitioningTree(config, params, self.encoder, tree) for tree in config.trees])
        
        self.num_tree_parameters = 0
        for i, tree in enumerate(self.trees):
            self.num_tree_parameters += tree.num_tree_parameters
            setattr(self, f"_PartitioningTree{i}_decoder0_", tree.decoder0)
            setattr(self, f"_PartitioningTree{i}_decoder1_", tree.decoder1)
        
    def forward(self, x, encoder=None):
        if not encoder:
            encoder = self.encoder
        
        if (not hasattr(self, "sample_points")) or self.sample_points.shape[0] < x.shape[0]:
            self.create_sample_points(x)
            
        x = encoder(x)
        sample_points = self.sample_points[:x.shape[0]]
        
        y = torch.zeros((x.shape[0], *self.output_shape), dtype=torch.float32, device=x.device)
        for tree in self.trees:
            x = tree.render(x, y, sample_points)
            
        return y
    
    def create_sample_points(self, x):
        n = self.encoder.downsampling_factor
        self.sample_points = np.asarray(np.arange(n), dtype=np.float32)
        self.sample_points = (2*self.sample_points + 1) / (2*n)
        self.sample_points = np.meshgrid(self.sample_points, self.sample_points)
        self.sample_points = [np.expand_dims(p,axis=0) for p in self.sample_points]
        self.sample_points = np.concatenate(self.sample_points, axis=0)
        self.sample_points = np.expand_dims(self.sample_points, axis=0)
        self.sample_points = np.tile(self.sample_points, [x.shape[0], 1, self.encoder.feature_size, self.encoder.feature_size])
        self.sample_points = torch.from_numpy(self.sample_points).float().to(x.device)
        
        self.pixel_positions = np.arange(x.shape[-1], dtype=np.float32)
        self.pixel_positions = np.meshgrid(self.pixel_positions, self.pixel_positions)
        self.pixel_positions = [np.expand_dims(p,axis=0) for p in self.pixel_positions]
        self.pixel_positions = np.concatenate(tuple(reversed(self.pixel_positions)), axis=0)
        self.pixel_positions = np.expand_dims(self.pixel_positions, axis=0)
        self.pixel_positions = np.tile(self.pixel_positions, (x.shape[0],1,1,1))
        self.pixel_positions = torch.from_numpy(self.pixel_positions).float().to(x.device)
        
    def get_all_tree_parameters(self):
        tree_parameters = [tree.tree_parameters[0] for tree in self.trees]
        tree_parameters.extend([tree.tree_parameters[1] for tree in self.trees])
        return torch.cat(tree_parameters, dim=1)
        
    def get_loss_function(self, class_weights, ignore_class):
        config = self.loss_config
        if type(self.encoder) != SegForestNetVariationalEncoder:
            config.weights[-1] = 0
        
        class SegForestLoss():
            def __init__(self, model):
                assert config.cross_entropy in ("pixels", "leaves", "leaves_argmax", "leaves_stop", "leaves_argmax_stop")
                assert len(config.weights) == 5
                if config.min_region_size <= 0:
                    config.weights[2] = 0
                
                self.encoder = model.encoder
                self.num_classes = model.output_shape[0]
                self.trees = model.trees
                self.tree_weights = [1/len(self.trees)] * len(self.trees)
                if config.tree_weights:
                    for i, tree in enumerate(self.trees):
                        self.tree_weights[i] = tree.config.outputs.shape[0] / self.num_classes
                
                self.loss_weights = [config.weights[i]/np.sum(config.weights) for i in range(len(config.weights))]
                if config.cross_entropy == "pixels":
                    self.ce_func = lambda yp, yt, _: nn.functional.cross_entropy(yp, yt, weight=class_weights, ignore_index=ignore_class)
                else:
                    apply_argmax = "argmax" in config.cross_entropy
                    apply_detach = "stop" in config.cross_entropy
                    weights = class_weights.clone()
                    if 0 <= ignore_class < weights.shape[0]:
                        weights[ignore_class] = 0
                    self.ce_func = lambda _0, _1, yts: self.leaf_loss(yts, apply_argmax, apply_detach, weights)
                self.dist_func = getattr(SegForestLoss, config.distribution_metric)
                self.min_region_size = config.min_region_size
            
            def __call__(self, yp, yt):
                yt_onehot = nn.functional.one_hot(yt, num_classes=self.num_classes)
                region_dists = [tree.get_region_distributions(yt_onehot) for tree in self.trees]
                ce_loss = self.ce_func(yp, yt, [p for p,_ in region_dists])
                
                if config.debug_region_loss:
                    with torch.no_grad():
                        images = [(self.tree_weights[i] * self.dist_func(p)).detach() for i,(p,s) in enumerate(region_dists)]
                        images = [img.repeat_interleave(self.trees[i].downsampling_factor,3) for i, img in enumerate(images)]
                        images = [img.repeat_interleave(self.trees[i].downsampling_factor,2) for i, img in enumerate(images)]
                        region_maps = [tree.region_map.clone().detach().permute(1, 0, 2, 3) for tree in self.trees]
                        images = [img*region_map for img, region_map in zip(images, region_maps)]
                        for img in images:
                            for i in range(1, img.shape[0]):
                                img[0] += img[i]
                        self.region_loss_images = [img[0].cpu().numpy() for img in images]
                
                loss = [(
                    self.tree_weights[i] * self.dist_func(p).mean(),
                    self.tree_weights[i] * torch.maximum(self.min_region_size-s, torch.zeros(s.shape,device=s.device)).mean(),
                    self.tree_weights[i] * self.dist_func(self.trees[i].region_map.permute(0,2,3,1).clone()).mean()
                ) for i,(p,s) in enumerate(region_dists)]
                loss = functools.reduce(lambda x,y: (x[0]+y[0], x[1]+y[1], x[2]+y[2]), loss)
                
                return self.loss_weights[0]*ce_loss + \
                    self.loss_weights[1]*loss[0] + \
                    self.loss_weights[2]*loss[1] + \
                    self.loss_weights[3]*loss[2] + \
                    self.loss_weights[4]*self.encoder.encoder_loss
            
            def leaf_loss(self, yts, apply_argmax, apply_detach, class_weights):
                loss = []
                for tree_weight, tree, yt in zip(self.tree_weights, self.trees, yts):
                    yp = tree.tree_parameters[1]
                    yp = yp.reshape(yp.shape[0]*yt.shape[0], yp.shape[1]//yt.shape[0], *yp.shape[2:4])
                    if tree.config.outputs.shape[0] != self.num_classes:
                        temp = torch.empty([yp.shape[0], yp.shape[1]+1, *yp.shape[2:]], dtype=yp.dtype, device=core.device)
                        temp[:,:-1] = yp
                        temp[:,-1] = config.ce_constant
                        yp = temp
                    yp = nn.functional.log_softmax(yp, dim=1)
                    
                    yt = yt.permute(1, 0, 4, 2, 3)
                    yt = yt.reshape(yt.shape[0]*yt.shape[1], *yt.shape[2:])
                    if apply_argmax:
                        yt = nn.functional.one_hot(yt.argmax(1), num_classes=yt.shape[1]).permute(0, 3, 1, 2)
                    if apply_detach:
                        yt = yt.detach()
                    
                    ce = -yt * yp
                    for i, class_weight in enumerate(class_weights[tree.config.outputs]):
                        ce[:,i] *= class_weight
                    
                    loss.append(tree_weight * ce.sum(1).mean())
                return functools.reduce(lambda x,y: x+y, loss)
            
            @staticmethod
            def entropy(p):
                p[p<10**-6] = 10**-6
                return -(p * p.log()).sum(-1)
                
            @staticmethod
            def gini(p):
                return 1 - (p**2).sum(-1)
            
        return SegForestLoss(self)
