import core
import numpy as np
import torch
import torch.nn as nn
import types
import utils
import models
import functools
from .SegForestTree import PartitioningTree


class SegForestNet(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        if params.input_shape[1] != params.input_shape[2]:
            raise RuntimeError("input shape for SegForestNet must be square")
        self.softmax_temperature = utils.get_scheduler(config.region_map.softmax_temperature)
        
        if len(config.trees) == 1 and getattr(config.trees[0], "one_tree_per_class", False):
            assert not hasattr(config.trees[0], "outputs")
            tree = config.trees[0].__dict__
            tree["outputs"] = [0]
            for i in range(1, params.num_classes):
                config.trees.append(types.SimpleNamespace(**tree.copy()))
                config.trees[-1].outputs = [i]
        
        class_to_tree_map = [-1] * params.num_classes
        num_encoder_output_features = 0
        for i, tree in enumerate(config.trees):
            if not hasattr(tree, "outputs"):
                assert len(config.trees) == 1
                tree.outputs = list(range(params.num_classes))
            if hasattr(tree, "share_architecture"):
                assert tree.share_architecture < i
                for k, v in config.trees[tree.share_architecture].__dict__.items():
                    if k == "outputs":
                        continue
                    tree.__dict__[k] = v
            if type(tree.outputs) != np.ndarray:
                for output in tree.outputs:
                    assert type(output) == int
                    assert 0 <= output < params.num_classes
                tree.outputs = np.asarray(tree.outputs, dtype=np.int32)
            for output in tree.outputs:
                assert class_to_tree_map[output] == -1
                class_to_tree_map[output] = i
            tree.actual_num_features = types.SimpleNamespace(**tree.num_features.__dict__)
            if config.decoder.vq.type[0] == 7: # references VQGumbel in create_vector_quantization_layer in utils/vectorquantization.py
                tree.num_features.shape = config.decoder.vq.codebook_size
            if config.decoder.vq.type[1] == 7: # references VQGumbel in create_vector_quantization_layer in utils/vectorquantization.py
                tree.num_features.content = config.decoder.vq.codebook_size
            num_encoder_output_features += tree.num_features.shape + tree.num_features.content
        assert -1 not in tuple(class_to_tree_map)

        self.encoder = models.XceptionFeatureExtractor(config, params.input_shape[0])
        self.encoder.downsampling_factor = 2**self.encoder.downsampling
        self.encoder.feature_size = params.input_shape[-1] // self.encoder.downsampling_factor
        encoder_suffix = [self.encoder.padding(config.features.context)] if config.features.context > 0 else []
        encoder_suffix.extend([
            nn.Conv2d(self.encoder.num_features[-1], num_encoder_output_features, 1+2*config.features.context),
            self.encoder.normalization(num_encoder_output_features),
            self.encoder.activation(num_encoder_output_features)
        ])
        self.encoder_suffix = nn.Sequential(*encoder_suffix).to(core.device)        
        self.model_name = "SegForestNet"
        
        self.output_shape = (params.num_classes, *params.input_shape[1:])
        self.trees = tuple([PartitioningTree(config, params, self.encoder, tree) for tree in config.trees])
        
        self.num_tree_parameters = 0
        for i, tree in enumerate(self.trees):
            self.num_tree_parameters += tree.num_tree_parameters
            if hasattr(tree.config, "share_architecture"):
                other_tree = self.trees[tree.config.share_architecture]
                tree.decoder0 = other_tree.decoder0
                tree.decoder1 = other_tree.decoder1
                if hasattr(tree, "region_map_rendering_params"):
                    tree.region_map_rendering_params = other_tree.region_map_rendering_params
                    tree.region_map_rendering.node_weight[1] = other_tree.region_map_rendering_params
                if hasattr(tree, "classifier"):
                    tree.classifier = other_tree.classifier
            else:
                setattr(self, f"_PartitioningTree{i}_decoder0_", tree.decoder0)
                setattr(self, f"_PartitioningTree{i}_decoder1_", tree.decoder1)
                if hasattr(tree, "region_map_rendering_params"):
                    setattr(self, f"_PartitioningTree{i}_region_map_rendering_params_", tree.region_map_rendering_params)
                if hasattr(tree, "classifier"):
                    setattr(self, f"_PartitioningTree{i}_classifier_", tree.classifier)
                    
        config = config.loss
        assert config.cross_entropy in ("pixels", "leaves", "leaves_argmax", "leaves_stop", "leaves_argmax_stop")
        assert len(config.weights) == 5
        if config.min_region_size <= 0:
            config.weights[2] = 0
                
        self.tree_weights = [tree.config.outputs.shape[0]/params.num_classes for tree in self.trees]
        self.loss_weights = [config.weights[i]/np.sum(config.weights) for i in range(len(config.weights))]
                
        if config.cross_entropy != "pixels":
            apply_argmax = "argmax" in config.cross_entropy
            apply_detach = "stop" in config.cross_entropy
            self.ce_loss_func = lambda yts, w, ic: self.leaf_loss(yts, apply_argmax, apply_detach, w, ic)
        self.dist_func = getattr(SegForestNet, f"dist_metric_{config.distribution_metric}")
        self.min_region_size = config.min_region_size
                    
    def prepare_for_epoch(self, epoch, epochs):
        self.softmax_temperature(epoch, epochs)
        for i, tree in enumerate(self.trees):
            tree.region_map_rendering.softmax_temperature = self.softmax_temperature.value
            if hasattr(tree.config, "share_architecture"):
                continue
            for decoder in (tree.decoder0, tree.decoder1):
                decoder[0][0].prepare_for_epoch(epoch, epochs) # vector quantization layer
        
    def forward(self, x, yt, ce_loss_func, weight, ignore_index, x_vis=None, lut=None, region_visualization_path=""):
        if (not hasattr(self, "sample_points")) or self.sample_points.shape[0] < x.shape[0]:
            self.create_sample_points(x)
        
        x = [x, *self.encoder(x)]
        z = self.encoder_suffix(x[-1])
        for i in {tree.config.classifier_skip_from for tree in self.trees if hasattr(tree.config, "classifier_skip_from") and tree.config.classifier_skip_from > 0}:
            x[i] = nn.functional.interpolate(x[i], x[0].shape[2:], mode="bilinear")
        sample_points = self.sample_points[:x[0].shape[0]]

        yp = torch.zeros((x[0].shape[0], *self.output_shape), dtype=torch.float32, device=x[0].device)
        for i, tree in enumerate(self.trees):
            z = tree.render(x, z, yp, sample_points)
        
        yt_onehot = nn.functional.one_hot(yt, num_classes=self.output_shape[0])
        region_dists = [tree.get_region_distributions(yt_onehot, ignore_class=ignore_index) for tree in self.trees]
        if hasattr(self, "ce_loss_func"):
            ce_loss = self.ce_loss_func([p for p,_ in region_dists], weight, ignore_index)
        else:
            ce_loss = ce_loss_func(yp, yt, weight=weight, ignore_index=ignore_index)
                
        if len(region_visualization_path):
            self.visualize_regions(x_vis, yp, yt, lut, region_dists, region_visualization_path)
            
        loss = [(
            w * self.dist_func(p).mean(),
            w * torch.maximum(self.min_region_size-s, torch.zeros(s.shape,device=s.device)).mean(),
            w * self.dist_func(tree.region_map.permute(0,2,3,1).clone()).mean(),
            #w * tree.cov.abs().mean(),
            w * functools.reduce(lambda x,y: x+y, (*tree.decoder0[0][0].loss, *tree.decoder1[0][0].loss))
        ) for i,(w,tree,(p,s)) in enumerate(zip(self.tree_weights,self.trees,region_dists))]
        loss = functools.reduce(lambda xs,ys: tuple([x+y for x,y in zip(xs,ys)]), loss)
            
        # loss[0] -> region distribution
        # loss[1] -> minimum region size
        # loss[2] -> region map distribution
        # loss[3] -> vector quantization
            
        loss = self.loss_weights[0]*ce_loss + \
            self.loss_weights[1]*loss[0] + \
            self.loss_weights[2]*loss[1] + \
            self.loss_weights[3]*loss[2] + \
            self.loss_weights[4]*loss[3]
        
        return yp, loss
    
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

    def leaf_loss(self, yts, apply_argmax, apply_detach, class_weights, ignore_class):
        loss = []
        for tree_weight, tree, yt in zip(self.tree_weights, self.trees, yts):
            yp = tree.tree_parameters[1]
            yp = yp.reshape(yp.shape[0]*yt.shape[0], yp.shape[1]//yt.shape[0], *yp.shape[2:4])
            if tree.config.outputs.shape[0] != self.output_shape[0]:
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
            if ignore_class >= 0:
                ce[:,ignore_class] *= 0
                    
            loss.append(tree_weight * ce.sum(1).mean())
        return functools.reduce(lambda x,y: x+y, loss)
            
    @staticmethod
    def dist_metric_entropy(p):
        p[p<10**-6] = 10**-6
        return -(p * p.log()).sum(-1)
                
    @staticmethod
    def dist_metric_gini(p):
        return 1 - (p**2).sum(-1)
    
    def visualize_regions(self, x_vis, yp, yt, lut, region_dists, path):
        with torch.no_grad():
            yp = yp.detach().argmax(1).cpu().numpy()
            yt = yt.cpu().numpy()
            images = [(self.tree_weights[i] * self.dist_func(p)).detach() for i,(p,s) in enumerate(region_dists)]
            images = [img.repeat_interleave(self.trees[i].downsampling_factor,3) for i, img in enumerate(images)]
            images = [img.repeat_interleave(self.trees[i].downsampling_factor,2) for i, img in enumerate(images)]
            region_maps = [tree.region_map.clone().detach().permute(1, 0, 2, 3) for tree in self.trees]
            images = [img*region_map for img, region_map in zip(images, region_maps)]
            for img in images:
                for i in range(1, img.shape[0]):
                    img[0] += img[i]
            region_loss_images = [img[0].cpu().numpy() for img in images]
            
            region_images = []
            for _, region_map in enumerate(region_maps):
                region_map = region_map.cpu().numpy()
                regions = [utils.hsv2rgb(i/region_map.shape[0],1,1) for i in range(region_map.shape[0])]
                regions = [np.expand_dims(np.asarray(region), (1,2)) for region in regions]
                regions = [[region*region_map[i,j] for i, region in enumerate(regions)] for j in range(region_map.shape[1])]
                regions = [functools.reduce(lambda x,y: x+y, r) for r in regions]
                regions = np.asarray(regions, dtype=np.uint8)
                region_images.append(np.moveaxis(regions, 1, 3))
        
        import matplotlib.pyplot as plt
        base_size = 4
        n = 3 + 2*len(self.trees)
        fig, axes = plt.subplots(x_vis.shape[0], n, figsize=(n*base_size, x_vis.shape[0]*base_size))
        for i in range(x_vis.shape[0]):
            axes[i,0].imshow(np.moveaxis(x_vis[i], 0, 2))
            axes[i,1].imshow(lut[yt[i]])
            axes[i,2].imshow(lut[yp[i]])
            for j in range(len(self.trees)):
                fig.colorbar(axes[i,3+2*j].imshow(region_loss_images[j][i]), ax=axes[i,3+2*j])
                axes[i,4+2*j].imshow(region_images[j][i])
            if i == 0:
                for j in range(n):
                    if j < 3:
                        axes[i,j].set_title(("input", "ground truth", "prediction")[j])
                    else:
                        k = j - 3
                        k = k//2, ("region loss", "regions")[k%2]
                        axes[i,j].set_title(f"tree {k[0]}: {k[1]}")
            for j in range(n):
                axes[i,j].set_xticks(())
                axes[i,j].set_yticks(())
        fig.tight_layout()
        fig.savefig(f"{path}.pdf")
        plt.close(fig)
