import core
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu, leaky_relu
import types
import functools
from .SegForestComponents import *
from .SegForestTreeDecoder import TreeFeatureDecoder, BSPSegNetDecoder

    
class PartitioningTree():
    def __init__(self, config, params, encoder, tree):
        self.config = tree
        self.downsampling_factor = encoder.downsampling_factor
        self.region_map_rendering = types.SimpleNamespace(
            func = getattr(PartitioningTree, f"render_region_map_{config.region_map.accumulation}"),
            node_weight = config.region_map.node_weight
        )
        if config.region_map.accumulation == "mul2":
            self.region_map_rendering.node_weight[0] = getattr(
                PartitioningTree,
                f"distance_transform_{self.region_map_rendering.node_weight[0]}"
            )
            if self.region_map_rendering.node_weight[3]:
                self.region_map_rendering_params = nn.Parameter(torch.as_tensor(
                    self.region_map_rendering.node_weight[1],
                    dtype=torch.float32, device=core.device
                ))
                self.region_map_rendering.node_weight[1] = self.region_map_rendering_params
        self.per_region_outputs = tree.outputs.shape[0] if len(tree.classifier) == 0 else tree.classifier[0]
        
        self.inner_nodes = []
        num_params = 0
        self.num_leaf_nodes = 0

        queue = [eval(tree.graph)]
        while len(queue) > 0:
            node = queue.pop(0)
            assert config.region_map.accumulation != "mul2" or type(node) == BSPNode
            self.inner_nodes.append(node)
            num_params += node.num_params
            for i, child in enumerate(node.children):
                if child == Leaf:
                    node.children[i] = Leaf(self.num_leaf_nodes)
                    self.num_leaf_nodes += 1
                else:
                    queue.append(child)
        self.region_map_rendering.base_shape = (self.num_leaf_nodes, *params.input_shape[1:])
        
        num_params = max(num_params, 1)
        self.num_tree_parameters = num_params + self.num_leaf_nodes * tree.outputs.shape[0]

        for node in reversed(self.inner_nodes):
            node.indices = []
            for child in node.children:
                if type(child) == Leaf:
                    node.indices.append(np.asarray([child.index], dtype=np.int32))
                else:
                    node.indices.append(np.concatenate(child.indices))
            node.children = tuple(node.children)
            node.indices = tuple(node.indices)
        self.inner_nodes = tuple(self.inner_nodes)
        
        decoder_factory = globals()[getattr(config.decoder, "type", "TreeFeatureDecoder")]
        self.decoder0 = decoder_factory(
            config, encoder, tree.actual_num_features.shape, num_params, is_shape_decoder = True
        )
        assert tree.shape_to_content in (0, 1, 2)
        num_params = (0, tree.actual_num_features.shape, num_params)[tree.shape_to_content]
        self.decoder1 = decoder_factory(
            config, encoder, tree.actual_num_features.content + num_params,
            self.num_leaf_nodes * self.per_region_outputs, tree.actual_num_features.content
        )
        
        if len(tree.classifier) > 0:
            num_features = tree.classifier.copy()
            if tree.classifier_skip_from == 0:
                num_features[0] += params.input_shape[0]
            elif tree.classifier_skip_from > 0:
                num_features[0] += encoder.num_features[tree.classifier_skip_from-1]
            self.classifier = []
            for in_features, out_features in zip(num_features[:-1], num_features[1:]):
                if tree.classifier_context > 0:
                    self.classifier.append(encoder.padding(tree.classifier_context))
                self.classifier.extend([
                    nn.Conv2d(in_features, out_features, 1+2*tree.classifier_context),
                    encoder.normalization(out_features),
                    encoder.activation(out_features)
                ])
            if tree.classifier_context > 0:
                self.classifier.append(encoder.padding(tree.classifier_context))
            self.classifier.append(nn.Conv2d(num_features[-1], tree.outputs.shape[0], 1+2*tree.classifier_context))
            self.classifier = nn.Sequential(*self.classifier).to(core.device)

    def render(self, x, z, y, sample_points):
        # seperate features into shape and content and decode them seperately
        shape, content, z = (
            z[:,:self.config.num_features.shape],
            z[:,-self.config.num_features.content:],
            z[:,self.config.num_features.shape:-self.config.num_features.content]
        )
        if self.config.shape_to_content > 0:
            self.tree_parameters = (self.decoder0(shape), None) # delay decoding
            if self.config.shape_to_content == 1:
                self.tree_parameters = (
                    self.tree_parameters[0],
                    self.decoder1(torch.cat((self.decoder0[0][0](shape).detach(), content), dim=1))
                )
            elif self.config.shape_to_content == 2:
                self.tree_parameters = (
                    self.tree_parameters[0],
                    self.decoder1(torch.cat((self.tree_parameters[0].detach(), content), dim=1))
                )
        else:
            self.tree_parameters = (self.decoder0(shape), self.decoder1(content))

        #d_shape = shape - shape.mean(dim=(0, 2, 3))[None,:,None,None]
        #d_content = content - content.mean(dim=(0, 2, 3))[None,:,None,None]
        #self.cov = d_shape[:,:,None,:,:] * d_content[:,None,:,:,:]
        #self.cov = self.cov.mean(dim=(0, 3, 4))
        
        # process shape features
        shape = self.tree_parameters[0].repeat_interleave(self.downsampling_factor, 3)
        shape = shape.repeat_interleave(self.downsampling_factor, 2)

        # render partitioning tree
        self.region_map = self.region_map_rendering.func(
            (x[0].shape[0], *self.region_map_rendering.base_shape), self.region_map_rendering.node_weight,
            self.region_map_rendering.softmax_temperature, shape, self.inner_nodes, sample_points,
        )
        
        # process content features
        content = self.tree_parameters[1].reshape(z.shape[0], self.num_leaf_nodes, self.per_region_outputs, *z.shape[2:])
        content = content.repeat_interleave(self.downsampling_factor, 4).repeat_interleave(self.downsampling_factor, 3)
        for i in range(self.num_leaf_nodes): # update prediction
            if hasattr(self, "classifier"):
                if self.config.classifier_skip_from >= 0:
                    logits = self.classifier(torch.cat((x[self.config.classifier_skip_from], content[:,i]), dim=1))
                else:
                    logits = self.classifier(content[:,i])
            else:
                logits = content[:,i]
            y[:,self.config.outputs] += self.region_map[:,i].unsqueeze(1) * logits
        
        return z
    
    @staticmethod
    def render_region_map_add(region_map_shape, node_weight, softmax_temperature, shape, nodes, sample_points):
        region_map = torch.zeros(region_map_shape, dtype=torch.float32, device=sample_points.device)
        for node in nodes:
            node_params, shape = shape[:,:node.num_params], shape[:,node.num_params:]
            node.apply_add(region_map, node_params, sample_points, node_weight)
        region_map = region_map / softmax_temperature
        return region_map.softmax(1)
    
    @staticmethod
    def render_region_map_mul(region_map_shape, node_weight, softmax_temperature, shape, nodes, sample_points):
        region_map = torch.ones(region_map_shape, dtype=torch.float32, device=sample_points.device)
        for node in nodes:
            node_params, shape = shape[:,:node.num_params], shape[:,node.num_params:]
            node.apply_mul(region_map, node_params, sample_points, node_weight)
        return region_map / region_map.sum(1, keepdim=True) 
    
    @staticmethod
    def render_region_map_mul2(region_map_shape, node_weight, softmax_temperature, shape, nodes, sample_points):
        region_map = torch.empty(region_map_shape, dtype=torch.float32, device=sample_points.device)
        distance_maps = torch.empty((region_map_shape[0], 2, len(nodes), *region_map_shape[2:]), dtype=torch.float32, device=sample_points.device)
        
        transform_func, transform_weights, leaky_slope = node_weight[:3]
        for i, node in enumerate(nodes):
            node_params, shape = shape[:,:node.num_params], shape[:,node.num_params:]
            distance_maps[:,0,i] = node.sdf.compute(node_params, sample_points)
        distance_maps[:,1] = -distance_maps[:,0]
        distance_maps = transform_func(
            transform_weights[0] * distance_maps + transform_weights[1],
            transform_weights[2], leaky_slope
        )
        
        for i in range(region_map_shape[1]):
            distances = [distance_maps[:,0,j] for j,node in enumerate(nodes) if i in node.indices[0]]
            distances.extend([distance_maps[:,1,j] for j,node in enumerate(nodes) if i in node.indices[1]])
            region_map[:,i] = functools.reduce(lambda x,y: x*y, distances)
        
        region_map = region_map / softmax_temperature
        return region_map.softmax(1)
    
    @staticmethod
    def distance_transform_relu(distance_maps, weight, leaky_slope):
        return weight * relu(distance_maps)
    
    @staticmethod
    def distance_transform_leaky_relu(distance_maps, weight, leaky_slope):
        return weight * leaky_relu(distance_maps, negative_slope=leaky_slope)
    
    @staticmethod
    def distance_transform_sigmoid(distance_maps, weight, leaky_slope):
        return weight * torch.sigmoid(distance_maps)
    
    @staticmethod
    def distance_transform_clamp(distance_maps, weight, leaky_slope):
        return weight * torch.clamp(distance_maps, 0, 1)
    
    @staticmethod
    def distance_transform_leaky_clamp(distance_maps, weight, leaky_slope):
        distance_maps[distance_maps<0] *= leaky_slope
        i = distance_maps>1
        distance_maps[i] = 1 + (distance_maps[i] - 1) * leaky_slope
        return weight * distance_maps
    
    @staticmethod
    def distance_transform_smoothstep(distance_maps, weight, leaky_slope):
        distance_maps = torch.clamp(distance_maps, 0, 1)
        distance_maps = 3 * distance_maps**2 - 2 * distance_maps**3
        return weight * distance_maps
    
    @staticmethod
    def distance_transform_leaky_smoothstep(distance_maps, weight, leaky_slope):
        distance_maps[distance_maps<0] *= leaky_slope
        i = distance_maps>1
        distance_maps[i] = 1 + (distance_maps[i] - 1) * leaky_slope
        distance_maps = 3 * distance_maps**2 - 2 * distance_maps**3
        return weight * distance_maps
    
    @staticmethod
    def render_region_map_old(region_map_shape, node_weight, softmax_temperature, shape, nodes, sample_points):
        region_map = torch.ones(region_map_shape, dtype=torch.float32, device=sample_points.device)
        for node in nodes:
            node_params, shape = shape[:,:node.num_params], shape[:,node.num_params:]
            node.apply_old(region_map, node_params, sample_points, node_weight)
        region_map = region_map / softmax_temperature
        return region_map.softmax(1)
    
    def get_region_distributions(self, yt, ignore_class=-1, instances=False):
        assert ignore_class >= 0 or instances
        
        if yt.shape[-1] != self.config.outputs.shape[0] and not instances:
            temp = torch.zeros([*yt.shape[:-1], self.config.outputs.shape[0]+1], dtype=yt.dtype, device=core.device)
            temp[:,:,:,:-1] = yt[:,:,:,self.config.outputs]
            yt = temp
            i = torch.nonzero(yt.sum(-1)==0, as_tuple=True)
            yt[i[0],i[1],i[2],-1] = 1
        
        downsampled_shape = (
            yt.shape[0],
            yt.shape[1] // self.downsampling_factor, self.downsampling_factor,
            yt.shape[2] // self.downsampling_factor, self.downsampling_factor,
            yt.shape[3]
        )
        
        p = torch.empty((self.num_leaf_nodes, *downsampled_shape[:2], downsampled_shape[3], yt.shape[3]), dtype=torch.float32, device=yt.device)
        s = torch.empty((*p.shape[:-1],), dtype=torch.float32, device=yt.device)
        
        for i in range(self.num_leaf_nodes):
            p[i] = (yt * self.region_map[:,i,:,:,None]).reshape(*downsampled_shape).sum(-2).sum(-3)
            s[i] = p[i].sum(-1)
            s[i][s[i]<10**-6] = 10**-6
            p[i] = p[i].clone() / s[i].unsqueeze(-1).clone()

        if self.config.outputs.shape[0] > 1 and ignore_class in self.config.outputs and not instances:
            index = np.nonzero(self.config.outputs==ignore_class)[0][0]
            p[:,:,:,:,index] = 0
            i = p.sum(-1) < 10**-6
            j = p[i]
            j[:,0 if index>0 else 1] = 1
            p[i] = j
            p = p / p.sum(-1).unsqueeze(-1)
            
        return p, s
