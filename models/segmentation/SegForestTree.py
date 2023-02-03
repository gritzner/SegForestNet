import core
import numpy as np
import torch
import torch.nn as nn
import types
from .SegForestComponents import *


class TreeFeatureDecoder(nn.ModuleList):
    def __init__(self, config, encoder, num_input_features, num_output_features, effective_num_input_features = -1):
        super().__init__()
        self.use_residuals = config.decoder.use_residual_blocks
        c = (config.decoder.context, 2*config.decoder.context + 1)
        f = config.decoder.intermediate_features
        assert type(c[0]) == int and 0 <= c[0]
        
        if effective_num_input_features < 0:
            effective_num_input_features = num_input_features
        if effective_num_input_features >= num_output_features:
            print(f"[WARNING] features for tree parameter prediction not bottlenecked! (input: {effective_num_input_features}, output: {num_output_features})")
        
        self.append(
            nn.Sequential(
                nn.Conv2d(num_input_features, f, 1),
                encoder.normalization(f)
            ).to(core.device)
        )
        
        for i in range(config.decoder.num_blocks):
            block = [encoder.activation(f)]
            if c[0] > 0:
                block.extend([
                    encoder.padding(c[0]),
                    nn.Conv2d(f, f, c[1], groups=f),
                    encoder.normalization(f),
                    encoder.activation(f)
                ])
            block.extend([
                nn.Conv2d(f, f, 1),
                encoder.normalization(f)
            ])
            self.append(
                nn.Sequential(*block).to(core.device)
            )
        
        self.append(
            nn.Sequential(
                encoder.activation(f),
                nn.Conv2d(f, num_output_features, 1)
            ).to(core.device)
        )
        
    def forward(self, x):
        for i, layer in enumerate(self):
            x = layer(x) if (i==0 or i+1 == len(self) or not self.use_residuals) else layer(x) + x
        return x
    

def BSPSegNetDecoder(config, encoder, num_input_features, num_output_features):
    features = [num_input_features]
    features.extend(config.decoder.intermediate_features)
    
    decoder = []
    for i in range(len(features)-1):
        decoder.extend([
            nn.Conv2d(features[i], features[i+1], 1),
            encoder.normalization(features[i+1]),
            encoder.activation(features[i+1])
        ])
    decoder.append(nn.Conv2d(features[-1], num_output_features, 1))
    
    return nn.Sequential(*decoder).to(core.device)

    
class PartitioningTree():
    def __init__(self, config, params, encoder, tree):
        self.config = tree
        self.downsampling_factor = encoder.downsampling_factor
        self.region_map_rendering = types.SimpleNamespace(
            func = getattr(PartitioningTree, f"render_region_map_{config.region_map.accumulation}"),
            node_weight = config.region_map.node_weight
        )
        self.region_encoder = params.region_encoder[0] if getattr(params, "region_encoder", False) else None
        self.per_region_outputs = tree.outputs.shape[0] + params.extra_payload
        
        self.inner_nodes = []
        num_params = 0
        self.num_leaf_nodes = 0

        queue = [eval(tree.graph)]
        while len(queue) > 0:
            node = queue.pop(0)
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
            config, encoder, tree.num_features.shape, num_params
        )
        if self.region_encoder:
            self.decoder1 = decoder_factory(
                config, encoder, tree.num_features.content + params.region_encoder[1], self.per_region_outputs, tree.num_features.content
            )
        else:
            self.decoder1 = decoder_factory(
                config, encoder, tree.num_features.content, self.num_leaf_nodes * self.per_region_outputs
            )

    def render(self, x, y, sample_points):
        # seperate features into shape and content and decode them seperately
        shape, content, x = (
            x[:,:self.config.num_features.shape],
            x[:,-self.config.num_features.content:],
            x[:,self.config.num_features.shape:-self.config.num_features.content]
        )
        if self.region_encoder:
            self.tree_parameters = (self.decoder0(shape), None) # delay decoding until after region map rendering
        else:
            self.tree_parameters = (self.decoder0(shape), self.decoder1(content))
        
        # process shape features
        shape = self.tree_parameters[0].repeat_interleave(self.downsampling_factor, 3)
        shape = shape.repeat_interleave(self.downsampling_factor, 2)

        # render partitioning tree
        self.region_map = self.region_map_rendering.func(
            (x.shape[0], *self.region_map_rendering.base_shape), self.region_map_rendering.node_weight,
            shape, self.inner_nodes, sample_points,
        )
        
        # process content features
        if self.region_encoder:
            regions = self.region_map.reshape(
                *self.region_map.shape[:2],
                self.region_map.shape[2] // self.downsampling_factor, self.downsampling_factor,
                self.region_map.shape[3] // self.downsampling_factor, self.downsampling_factor
            )
            regions = regions.permute(0, 1, 2, 4, 3, 5)
            regions = regions.reshape(-1, 1, self.downsampling_factor, self.downsampling_factor)
            
            regions = self.region_encoder(regions)
            regions = regions.reshape(
                *self.region_map.shape[:2],
                self.region_map.shape[2] // self.downsampling_factor,
                self.region_map.shape[3] // self.downsampling_factor,
                -1
            )
            regions = regions.permute(0, 1, 4, 2, 3)
            
            content = [self.decoder1(torch.cat((content,regions[:,i]),dim=1)) for i in range(self.region_map.shape[1])]
            content = torch.cat(content, dim=1)
            
            self.tree_parameters = (self.tree_parameters[0], content)

        content = self.tree_parameters[1]

        # update prediction
        for i in range(self.num_leaf_nodes):
            payload = content[:,i*self.per_region_outputs:(i+1)*self.per_region_outputs]
            class_logits, extra_payload = payload[:,:self.config.outputs.shape[0]], payload[:,self.config.outputs.shape[0]:]
            class_logits = class_logits.repeat_interleave(self.downsampling_factor, 3)
            class_logits = class_logits.repeat_interleave(self.downsampling_factor, 2)
            
            y[:,self.config.outputs] += self.region_map[:,i].unsqueeze(1) * class_logits
        
        return x
    
    @staticmethod
    def render_region_map_add(region_map_shape, node_weight, shape, nodes, sample_points):
        region_map = torch.zeros(region_map_shape, dtype=torch.float32, device=sample_points.device)
        for node in nodes:
            node_params, shape = shape[:,:node.num_params], shape[:,node.num_params:]
            node.apply_add(region_map, node_params, sample_points, node_weight)
        return region_map.softmax(1)
    
    @staticmethod
    def render_region_map_mul(region_map_shape, node_weight, shape, nodes, sample_points):
        region_map = torch.ones(region_map_shape, dtype=torch.float32, device=sample_points.device)
        for node in nodes:
            node_params, shape = shape[:,:node.num_params], shape[:,node.num_params:]
            node.apply_mul(region_map, node_params, sample_points, node_weight)
        return region_map / region_map.sum(1, keepdim=True) 
    
    @staticmethod
    def render_region_map_old(region_map_shape, node_weight, shape, nodes, sample_points):
        region_map = torch.ones(region_map_shape, dtype=torch.float32, device=sample_points.device)
        for node in nodes:
            node_params, shape = shape[:,:node.num_params], shape[:,node.num_params:]
            node.apply_old(region_map, node_params, sample_points, node_weight)
        return region_map.softmax(1)
    
    def get_region_distributions(self, yt):
        if yt.shape[-1] != self.config.outputs.shape[0]:
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
        return p, s
