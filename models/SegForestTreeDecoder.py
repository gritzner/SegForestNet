import core
import torch
import torch.nn as nn
import utils


class TreeFeatureDecoder(nn.ModuleList):
    def __init__(self, config, encoder, num_input_features, num_output_features, effective_num_input_features = -1, is_shape_decoder = False):
        super().__init__()
        self.use_residuals = config.decoder.use_residual_blocks
        c = (config.decoder.context, 2*config.decoder.context + 1)
        f = config.decoder.intermediate_features
        assert type(c[0]) == int and 0 <= c[0]
        
        if effective_num_input_features < 0:
            effective_num_input_features = num_input_features
        if effective_num_input_features >= num_output_features:
            print(f"[WARNING] features for tree parameter prediction not bottlenecked! (input: {effective_num_input_features}, output: {num_output_features})")
        
        vq_config = config.decoder.vq.__dict__.copy()
        vq_config["type"] = vq_config["type"][0 if is_shape_decoder else 1]
        if num_input_features == effective_num_input_features:
            vq_layer = utils.create_vector_quantization_layer(feature_size=num_input_features, **vq_config)
        else:
            assert num_input_features > effective_num_input_features
            
            class PartialVectorQuantization(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.vq_layer = utils.create_vector_quantization_layer(feature_size=effective_num_input_features, **vq_config)
                    self.num_other_features = num_input_features - effective_num_input_features
                    
                def prepare_for_epoch(self, epoch, epochs):
                    self.vq_layer.prepare_for_epoch(epoch, epochs)
                    
                def forward(self, x):
                    x = x[:,:self.num_other_features], self.vq_layer(x[:,self.num_other_features:])
                    self.loss = self.vq_layer.loss
                    return torch.cat(x, dim=1)
            
            vq_layer = PartialVectorQuantization()
                
        self.append(
            nn.Sequential(
                vq_layer,
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
    

def BSPSegNetDecoder(config, encoder, num_input_features, num_output_features, effective_num_input_features = -1, is_shape_decoder = False):
    features = [num_input_features]
    features.extend(config.decoder.intermediate_features)
    
    decoder = [nn.Sequential(utils.create_vector_quantization_layer(type=0))]
    for i in range(len(features)-1):
        decoder.extend([
            nn.Conv2d(features[i], features[i+1], 1),
            encoder.normalization(features[i+1]),
            encoder.activation(features[i+1])
        ])
    decoder.append(nn.Conv2d(features[-1], num_output_features, 1))
    
    return nn.Sequential(*decoder).to(core.device)
