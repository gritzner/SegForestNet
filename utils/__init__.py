from .modelfitter import ModelFitter
from .lrscheduler import LearningRateScheduler
from .confusionmatrix import ConfusionMatrix
from .vectorquantization import create_vector_quantization_layer
from .sam import SAM
import numpy as np
import torch
import torch.nn as nn
import types


def hsv2rgb(h, s, v):
    def f(n):
        k = (n + 6*h) % 6
        return v - v*s*max(0,min(k,4-k,1))
    return round(255*f(5)), round(255*f(3)), round(255*f(1))


def relu_wrapper(relu_type):
    if type(relu_type) == list:
        relu_params = relu_type[1:]
        relu_type = relu_type[0]
    else:
        relu_params = []
    relu_func = getattr(nn, relu_type)
    if relu_type == "PReLU" and len(relu_params) > 0 and relu_params[0] != 1:
        return lambda x: relu_func(x)
    else:
        return lambda _: relu_func(*relu_params)

    
def norm_wrapper(norm_type):
    if norm_type == "LayerNormNd":
        return lambda x: nn.GroupNorm(1, x)
    elif norm_type == "PixelNorm2d":
        class PixelNorm2d(nn.Module):
            def __init__(self, k):
                super().__init__()
                self.alpha = nn.Parameter(torch.ones((k, 1, 1)))
                self.beta = nn.Parameter(torch.zeros((k, 1, 1)))
            
            def forward(self, x):
                d = x - x.mean(1, keepdim=True)
                s = d.pow(2).mean(1, keepdim=True)
                x = d / torch.sqrt(s + 10**-6)
                return self.alpha * x + self.beta
            
        return PixelNorm2d
    return getattr(nn, norm_type)

def optim_wrapper(optim_type):
    if optim_type == "SAM":
        return SAM
    else:
        return getattr(torch.optim, optim_type)
    
def get_mini_batch_iterator(mini_batch_size, return_indices=False):
    class MiniBatchIterator():
        def __init__(self):
            self.mini_batch_size = mini_batch_size
            self.return_indices = return_indices
        
        def __call__(self, data, *data2):
            if len(data2) > 0:
                for i in data2:
                    assert data.shape[0] == i.shape[0]
            with torch.no_grad():
                for j in range(0, data.shape[0], self.mini_batch_size):
                    batch_data = data[j:j+self.mini_batch_size]
                    if len(data2) == 0 and not self.return_indices:
                        yield batch_data
                    else:
                        if self.return_indices:
                            batch_data = [np.asarray(range(j,j+self.mini_batch_size))[:batch_data.shape[0]], batch_data]
                        else:
                            batch_data = [batch_data]
                        batch_data.extend([
                            i[j:j+self.mini_batch_size] for i in data2
                        ])
                        yield tuple(batch_data)
        
    return MiniBatchIterator()

def get_scheduler(config):
    if isinstance(config, dict):
        config = types.SimpleNamespace(**config)
    
    class Scheduler():
        def __init__(self):
            params = ",".join(config.parameters)
            self.func = eval(f"lambda {params}: {config.func}")
            self.value_range = config.value_range
            self.value = self.value_range[0]
        
        def __call__(self, *params):
            self.value = self.func(*params)
            self.value = (1-self.value)*self.value_range[0] + self.value*self.value_range[1]
        
    return Scheduler()
