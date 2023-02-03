import core
import numpy as np
import torch
import PIL.Image
import PIL.ImageDraw
import itertools
import types


class PseudoClassificationDataset():
    def __init__(self, config, params):
        print("generating dataset ...")
        
        n = config.samples_per_class.training + config.samples_per_class.validation + config.samples_per_class.test
        xs = np.empty((4, n, 3, *config.input_shape), dtype=np.float32)
        ys = np.empty((4, n), dtype=np.int64)
        
        colors = (
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 255)
        )
        rng = np.random.RandomState(core.random_seeds[config.seed])
        
        for c, i in itertools.product(range(4), range(n)):
            img = PIL.Image.new("RGB", (config.input_shape[-1], config.input_shape[-2]), (0, 0, 0))
            draw = PIL.ImageDraw.Draw(img)
            for _ in range(config.objects_per_sample):
                col = colors[int(len(colors) * rng.rand())]
                x0 = int(config.input_shape[-1] * rng.rand())
                y0 = int(config.input_shape[-2] * rng.rand())
                x1 = x0 + int((config.input_shape[-1]-x0) * rng.rand())
                y1 = y0 + int((config.input_shape[-2]-y0) * rng.rand())
                if c == 0:
                    draw.ellipse((x0, y0, x1, y1), outline=col, width=config.outline_width)
                elif c == 1:
                    draw.ellipse((x0, y0, x1, y1), fill=col)
                elif c == 2:
                    draw.rectangle((x0, y0, x1, y1), outline=col, width=config.outline_width)
                else:
                    draw.rectangle((x0, y0, x1, y1), fill=col)

            xs[c,i] = (np.moveaxis(np.asarray(img), -1, 0) - 127.5) / 127.5
            ys[c,i] = c
            
        n = (config.samples_per_class.training, config.samples_per_class.training+config.samples_per_class.validation)
        self.training = types.SimpleNamespace(
            x = np.concatenate([xs[c,:n[0]] for c in range(4)], axis=0),
            y = np.concatenate([ys[c,:n[0]] for c in range(4)], axis=0),
        )
        self.training.x_vis = self.training.x
        
        self.validation = types.SimpleNamespace(
            x = np.concatenate([xs[c,n[0]:n[1]] for c in range(4)], axis=0),
            y = np.concatenate([ys[c,n[0]:n[1]] for c in range(4)], axis=0),
        )
        self.validation.x_vis = self.validation.x
        
        self.test = types.SimpleNamespace(
            x = np.concatenate([xs[c,n[1]:] for c in range(4)], axis=0),
            y = np.concatenate([ys[c,n[1]:] for c in range(4)], axis=0),
        )
        self.test.x_vis = self.test.x
        
        self.num_classes = 4
        self.class_weights = torch.ones(4, dtype=torch.float32, device=core.device)
        self.ignore_class = -100
            
        print(f"# of training samples: {self.training.x.shape[0]}")
        print(f"# of validation samples: {self.validation.x.shape[0]}")
        print(f"# of test samples: {self.test.x.shape[0]}")
        print(f"input shape: {self.training.x.shape[1:]}")
        print(f"# of classes: {self.num_classes}")        
