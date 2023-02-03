import core
import numpy as np
import torch
import types
import rust


class ToyDataset():
    def __init__(self, config, params):
        print(f"generating toy dataset ...")
        
        generator = getattr(ToyDataset, f"{config.func.name}_generator")
        rng = np.random.RandomState(core.random_seeds[config.random_seed])
        colors = np.asarray((
            (0, 0, 0),
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (255, 255, 255)
        ), dtype=np.uint8)
        colors = np.ascontiguousarray(colors[rng.permutation(colors.shape[0])])
        
        self.lut = getattr(ToyDataset, f"{config.func.name}_generator_lut")(colors)
        self.num_classes = len(self.lut)
        self.class_weights = torch.ones(self.num_classes, dtype=torch.float32, device=core.device)
        self.ignore_class = -100
        
        self.training = self.generate_subset(generator, rng, colors, config.num_samples.training, config)
        self.validation = self.generate_subset(generator, rng, colors, config.num_samples.validation, config)
        self.test = self.generate_subset(generator, rng, colors, config.num_samples.test, config)

    def generate_subset(self, generator, rng, colors, num_samples, config):
        subset = types.SimpleNamespace(
            x_vis = np.empty((num_samples, *config.patch_size, 3), dtype=np.uint8),
            y = np.empty((num_samples, *config.patch_size), dtype=np.int32)
        )
        
        for i in range(num_samples):
            generator(subset.x_vis[i], subset.y[i], rng, colors, **config.func.params.__dict__)
            
        subset.x_vis = np.moveaxis(subset.x_vis, -1, 1)
        subset.x = (np.asarray(subset.x_vis, dtype=np.float32) / 127.5) - 1
        divider = self.num_classes / 2.0
        subset.x_gt = (np.asarray(subset.y, dtype=np.float32) / divider) - 1
        
        return subset
    
    @staticmethod
    def circles_generator(img, gt, rng, colors, num_circles, out_margin, min_radius, max_radius):
        img[:,:] = colors[0]
        gt[:] = 0
        for _ in range(num_circles):
            i = rng.randint(1, colors.shape[0])
            
            center = np.asarray((
                rng.randint(-out_margin, img.shape[0]+out_margin),
                rng.randint(-out_margin, img.shape[1]+out_margin),
            ), dtype=np.int32)
            r = min_radius + rng.rand() * (max_radius - min_radius)
            r = r, int(np.ceil(r)) + 1
            
            yrange = np.asarray((max(center[0]-r[1],0), min(center[0]+r[1],img.shape[0])), dtype=np.int32)
            xrange = np.asarray((max(center[1]-r[1],0), min(center[1]+r[1],img.shape[1])), dtype=np.int32)
            
            rust.draw_circle(img, gt, center, r[0], colors[i], i, yrange, xrange)
                    
    @staticmethod
    def circles_generator_lut(colors):
        return tuple([tuple(color) for color in colors])
    
    @staticmethod
    def quadtree_generator(img, gt, rng, colors, margin):
        i = rng.permutation(4)
        j = 2*i + rng.randint(2, size=4)
    
        y = rng.randint(margin, img.shape[0]-margin)
        x = rng.randint(margin, img.shape[1]-margin)
    
        img[:y,:x] = colors[j[0]]
        gt[:y,:x] = i[0]
        img[:y,x:] = colors[j[1]]
        gt[:y,x:] = i[1]
        img[y:,:x] = colors[j[2]]
        gt[y:,:x] = i[2]
        img[y:,x:] = colors[j[3]]
        gt[y:,x:] = i[3]
    
    @staticmethod
    def quadtree_generator_lut(colors):
        assert colors.shape[0] == 8
        return tuple([colors[2*i] for i in range(4)])
    
    @staticmethod
    def kdtree_generator(img, gt, rng, colors, margin):
        i = rng.permutation(4)
        j = 2*i + rng.randint(2, size=4)
        y = rng.randint(margin, img.shape[0]-margin)
        
        x = rng.randint(margin, img.shape[1]-margin)
        img[:y,:x] = colors[j[0]]
        gt[:y,:x] = i[0]
        img[:y,x:] = colors[j[1]]
        gt[:y,x:] = i[1]
        
        x = rng.randint(margin, img.shape[1]-margin)        
        img[y:,:x] = colors[j[2]]
        gt[y:,:x] = i[2]
        img[y:,x:] = colors[j[3]]
        gt[y:,x:] = i[3]
    
    @staticmethod
    def kdtree_generator_lut(colors):
        assert colors.shape[0] == 8
        return tuple([colors[2*i] for i in range(4)])
                