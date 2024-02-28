import glob
import cv2 as cv
import numpy as np


lut = ( # for ground truth visualization
    (  0,  0,  0), # other
    (255,255,255) # building
)


class SynthinelLoader():
    def __init__(self, config, subsets):
        self.images = []
        for subset in subsets:
            for fn in glob.iglob(f"{config.dataset_path}/random_city_s_{subset}_*_RGB.tif"):
                img_set = []
                img_set.append(cv.imread(fn, cv.IMREAD_UNCHANGED))
                fn = fn.split("_")
                fn[-1] = "GT.tif"
                fn = "_".join(fn)
                img_set.append(cv.imread(fn, cv.IMREAD_UNCHANGED))
                self.images.append(img_set)
            
        self.channels = {
            "red": (0, 2),
            "green": (0, 1),
            "blue": (0, 0),
            "gt": (1, 0)
        }
        self.num_classes = 2
        self.gsd = 30
        self.lut = lut

class DatasetLoader_synthinel(SynthinelLoader):
    def __init__(self, config):
        super().__init__(config, "abcghi")

class DatasetLoader_synthinel_redroof(SynthinelLoader):
    def __init__(self, config):
        super().__init__(config, "a")

class DatasetLoader_synthinel_paris(SynthinelLoader):
    def __init__(self, config):
        super().__init__(config, "b")

class DatasetLoader_synthinel_ancient(SynthinelLoader):
    def __init__(self, config):
        super().__init__(config, "c")

class DatasetLoader_synthinel_scifi(SynthinelLoader):
    def __init__(self, config):
        super().__init__(config, "d")

class DatasetLoader_synthinel_palace(SynthinelLoader):
    def __init__(self, config):
        super().__init__(config, "e")

class DatasetLoader_synthinel_austin(SynthinelLoader):
    def __init__(self, config):
        super().__init__(config, "g")

class DatasetLoader_synthinel_venice(SynthinelLoader):
    def __init__(self, config):
        super().__init__(config, "h")

class DatasetLoader_synthinel_modern(SynthinelLoader):
    def __init__(self, config):
        super().__init__(config, "i")
