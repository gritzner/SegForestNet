import glob
import cv2 as cv
import numpy as np


class IPILoader():
    def __init__(self, config):
        self.images = []
        for fn in glob.iglob(f"{config.dataset_path}/gt/*.tif"):
            img_set = []
            img_set.append(cv.imread(fn, cv.IMREAD_UNCHANGED))
            tokens = fn.split("/")
            tokens[-2] = "RGB"
            fn = "/".join(tokens)
            img_set.append(cv.imread(fn, cv.IMREAD_UNCHANGED))
            tokens[-2] = "IR"
            fn = "/".join(tokens)
            img_set.append(cv.imread(fn, cv.IMREAD_UNCHANGED))
            self.images.append(img_set)
            
        self.channels = {
            "gt": (0, 0),
            "red": (1, 2),
            "green": (1, 1),
            "blue": (1, 0),
            "ir": (2, 0)
        }
        self.num_classes = 8
        self.gsd = 20
        self.lut = (
            (255,128,  0), # building
            (128,128,128), # sealed surface
            (200,135, 70), # soil
            (  0,255,  0), # grass
            ( 64,128,  0), # tree
            (  0,  0,255), # water
            (255,  0,  0), # car
            (128,  0, 25)  # other
        )

class DatasetLoader_hameln(IPILoader):
    def __init__(self, config):
        super().__init__(config)

class DatasetLoader_schleswig(IPILoader):
    def __init__(self, config):
        super().__init__(config)

class DatasetLoader_mecklenburg_vorpommern(IPILoader):
    def __init__(self, config):
        super().__init__(config)
        
        self.num_classes = 10
        self.gsd = 10
        self.lut = (
            (255,128,  0), # building
            (128,128,128), # sealed surface
            (210,210,200), # unpaved road
            (200,135, 70), # soil
            (255,255,  0), # crops
            (  0,255,  0), # grass
            ( 64,128,  0), # tree
            (  0,  0,255), # water
            ( 64, 64, 64), # railway
            (128,  0, 25)  # other
        )

class IPIDALoader():
    def __init__(self, config):
        self.images = []
        for fn in sorted(glob.iglob(f"{config.dataset_path}/**/L_DA.png")):
            img_set = []
            img_set.append(cv.imread(fn, cv.IMREAD_UNCHANGED))
            fn = fn[:fn.rfind("/")]
            img_set.append(cv.imread(f"{fn}/R_G_B_IR.png", cv.IMREAD_UNCHANGED))
            img = cv.imread(f"{fn}/NDSM.tif", cv.IMREAD_UNCHANGED)
            img_set.append(np.asarray(img, dtype=np.float32))
            self.images.append(img_set)
            
        self.channels = {
            "gt": (0, 0),
            "ir": (1, 3),
            "red": (1, 2),
            "green": (1, 1),
            "blue": (1, 0),
            "depth": (2, 0)
        }
        self.num_classes = 6
        self.gsd = 20
        self.lut = ( # for ground truth visualization
            (  0,  0,  0), # sealed surface
            (  0,  0,255), # building
            (  0,255,  0), # low vegetation
            (255,  0,  0), # tree
            (255,255,  0), # car
            (255,255,255)  # clutter
        )

class DatasetLoader_hameln_DA(IPIDALoader):
    def __init__(self, config):
        super().__init__(config)

class DatasetLoader_schleswig_DA(IPIDALoader):
    def __init__(self, config):
        super().__init__(config)

