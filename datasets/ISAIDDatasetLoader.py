import numpy as np
import core
import glob
import cv2 as cv
from .SparseInstanceImage import SparseInstanceImage


lut = ( # for ground truth visualization
    (  0,  0,  0), # background
    (  0, 63, 63), # storange tank
    (  0,127,127), # large vehicle
    (  0,  0,127), # small vehicle
    (  0,127,255), # plane
    (  0,  0, 63), # ship
    (  0,  0,255), # swimming pool
    (  0,100,155), # harbor
    (  0, 63,127), # tennis court
    (  0, 63,255), # ground track field
    (  0,127,191), # soccer ball field
    (  0, 63,  0), # baseball diamond
    (  0,127, 63), # bridge
    (  0, 63,191), # basketball court
    (  0,191,127), # roundabout
    (  0,  0,191)  # helicopter
)


class DatasetLoader_isaid():
    def __init__(self, config):
        dota_root_path, isaid_root_path = config.dataset_path

        class_map = np.empty((len(lut), 3), dtype=np.uint8)
        for i, rgb in enumerate(lut):
            class_map[i] = rgb
        class_map = np.ascontiguousarray(np.flip(class_map,axis=1))
        
        self.images = []
        self.image_subsets = []
        for subset, subset_id in (("training", 0), ("validation", 2)):
            path_prefix = f"{isaid_root_path}/{subset}"
            for img_set in core.thread_pool.map(lambda fn: DatasetLoader_isaid.load_image(fn,path_prefix,class_map), sorted(glob.iglob(f"{dota_root_path}/{subset}/tmp/images/*.png"))):
                self.images.append(img_set)
                self.image_subsets.append(subset_id)
                
        self.channels = {
            "red": (0, 2),
            "green": (0, 1),
            "blue": (0, 0),
            "gt": (1, 0)
        }
        self.num_classes = 16
        self.gsd = 26 # GSD varies wildy in DOTA, median of all valid GSDs: 26, mean: 39
        self.lut = lut

    @staticmethod
    def load_image(fn, path_prefix, class_map):
        img_set = [cv.imread(fn)]
    
        img_id = fn.split("/")[-1].split(".")[0]
        sem_img = cv.imread(
            f"{path_prefix}/Semantic_masks/tmp/images/{img_id}_instance_color_RGB.png",
            cv.IMREAD_UNCHANGED
        )
        inst_img = cv.imread(
            f"{path_prefix}/Instance_masks/tmp/images/{img_id}_instance_id_RGB.png",
            cv.IMREAD_UNCHANGED
        )
        img_set.append(SparseInstanceImage(sem_img, inst_img, class_map))

        return img_set
