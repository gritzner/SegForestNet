import numpy as np
import core
import glob
import PIL.Image
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
        root_path = "/data/vehicle_classification/datasets"
        dota_root_path = f"{root_path}/DOTA/v10"
        isaid_root_path = f"{root_path}/iSAID"

        class_map = np.empty((len(lut), 3), dtype=np.uint8)
        for i, rgb in enumerate(lut):
            class_map[i] = rgb
        
        self.images = []
        self.image_subsets = []
        for subset, subset_id in (("training", 0), ("validation", 2)):
            path_prefix = f"{isaid_root_path}/{subset}"
            for img_set in core.thread_pool.map(lambda fn: DatasetLoader_isaid.load_image(fn,path_prefix,class_map), sorted(glob.iglob(f"{dota_root_path}/{subset}/tmp/images/*.png"))):
                self.images.append(img_set)
                self.image_subsets.append(subset_id)
                
        self.channels = {
            "red": (0, 0),
            "green": (0, 1),
            "blue": (0, 2),
            "gt": (1, 0),
            "instances": (1, 0)
        }
        self.num_classes = 16
        self.instances = {
            1: "storage_tank",
            2: "large_vehicle",
            3: "small_vehicle",
            4: "plane",
            5: "ship",
            6: "swimming_pool",
            7: "harbor",
            8: "tennis_court",
            9: "ground_track_field",
            10: "soccer_ball_field",
            11: "baseball_diamond",
            12: "bridge",
            13: "basketball_court",
            14: "roundabout",
            15: "helicopter"
        }
        self.gsd = 26 # GSD varies wildy in DOTA, median of all valid GSDs: 26, mean: 39
        self.lut = lut

    @staticmethod
    def load_image(fn, path_prefix, class_map):
        img_set = []
    
        img = PIL.Image.open(fn)
        if len(img.getbands()) == 1:
            img = img.convert("RGB")
        img_set.append(np.asarray(img))
    
        img_id = fn.split("/")[-1].split(".")[0]
        sem_img = PIL.Image.open(f"{path_prefix}/Semantic_masks/tmp/images/{img_id}_instance_color_RGB.png")
        inst_img = PIL.Image.open(f"{path_prefix}/Instance_masks/tmp/images/{img_id}_instance_id_RGB.png")                
        img_set.append(SparseInstanceImage(sem_img, inst_img, class_map))

        return img_set
