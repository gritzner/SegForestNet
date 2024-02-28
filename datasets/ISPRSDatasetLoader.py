import glob
import cv2 as cv
import numpy as np


# ISPRS classes:
#  (255,255,255), 0: impervious surfaces
#  (  0,  0,255), 1: buildings
#  (  0,255,255), 2: low vegetation
#  (  0,255,  0), 3: trees
#  (255,255,  0), 4: cars
#  (255,  0,  0), 5: clutter/background

rgb2class_map = {
    (255,255,255): 0,
    (  0,  0,255): 1,
    (  0,255,255): 2,
    (  0,255,  0): 3,
    (255,255,  0): 4,
    (255,  0,  0): 5
}

def rgb2class(fn):
    gt = cv.imread(fn, cv.IMREAD_UNCHANGED)
    new_gt = np.empty(gt.shape[:2], dtype=np.uint8)
    for k, v in rgb2class_map.items():
        i = np.logical_and(gt[:,:,0]==k[2], gt[:,:,1]==k[1])
        i = np.logical_and(i, gt[:,:,2]==k[0])
        i = np.where(i)
        new_gt[i[0],i[1]] = v
    return new_gt


lut = ( # for ground truth visualization
    (  0,  0,  0), # sealed surface
    (  0,  0,255), # building
    (  0,255,  0), # low vegetation
    (255,  0,  0), # tree
    (255,255,  0), # car
    (255,255,255)  # clutter
)


class DatasetLoader_vaihingen():
    def __init__(self, config):
        filenames = [fn.split("/")[-1] for fn in glob.iglob(f"{config.dataset_path}/ground_truth_COMPLETE/*.tif")]
            
        self.images = []
        self.image_subsets = []
        for fn in sorted(filenames):
            i = int(fn.split(".")[0].split("a")[-1])
            if i in (1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37):
                self.image_subsets.append(0)
            else:
                self.image_subsets.append(2)
            
            dsm_fn = fn.replace("top_mosaic_09cm", "dsm_09cm_matching")
            img_set = []
            img_set.append(cv.imread(
                f"{config.dataset_path}/semantic_labeling/top/{fn}",
                cv.IMREAD_UNCHANGED
            ))
            img_set.append(cv.imread(
                f"{config.dataset_path}/semantic_labeling/dsm/{dsm_fn}",
                cv.IMREAD_UNCHANGED
            ))
            img_set.append(rgb2class(f"{config.dataset_path}/ground_truth_COMPLETE/{fn}"))
            self.images.append(img_set)
            
        self.channels = {
            "ir": (0, 2),
            "red": (0, 1),
            "green": (0, 0),
            "depth": (1, 0),
            "gt": (2, 0)
        }
        self.num_classes = 6
        self.gsd = 8 # filenames say 9cm, but all the accompanying documents say 8cm
        self.lut = lut

class DatasetLoader_potsdam():
    def __init__(self, config):
        filenames = [fn.split("/")[-1] for fn in glob.iglob(f"{config.dataset_path}/5_Labels_all/*.tif")]
        filenames = list(map(lambda fn: fn[4:fn.rfind("_")], filenames))
        
        self.images = []
        self.image_subsets = []
        for fn in sorted(filenames):
            i = 100*int(fn[8]) + int(fn[10:])
            if i in (210, 211, 212, 310, 311, 312, 410, 411, 412, 510, 511, 512, 607, 608, 609, 610, 611, 612, 707, 708, 709, 710, 711, 712):
                self.image_subsets.append(0)
            else:
                self.image_subsets.append(2)
            
            dsm_fn = fn.replace("potsdam_", "dsm_potsdam_0")
            dsm_fn = dsm_fn.replace("_7", "_07")
            dsm_fn = dsm_fn.replace("_8", "_08")
            dsm_fn = dsm_fn.replace("_9", "_09")
            img_set = []
            img_set.append(cv.imread(
                f"{config.dataset_path}/2_Ortho_RGB/top_{fn}_RGB.tif",
                cv.IMREAD_UNCHANGED
            ))
            img_set.append(cv.imread(
                f"{config.dataset_path}/3_Ortho_IRRG/top_{fn}_IRRG.tif",
                cv.IMREAD_UNCHANGED
            ))
            img_set.append(cv.imread(
                f"{config.dataset_path}/1_DSM/{dsm_fn}.tif",
                cv.IMREAD_UNCHANGED
            ))
            if "3_13" in fn:
                img_set[-1] = cv.resize(img_set[-1], (6000,6000), interpolation=cv.INTER_CUBIC)
            if "_4_12" in fn or "6_7" in fn:
                img_set.append(rgb2class(f"{config.dataset_path}/5_Labels_for_participants/top_{fn}_label.tif"))
            else:
                img_set.append(rgb2class(f"{config.dataset_path}/5_Labels_all/top_{fn}_label.tif"))
                
            self.images.append(img_set)
            
        self.channels = {
            "ir": (1, 2),
            "red": (0, 2),
            "green": (0, 1),
            "blue": (0, 0),
            "depth": (2, 0),
            "gt": (3, 0)
        }
        self.num_classes = 6
        self.gsd = 5
        self.lut = lut
