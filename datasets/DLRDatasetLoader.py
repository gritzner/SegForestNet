import glob
import cv2 as cv
import numpy as np
import os


class DatasetLoader_dlr_landcover():
    def __init__(self, config):
        self.images = []
        for city in ("Berlin", "Munich"):
            full_img = cv.imread(
                f"{config.dataset_path}/images/{city}_converted.tif",
                cv.IMREAD_UNCHANGED
            )
            assert full_img.shape[0]%5 == 0
            full_img = full_img.reshape((5, -1, full_img.shape[1]))
            full_img, full_sar = full_img[:4], full_img[-1]
            full_img = np.moveaxis(full_img, 0, 2)
            full_gt = cv.imread(
                f"{config.dataset_path}/annotations/{city}_converted.tif",
                cv.IMREAD_UNCHANGED
            )
            assert full_gt.shape == full_sar.shape
            
            ys = (full_sar.shape[0]//2, full_sar.shape[0])
            min_ys = (0, 0, ys[0], ys[0])
            max_ys = (ys[0], ys[0], ys[1], ys[1])
            
            xs = (full_sar.shape[1]//2, full_sar.shape[1])
            min_xs = (0, xs[0], 0, xs[0])
            max_xs = (xs[0], xs[1], xs[0], xs[1])
            
            for y0,y1,x0,x1 in zip(min_ys, max_ys, min_xs, max_xs):
                sar = full_sar[y0:y1,x0:x1]
                img = full_img[y0:y1,x0:x1]
                gt = full_gt[y0:y1,x0:x1]
            
                img_set = []            
                img_set.append(sar)
                img_set.append(img)
                
                rgb = np.empty((*sar.shape, 3), dtype=np.uint8)
                for c in range(3):
                    temp = img[:,:,c]
                    temp = (temp - np.min(temp)) / 3000
                    temp[temp>1] = 1
                    rgb[:,:,2-c] = 255 * temp
                img_set.append(rgb)
                
                img_set.append(gt)
                self.images.append(img_set)

        self.channels = {
            "sar": (0, 0),
            "blue": (1, 0),
            "green": (1, 1),
            "red": (1, 2),
            "ir": (1, 3),
            "vis_red": (2, 0),
            "vis_green": (2, 1),
            "vis_blue": (2, 2),
            "gt": (3, 0)
        }            
        self.num_classes = 5
        self.gsd = 1000
        self.lut = (
            (  0,  0,  0), # ??? might be roads/paths
            (255,255,  0), # agriculture
            (  0,255,  0), # forest
            (255,  0,  0), # built-up
            (  0,  0,255)  # water
        )

class DatasetLoader_dlr_roadmaps():
    def __init__(self, config):
        self.images = []
        for subset in ("AerialKITTI", "Bavaria"):
            for fn in glob.iglob(f"{config.dataset_path}/{subset}/images/*.jpg"):
                img_set = []
                img_set.append(cv.imread(fn, cv.IMREAD_UNCHANGED))
                fn = fn[:-4]
                mask = cv.imread(fn + "_mask.tif", cv.IMREAD_UNCHANGED)
                fn = fn.split("/")
                fn[-2] = "road_annotation_full"
                fn = "/".join(fn)
                if not(config.load_full_dlr_roadmaps_annotations and os.path.exists(fn + "_gt.tif")):
                    fn = fn.split("/")
                    fn[-2] = "road_annotation"
                    fn = "/".join(fn)
                raw_gt = cv.imread(fn + "_gt.tif", cv.IMREAD_UNCHANGED)
                if len(raw_gt.shape) == 3:
                    raw_gt = raw_gt[:,:,0]
                gt = np.ones(raw_gt.shape, dtype=np.uint8)
                gt[raw_gt==255] = 2
                gt[mask!=255] = 0
                img_set.append(gt)
                self.images.append(img_set)

        self.channels = {
            "red": (0, 2),
            "green": (0, 1),
            "blue": (0, 0),
            "gt": (1, 0)
        }            
        self.num_classes = 3
        self.gsd = 13
        self.lut = (
            (127,127,127), # ignore
            (  0,  0,  0), # other
            (255,255,255)  # road
        )

rgb2class_map = {
    (127,127,127): 0,
    (  0,  0,  0): 1,
    (255,  0,  0): 2,
    (255,105,180): 3,
    (  0,  0,255): 4,
    (255,255,  0): 5
}

def rgb2class(fn):
    gt = cv.imread(fn, cv.IMREAD_UNCHANGED)
    new_gt = np.empty(gt.shape[:2], dtype=np.uint8)
    for k, v in rgb2class_map.items():
        i = np.logical_and(gt[:,:,2]==k[0], gt[:,:,1]==k[1])
        i = np.logical_and(i, gt[:,:,0]==k[2])
        i = np.where(i)
        new_gt[i[0],i[1]] = v
    return new_gt

class DatasetLoader_dlr_roadsegmentation():
    def __init__(self, config):
        self.images = []
        self.image_subsets = []
        for subset in ("train", "test"):
            for fn in glob.iglob(f"{config.dataset_path}/Aerial/{subset}_images/*.jpg"):
                img_set = []
                img_set.append(cv.imread(fn, cv.IMREAD_UNCHANGED))
                fn = fn[:-4]
                mask = cv.imread(fn + "_mask.tif", cv.IMREAD_UNCHANGED)
                fn = fn.split("/")
                fn[-2] = "colorAnnotation"
                fn = "/".join(fn)
                gt = rgb2class(fn + ".png")
                gt[mask!=255] = 0
                img_set.append(gt)
                self.images.append(img_set)
            self.image_subsets.extend(
                [0 if subset=="train" else 2] * (len(self.images) - len(self.image_subsets))
            )
                
        self.channels = {
            "red": (0, 2),
            "green": (0, 1),
            "blue": (0, 0),
            "gt": (1, 0)
        }            
        self.num_classes = 6
        self.gsd = 9
        self.lut = (
            (127,127,127), # ignore
            (  0,  0,  0), # background
            (255,  0,  0), # building
            (255,105,180), # road
            (  0,  0,255), # sidewalk
            (255,255,  0)  # parking
        )
