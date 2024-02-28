import core
import cv2 as cv
import glob
import json
import bz2
import numpy as np
import rust
import itertools


# classes:
#  1: trees
#  2: buildings
#  3: vehicles
#  4: objects
#  5: tracks
#  6: streets
#  7: traffic routes
#  8: water
#  9: sealed surfaces
# 10: fields with vegetation
# 11: fields without vegetation
# 12: fields
# 13: low vegetation
# 14: unsealed surfaces
# 15: surfaces


class LGNDatasetLoader():
    def __init__(self, left, bottom, is_hannover, config):
        images = []
        images.append(cv.imread(
            next(glob.iglob(f"{config.dataset_path}/*_col.tif")),
            cv.IMREAD_UNCHANGED
        ))
        images.append(cv.imread(
            next(glob.iglob(f"{config.dataset_path}/*_ir.tif")),
            cv.IMREAD_UNCHANGED
        ))
        images.append(cv.imread(
            next(glob.iglob(f"{config.dataset_path}/dom/dom_cleaned.tif")),
            cv.IMREAD_UNCHANGED
        ))
        
        images.append(self.rasterize_ground_truth(config.dataset_path, left, bottom, images[1].shape, is_hannover))
        self.split_images(images, is_hannover, config)
                
        self.channels = {
            "ir": (1, 0),
            "red": (0, 2),
            "green": (0, 1),
            "blue": (0, 0),
            "depth": (2, 0),
            "gt": (3, 0)
        }
        self.num_classes = 15
        self.gsd = 20
        self.lut = ( # for ground truth visualization, IDs get decreased by one during rasterization
            (255,   0,   0), #  1: trees
            (  0,   0, 255), #  2: buildings
            (255, 255,   0), #  3: vehicles
            (127, 127,   0), #  4: objects
            (255,   0, 255), #  5: tracks
            (170,   0, 170), #  6: streets
            ( 85,   0,  85), #  7: traffic routes
            (  0, 255, 255), #  8: water
            (127, 127, 127), #  9: sealed surfaces
            (  0, 255,   0), # 10: fields with vegetation
            (  0, 191,   0), # 11: fields without vegetation
            (  0, 127,   0), # 12: fields
            (  0,  63,   0), # 13: low vegetation
            (  0,   0,   0), # 14: unsealed surfaces
            (255, 255, 255), # 15: surfaces
        )
        
    def rasterize_ground_truth(self, path, left, bottom, img_size, is_hannover):
        data = {}        
        with bz2.open(f"{path}/ground_truth.geojson.bz2", "r") as f:
            shapes = json.load(f)
            
        object_ids = set()
        for feature in shapes["features"]:
            object_id = feature["properties"]["id"]
            assert not object_id in object_ids
            object_ids.add(object_id)
                
            geom = feature["geometry"]
            assert geom["type"] in ("Polygon", "MultiPolygon")
            if len(geom["coordinates"]) == 0:
                continue
            if geom["type"] == "Polygon":
                geom = [geom["coordinates"]]
            else:
                geom = geom["coordinates"]
            geom = [[np.asarray(coords) for coords in poly] for poly in geom]
            
            object_class = feature["properties"]["klasse"]
            if not object_class in data:
                data[object_class] = []
            data[object_class].extend(geom)
        
        img = np.zeros(img_size, dtype=np.uint8) + 255
        for key in reversed(sorted(data.keys())):
            geometries = data[key]
            if len(geometries) == 0:
                continue
            geometry_lengths = np.empty(len(geometries), dtype=np.int32)
            max_length = np.max([len(geometry) for geometry in geometries])
            polygon_lengths = np.empty((len(geometries), max_length), dtype=np.int32)
            max_length = [np.max([polygon.shape[0] for polygon in geometry]) for geometry in geometries]
            max_length = np.max(max_length)
            polygons = np.empty((len(geometries), polygon_lengths.shape[1], max_length, 2), dtype=np.float64)
            for i, geometry in enumerate(geometries):
                geometry_lengths[i] = len(geometry)
                for j, polygon in enumerate(geometry):
                    polygon_lengths[i,j] = polygon.shape[0]
                    polygon[:,0] = (polygon[:,0] - left) / 2000 # x coordinate
                    polygon[:,1] = 1.0 - ((polygon[:,1] - bottom) / 2000) # y coordinate
                    polygons[i,j,:polygon.shape[0]] = polygon
            rust.rasterize_objects(img, geometry_lengths, polygon_lengths, polygons, key-1, core.num_threads)
        rust.flood_fill_holes(img)
        
        return img
        
    def split_images(self, images, is_hannover, config):
        self.images = []
        for i, j in itertools.product(range(4), range(4)):
            if i == 0 and j == 3 and is_hannover and getattr(config, "skip_Eilenriede", True):
                continue
            self.images.append([img[i*2500:(i+1)*2500,j*2500:(j+1)*2500] for img in images])

class DatasetLoader_hannover(LGNDatasetLoader):
    def __init__(self, config):
        super().__init__(550000, 5802000, True, config)
        
class DatasetLoader_buxtehude(LGNDatasetLoader):
    def __init__(self, config):
        super().__init__(546000, 5924000, False, config)
        
class DatasetLoader_nienburg(LGNDatasetLoader):
    def __init__(self, config):
        super().__init__(514000, 5830000, False, config)
