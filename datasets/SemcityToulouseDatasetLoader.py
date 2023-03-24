import PIL.Image
import numpy as np
import tifffile


lut = ( # for ground truth visualization
    (255,255,255), # void
    ( 38, 38, 38), # impervious surface
    (238,118, 33), # building
    ( 34,139, 34), # pervious surface
    (  0,222,137), # high vegetation
    (255,  0,  0), # car
    (  0,  0,238), # water
    (160, 30,230)  # sports venues
)

annotators = {"03": "3", "04": "3", "07": "2", "08": "2"}


class DatasetLoader_toulouse():
    def __init__(self, config):
        root_path = "/data/geoTL/daten/SemCityToulouse/tmp"
        
        self.images = []        
        for file_id in ("03", "04", "07", "08"):
            img_set = []
            img_set.append(PIL.Image.open(f"{root_path}/img_multispec_05/TLS_BDSD_RGB_noGeo/TLS_BDSD_RGB_noGeo_{file_id}.tif"))
            img_set.append(PIL.Image.open(f"{root_path}/img_multispec_05/TLS_BDSD_NIRRG_noGeo/TLS_BDSD_NIRRG_noGeo_{file_id}.tif"))
            img_set.append(PIL.Image.open(f"{root_path}/semantic_05/TLS_indMap_noGeo/TLS_indMap_noGeo_{file_id}_{annotators[file_id]}.tif"))
            img_set.append(PIL.Image.open(f"{root_path}/instances_building_05/TLS_instances_building_indMap/TLS_instances_building_indMap_{file_id}.tif"))
            self.images.append(img_set)
        
        self.channels = {
            "ir": (1, 0),
            "red": (0, 0),
            "green": (0, 1),
            "blue": (0, 2),
            "gt": (2, 0),
            "instances": (3, 0)
        }
        self.num_classes = 8
        self.instances = {
            2: "building"
        }
        self.gsd = 50
        self.lut = lut

class DatasetLoader_toulouse_multi():
    def __init__(self, config):
        root_path = "/data/geoTL/daten/SemCityToulouse/tmp"
        
        self.images = []
        for file_id in ("03", "04", "07", "08"):
            img_set = []
            img = tifffile.imread(f"{root_path}/img_multispec_05/TLS_BDSD_M/TLS_BDSD_M_{file_id}.tif")
            for i in range(img.shape[2]):
                img_set.append(PIL.Image.fromarray(img[:,:,i]))
            img_set.append(PIL.Image.open(f"{root_path}/img_multispec_05/TLS_BDSD_RGB_noGeo/TLS_BDSD_RGB_noGeo_{file_id}.tif"))
            img_set.append(PIL.Image.open(f"{root_path}/semantic_05/TLS_indMap_noGeo/TLS_indMap_noGeo_{file_id}_{annotators[file_id]}.tif"))
            img_set.append(PIL.Image.open(f"{root_path}/instances_building_05/TLS_instances_building_indMap/TLS_instances_building_indMap_{file_id}.tif"))
            self.images.append(img_set)
        
        self.channels = {
            "blue2": (0, 0),
            "blue": (1, 0),
            "green": (2, 0),
            "yellow": (3, 0),
            "red": (4, 0),
            "red2": (5, 0),
            "ir": (6, 0),
            "ir2": (7, 0),
            "vis_red": (8, 0),
            "vis_green": (8, 1),
            "vis_blue": (8, 2),
            "gt": (9, 0),
            "instances": (10, 0)
        }
        self.num_classes = 8
        self.instances = {
            2: "building"
        }
        self.gsd = 50
        self.lut = lut
