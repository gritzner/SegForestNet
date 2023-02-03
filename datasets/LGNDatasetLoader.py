import core
import PIL.Image
import glob
from osgeo import gdal, ogr
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
    def __init__(self, path, left, bottom, is_hannover, config):
        PIL.Image.MAX_IMAGE_PIXELS = None
        
        images = []
        images.append(PIL.Image.open(
            next(glob.iglob(f"{path}/*_col.tif"))
        ))
        images.append(PIL.Image.open(
            next(glob.iglob(f"{path}/*_ir.tif"))
        ))
        images.append(PIL.Image.open(
            next(glob.iglob(f"{path}/dom/dom_cleaned.tif"))
        ))
        
        img_size = (
            images[1].size[1],
            images[1].size[0]
        )
        images.append(self.rasterize_ground_truth(path, left, bottom, img_size, is_hannover))
        self.split_images(images, is_hannover, config)
                
        self.channels = {
            "ir": (1, 0),
            "red": (0, 0),
            "green": (0, 1),
            "blue": (0, 2),
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
        self.debug_counts = {"no ID":0, "no class":0, "empty geometry":0}
        
        for filename in glob.iglob(f"{path}/shape/*.shp"):
            shapes = gdal.OpenEx(filename, gdal.OF_VECTOR)
            if shapes is None:
                raise RuntimeError(f"cannot open '{filename}'")
            if shapes.GetLayerCount() != 1:
                raise RuntimeError(f"unsupported number of layers ({shapes.GetLayerCount()}) in '{filename}'")
            layer = shapes.GetLayerByIndex(0)
            layer.ResetReading()
            for feature in layer:
                object_id = None
                object_class = None
                for i in range(feature.GetFieldCount()):
                    field_name = feature.GetFieldDefnRef(i).GetName()
                    field_value = feature.GetField(i)
                    if field_name == "id":
                        object_id = int(field_value)
                    elif field_name == "klasse":
                        object_class = int(field_value)
                    else:
                        raise RuntimeError(f"unsupported field in '{filename}': {field_name} = {field_value}")
                if not object_id:
                    self.debug_counts["no ID"] += 1
                    continue
                if not object_class:
                    self.debug_counts["no class"] += 1
                    continue
                if is_hannover and object_class == 5:
                    continue # these shapes appear to be malformed; TODO: verify
                geometry = feature.GetGeometryRef()
                if not geometry:
                    self.debug_counts["empty geometry"] += 1
                    continue
                if not object_class in data:
                    data[object_class] = {}
                current_class = data[object_class]
                if object_id in current_class:
                    raise RuntimeError(f"duplicate ID: {object_id} in class {object_class}")
                else:
                    current_class[object_id] = []
                for polygon in geometry:
                    if polygon.GetGeometryType() == ogr.wkbPolygon:
                        if polygon.GetGeometryCount() != 1:
                            raise RuntimeError(f"unsupported polygon of subgeometry count {polygon.GetGeometryCount()} in '{filename}'")
                        polygon = polygon.GetGeometryRef(0)
                    if polygon.GetGeometryType() != ogr.wkbLineString and polygon.GetGeometryType() != ogr.wkbLinearRing:
                        raise RuntimeError(f"unsupported geometry type: {polygon.GetGeometryType()} in '{filename}'")
                    polygon_data = np.empty((polygon.GetPointCount(), 2), dtype=np.float64)
                    current_class[object_id].append(polygon_data)
                    for i in range(polygon_data.shape[0]):
                        p = polygon.GetPoint(i)
                        polygon_data[i,0] = (p[0] - left) / 2000 # x coordinate
                        polygon_data[i,1] = 1.0 - ((p[1] - bottom) / 2000) # y coordinate
        
        img = np.zeros(img_size, dtype=np.uint8) + 255
        for key in reversed(sorted(data.keys())):
            geometries = data[key].values()
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
                    polygons[i,j,:polygon.shape[0]] = polygon
            rust.rasterize_objects(img, geometry_lengths, polygon_lengths, polygons, key-1, core.num_threads)
        rust.flood_fill_holes(img)
        
        return PIL.Image.fromarray(img)
        
    def split_images(self, images, is_hannover, config):
        self.images = []
        for i, j in itertools.product(range(4), range(4)):
            if i == 0 and j == 3 and is_hannover and getattr(config, "skip_Eilenriede", True):
                continue
            roi = (j*2500, i*2500, (j+1)*2500, (i+1)*2500)
            self.images.append([img.crop(roi) for img in images])

class DatasetLoader_hannover(LGNDatasetLoader):
    def __init__(self, config):
        super().__init__("/data/geoTL/daten/LGN/Hannover", 550000, 5802000, True, config)
        
class DatasetLoader_buxtehude(LGNDatasetLoader):
    def __init__(self, config):
        super().__init__("/data/geoTL/daten/LGN/Buxtehude", 546000, 5924000, False, config)
        
class DatasetLoader_nienburg(LGNDatasetLoader):
    def __init__(self, config):
        super().__init__("/data/geoTL/daten/LGN/Nienburg", 514000, 5830000, False, config)
