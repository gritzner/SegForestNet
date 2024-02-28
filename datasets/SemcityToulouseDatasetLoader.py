import cv2 as cv
import glob


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

panoptic_training_set = "04", "08"
test_set = "03", "07"


class DatasetLoader_toulouse():
    def __init__(self, config):
        self.images = []
        self.image_subsets = []
        
        annotators = {}
        for fn in glob.iglob(f"{config.dataset_path}/semantic_05/TLS_indMap_noGeo/*.tif"):
            file_id, annotator = fn.split(".")[0].split("_")[-2:]
            if not file_id in annotators or annotator != "1":
                annotators[file_id] = annotator
        
        for file_id in sorted([file_id for file_id, annotator in annotators.items() if annotator != "1"]):
            img_set = []
            rgb = cv.imread(
                f"{config.dataset_path}/img_multispec_05/TLS_BDSD_RGB_noGeo/TLS_BDSD_RGB_noGeo_{file_id}.tif",
                cv.IMREAD_UNCHANGED
            )
            img = cv.imread(
                f"{config.dataset_path}/img_multispec_05/TLS_BDSD_M/bands_to_rows/TLS_BDSD_M_{file_id}.tif",
                cv.IMREAD_UNCHANGED
            )
            assert rgb.shape[0]*8 == img.shape[0] and img.shape[1] == rgb.shape[1]
            for i in range(8):
                offset = rgb.shape[0] * i
                img_set.append(img[offset:offset+rgb.shape[0]])
                assert img_set[-1].shape == rgb.shape[:2]
            img_set.append(rgb)
            img_set.append(cv.imread(
                f"{config.dataset_path}/semantic_05/TLS_indMap_noGeo/TLS_indMap_noGeo_{file_id}_{annotators[file_id]}.tif",
                cv.IMREAD_UNCHANGED
            ))

            self.images.append(img_set)
            if file_id in panoptic_training_set:
                self.image_subsets.append(1)
            elif file_id in test_set:
                self.image_subsets.append(2)
            else:
                self.image_subsets.append(0)
                        
        self.channels = {
            "blue2": (0, 0),
            "blue": (1, 0),
            "green": (2, 0),
            "yellow": (3, 0),
            "red": (4, 0),
            "red2": (5, 0),
            "ir": (6, 0),
            "ir2": (7, 0),
            "vis_red": (8, 2),
            "vis_green": (8, 1),
            "vis_blue": (8, 0),
            "gt": (9, 0)
        }
        self.num_classes = 8
        self.gsd = 50
        self.lut = lut

class DatasetLoader_toulouse_full():
    def __init__(self, config):
        self.images = []
        self.image_subsets = []
        
        annotators = {}
        for fn in glob.iglob(f"{config.dataset_path}/semantic_05/TLS_indMap_noGeo/*.tif"):
            file_id, annotator = fn.split(".")[0].split("_")[-2:]
            if not file_id in annotators or annotator != "1":
                annotators[file_id] = annotator
        
        for file_id in sorted(annotators.keys()):
            img_set = []
            rgb = cv.imread(
                f"{config.dataset_path}/img_multispec_05/TLS_BDSD_RGB_noGeo/TLS_BDSD_RGB_noGeo_{file_id}.tif",
                cv.IMREAD_UNCHANGED
            )
            img = cv.imread(
                f"{config.dataset_path}/img_multispec_05/TLS_BDSD_M/bands_to_rows/TLS_BDSD_M_{file_id}.tif",
                cv.IMREAD_UNCHANGED
            )
            assert rgb.shape[0]*8 == img.shape[0] and img.shape[1] == rgb.shape[1]
            for i in range(8):
                offset = rgb.shape[0] * i
                img_set.append(img[offset:offset+rgb.shape[0]])
                assert img_set[-1].shape == rgb.shape[:2]
            img_set.append(rgb)
            img_set.append(cv.imread(
                f"{config.dataset_path}/semantic_05/TLS_indMap_noGeo/TLS_indMap_noGeo_{file_id}_{annotators[file_id]}.tif",
                cv.IMREAD_UNCHANGED
            ))

            self.images.append(img_set)
            if file_id in panoptic_training_set:
                self.image_subsets.append(1)
            elif file_id in test_set:
                self.image_subsets.append(2)
            else:
                self.image_subsets.append(0)
                        
        self.channels = {
            "blue2": (0, 0),
            "blue": (1, 0),
            "green": (2, 0),
            "yellow": (3, 0),
            "red": (4, 0),
            "red2": (5, 0),
            "ir": (6, 0),
            "ir2": (7, 0),
            "vis_red": (8, 2),
            "vis_green": (8, 1),
            "vis_blue": (8, 0),
            "gt": (9, 0)
        }
        self.num_classes = 8
        self.gsd = 50
        self.lut = lut
