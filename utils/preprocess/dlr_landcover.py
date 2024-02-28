import os
import yaml
import tifffile
import numpy as np

home = os.environ["HOME"]
with open(f"{home}/.aethon/user.yaml", "r") as f:
    user_config = yaml.full_load("\n".join(f.readlines()))
    
for dataset_path in user_config["dataset_paths:map"]:
    if dataset_path[0] == "dlr_landcover":
        dataset_path = dataset_path[1]
        break
        
for city, city_gt, swap in (("Berlin","berlin",False), ("Munich","munich",True)):
    sar = tifffile.imread(f"{dataset_path}/images/{city}_s1.tif")
    img = tifffile.imread(f"{dataset_path}/images/{city}_s2.tif")
    assert sar.dtype == img.dtype
    if swap:
        img = np.moveaxis(img, 0, 2)
    assert sar.shape == img.shape[:2]
    new_img = np.empty((img.shape[0]*(img.shape[2]+1), img.shape[1]), dtype=img.dtype)
    for i in range(img.shape[2]):
        offset = img.shape[0] * i
        new_img[offset:offset+img.shape[0]] = img[:,:,i]
    offset = img.shape[0] * img.shape[2]
    new_img[offset:offset+img.shape[0]] = sar
    tifffile.imwrite(f"{dataset_path}/images/{city}_converted.tif", new_img, compression="zlib")
    img = tifffile.imread(f"{dataset_path}/annotations/{city_gt}_anno.tif")
    tifffile.imwrite(f"{dataset_path}/annotations/{city}_converted.tif", img, compression="zlib")
