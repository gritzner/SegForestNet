import os
import yaml
import subprocess
import glob
import tifffile
import numpy as np

home = os.environ["HOME"]
with open(f"{home}/.aethon/user.yaml", "r") as f:
    user_config = yaml.full_load("\n".join(f.readlines()))
    
for dataset_path in user_config["dataset_paths:map"]:
    if dataset_path[0] == "toulouse":
        dataset_path = dataset_path[1]
        break
        
dataset_path = f"{dataset_path}/img_multispec_05/TLS_BDSD_M"
converted_files_path = f"{dataset_path}/bands_to_rows"
subprocess.call(f"mkdir -p {converted_files_path}", shell=True)

for fn in glob.iglob(f"{dataset_path}/TLS_BDSD_M_*.tif"):
    img = tifffile.imread(fn)
    new_img = np.empty((img.shape[0]*img.shape[2], img.shape[1]), dtype=img.dtype)
    actual_fn = fn.split("/")[-1]
    print(f"{actual_fn}:", img.shape, img.dtype, "->", new_img.shape, new_img.dtype)
    for channel in range(img.shape[2]):
        offset = img.shape[0] * channel
        new_img[offset:offset+img.shape[0]] = img[:,:,channel]
    tifffile.imwrite(f"{converted_files_path}/{actual_fn}", new_img, compression="zlib")
    
dataset_path = "/".join(dataset_path.split("/")[:-2])
dataset_path = f"{dataset_path}/instances_building_05/TLS_instances_building_indMap"
converted_files_path = f"{dataset_path}/u16/"
subprocess.call(f"mkdir -p {converted_files_path}", shell=True)

next_id = 1
for fn in glob.iglob(f"{dataset_path}/*.tif"):
    img = tifffile.imread(fn)
    new_img = np.empty(img.shape, dtype=np.uint16)
    for instance_id in np.unique(img):
        if instance_id == 0:
            new_img[img==instance_id] = 0
        else:
            new_img[img==instance_id] = next_id
            next_id += 1
    actual_fn = fn.split("/")[-1]
    print(actual_fn, next_id-1)
    tifffile.imwrite(f"{converted_files_path}/{actual_fn}", new_img, compression="zlib")
