import timm
import gzip
import torch

for modelname, filename in (("mobilenetv2_100", "mobilenetv2.pt.gz"), ("legacy_xception", "xception.pt.gz")):
    weights = timm.create_model(modelname, pretrained=True).state_dict()
    with gzip.open(filename, "wb") as f:
        torch.save(weights, f)
