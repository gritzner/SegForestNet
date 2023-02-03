import core
import types
import glob
import json
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import utils


def parse_experiment(root_dir, config_names, ignore_class=None, history_prefix=""):
    if type(config_names) != tuple and type(config_names) != list:
        config_names = [config_names]
    data = []
        
    for fn in glob.iglob(f"{root_dir}/**/done", recursive=True):
        data.append(types.SimpleNamespace(filename=fn))
        
        for config_name in config_names:
            fn = fn.split("/")
            fn[-1] = f"{config_name}.full.yaml"
            fn = "/".join(fn)
            if os.path.exists(fn):
                break
            
        with open(fn, "r") as f:
            config = yaml.full_load(f.read())
        data[-1].config = core.parse_dict(config)
        data[-1].config.name = config_name
            
        fn = fn.split("/")
        fn[-1] = f"{history_prefix}/history.json"
        fn = "/".join(fn)
            
        with open(fn, "r") as f:
            history = json.load(f)
        for k, v in history.items():
            if "conf_mat" in k:
                if ignore_class is None:
                    setattr(data[-1], k, [utils.ConfusionMatrix.from_dict(conf_mat, -100).C for conf_mat in v])
                else:
                    setattr(data[-1], k, [types.SimpleNamespace(conf_mat=conf_mat.C, **conf_mat.compute_metrics().__dict__) for conf_mat in
                                          [utils.ConfusionMatrix.from_dict(conf_mat, ignore_class) for conf_mat in v]])
            else:
                setattr(data[-1], k, np.asarray(v))
        
    return data

def cdf_plot(data, num_samples=100):
    legend = []
    
    for label, values in data.items():
        if type(values) != np.ndarray:
            values = np.asarray(values)
        assert len(values.shape) == 1
        
        xs = np.linspace(values.min(), values.max(), num_samples)
        ys = np.asarray(
            [np.count_nonzero(values<=x)/values.shape[0] for x in xs],
            dtype=np.float32
        )
        
        plt.plot(xs, ys)
        legend.append(label)
    
    plt.title("cdf plot")
    plt.ylabel("cumulative probability")
    plt.legend(legend)
    