import gzip
import bz2
import builtins
import types
import argparse
from pathlib import Path
import re
import yaml
import os
import sys
import subprocess
call = lambda cmd: subprocess.call(cmd, shell=True)


def str2bool(s):
    return not s.lower() in ("", "0", "f", "false", "n", "no", "off")


def open(filename, mode):
    if filename[-3:] == ".gz":
        return gzip.open(filename, mode)
    elif filename[-4:] == ".bz2":
        return bz2.open(filename, mode)
    else:
        return builtins.open(filename, mode)


def list_to_tuple(v):
    if type(v) == list:
        v = tuple([list_to_tuple(x) for x in v])
    return v


def parse_map(v):
    d = {}
    for x in v:
        if len(x) != 2:
            raise RuntimeError(f"cannot parse '{x}' into map entry: length is not 2.")
        d[list_to_tuple(x[0])] = list_to_tuple(x[1])
    return d


def parse_dict(d):
    result = types.SimpleNamespace()
    for k, v in d.items():
        if type(v) == dict:
            v = parse_dict(v)
        elif type(v) == list:
            if k[-4:].lower() == ":map":
                k = k[:-4]
                v = parse_map(v)
            else:
                v = [parse_dict(x) if type(x)==dict else x for x in v]
        setattr(result, k, v)
    return result


def get_object_meta_info(name):
    if type(name) == list:
        if len(name) > 1:
            return name[0], globals()[f"{name[1]}_params"]
        if len(name) > 0:
            name = name[0]
        else:
            raise RuntimeError("requested meta-information of empty list")
    return name, globals()[f"{name}_params"]


def create_object(module, name, **params):
    name, object_config = get_object_meta_info(name)
    params = types.SimpleNamespace(**params)
    return getattr(module, name)(object_config, params)


def init(args=None):
    parser = argparse.ArgumentParser(description="Deep Learning for Remote Sensing")
    parser.add_argument("configuration", nargs="?", default="", type=str, help="configuration file")
    parser.add_argument("parameters", nargs="*", type=str, help="configuration parameters")
    parser.add_argument("--cpu", action="store_true", help="use CPU instead of GPU")
    parser.add_argument("--compile", action="store_true", help="force Rust code compilation and then terminate")
    parser.add_argument("--git-log", default="", type=str, help="echo text file instead of actually running git log")
    args = parser.parse_args([arg for arg in args.split(" ") if len(arg) > 0] if type(args)==str else args)
    
    base_path = Path(__file__).absolute().parent.parent
    cache_path = f"{base_path}/tmp/cache"
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    home = os.environ["HOME"]
    with open(f"{home}/.aethon/user.yaml", "r") as f:
        user = parse_dict(yaml.full_load("".join(f.readlines())))
    assert user.git_log_num_lines >= 0
        
    if args.compile:
        import rust
        rust.init(True, cache_path)
        sys.exit(0)
        
    if len(args.configuration) == 0:
        raise RuntimeError(f"configuration file argument is undefined, try running with --help for additional information")
    config_file = base_path.joinpath("cfgs", args.configuration + ".yaml")

    print("%s/python %s"%(base_path, " ".join(sys.argv)))
    
    if args.configuration[0] != "@" and not config_file.is_file():
        raise RuntimeError(f"unknown configuration file '{config_file}'")

    with open(f"{base_path}/core/defaults.yaml", "r") as f:
        defaults_yaml = "".join(f.readlines())
    raw_yaml = ""
    if args.configuration[0] != "@":
        print(f"parsing configuration file '{config_file}'")
        with config_file.open() as f:
            raw_yaml = f.read()
    config_yaml = f"{raw_yaml}"
    pattern = re.compile("\$(\d)")
    while True:
        match = pattern.search(config_yaml)
        if not match:
            break
        span = match.span()
        index = int(match.group(1))
        if index >= len(args.parameters):
            raise RuntimeError(f"insufficient number of parameters for configuration file; needs at least {index+1} parameter(s)")
        config_yaml = config_yaml[:span[0]] + str(args.parameters[index]) + config_yaml[span[1]:]
    full_config_yaml = f"{defaults_yaml}\n\n{config_yaml}"
    config = parse_dict(yaml.full_load(full_config_yaml))

    config.base_path = base_path
    config.cache_path = cache_path
    if args.configuration[0] != "@":
        if not (hasattr(config, "output_path") and config.output_path):
            config.output_path = f"{base_path}/tmp"
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)
        if getattr(config, "clear_output_path", False):
            call(f"rm -rf {config.output_path}/*")
        call(f"hostname > {config.output_path}/job_details.txt")
        with open(f"{config.output_path}/job_details.txt", "a") as f:
            f.write(" ".join(sys.argv) + "\n\n")
        call(f"nvidia-smi >> {config.output_path}/job_details.txt")
        call(f"echo >> {config.output_path}/job_details.txt")
        if len(args.git_log) == 0:
            call(f"git log -{user.git_log_num_lines} >> {config.output_path}/job_details.txt")
        else:
            call(f"cat {args.git_log} >> {config.output_path}/job_details.txt")
        if "SLURM_JOB_ID" in os.environ:
            slurm_job_id = os.environ["SLURM_JOB_ID"]
            call(f"scontrol -d show job={slurm_job_id} > {config.output_path}/slurm_job_details.txt")
        with open(f"{config.output_path}/{args.configuration}.yaml", "w") as f:
            f.write(config_yaml)
        with open(f"{config.output_path}/{args.configuration}.raw.yaml", "w") as f:
            f.write(raw_yaml)
        with open(f"{config.output_path}/{args.configuration}.full.yaml", "w") as f:
            f.write(full_config_yaml)

    config.num_threads = len(os.sched_getaffinity(0))
    
    import torch
    config.device = torch.device("cpu" if args.cpu else "cuda:0")
    
    from concurrent.futures import ThreadPoolExecutor
    config.thread_pool = ThreadPoolExecutor()
    
    import rust
    rust.init(False, cache_path)
    
    import numpy as np
    config.random_seeds = np.load(f"{base_path}/core/random_seeds.npy")
    
    for k, v in config.__dict__.items():
        if k[:2] == "__":
            continue
        globals()[k] = v
    globals()["args"] = args
    globals()["user"] = user
