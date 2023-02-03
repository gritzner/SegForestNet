import types
import argparse
from pathlib import Path
import re
import yaml
import os
import sys
import subprocess
# IMPORTANT: Do not add any imports here otherwise you may break the code below that limits the number of threads! Some libraries, e.g., NumPy, must not be imported before certain environment variables are set!


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


def parse_yaml(raw_yaml, defaults_yaml, args):
    config_yaml = f"{raw_yaml}"
    pattern = re.compile("\$(\d)")
    while True:
        match = pattern.search(config_yaml)
        if not match:
            break
        span = match.span()
        index = int(match.group(1))
        if index >= len(args):
            raise RuntimeError(f"insufficient number of parameters for configuration file; needs at least {index+1} parameter(s)")
        config_yaml = config_yaml[:span[0]] + str(args[index]) + config_yaml[span[1]:]
    full_config_yaml = f"{defaults_yaml}\n\n{config_yaml}"
    return parse_dict(yaml.full_load(full_config_yaml)), config_yaml, full_config_yaml


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
    parser.add_argument("configuration", type=str, help="configuration file")
    parser.add_argument("parameters", nargs="*", type=str, help="configuration parameters")
    parser.add_argument("--cpu", action="store_true", help="use CPU instead of GPU")
    parser.add_argument("--gpu", default=-1, type=int, help="GPU to use")
    parser.add_argument("--threads", default=-1, type=int, help="maximum number of threads")
    parser.add_argument("--compile", action="store_true", help="compile Rust code")
    parser.add_argument("--compile-only", action="store_true", help="compile Rust code and terminate")
    parser.add_argument("--check-embedded-code", action="store_true", help="compute result of embedded code and terminate")
    parser.add_argument("--array-id", default=-1, type=int, help="override SLURM_ARRAY_TASK_ID for embedded code if set to 0 or greater")
    parser.add_argument("--notify", action="store_true", help="send push notification upon completion")
    parser.add_argument("--slurm", action="store_true", help="submit SLURM job and terminate; embedded code will not be executed before submission!")
    args = parser.parse_args([arg for arg in args.split(" ") if len(arg) > 0] if type(args)==str else args)
    
    base_path = Path(__file__).absolute().parent.parent
    config_file = base_path.joinpath("cfgs", args.configuration + ".yaml")
    cache_path = f"{base_path}/tmp/cache"

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    if args.compile_only:
        import rust
        rust.init(True, cache_path)
        sys.exit(0)

    print("%s/python %s"%(base_path, " ".join(sys.argv)))
    
    if not config_file.is_file():
        raise RuntimeError(f"unknown configuration file '{config_file}'")

    home = os.environ["HOME"]
    with open(f"{home}/.aethon/user.yaml", "r") as f:
        user = parse_dict(yaml.full_load("".join(f.readlines())))
    with open(f"{base_path}/core/defaults.yaml", "r") as f:
        defaults_yaml = "".join(f.readlines())
    print(f"parsing configuration file '{config_file}'")
    with config_file.open() as f:
        raw_yaml = "".join(f.readlines())
    config, config_yaml, full_config_yaml = parse_yaml(raw_yaml, defaults_yaml, [
        args.parameters[i] if i<len(args.parameters) else "" for i in range(10)
    ])
    
    if args.slurm:
        command = " ".join([arg for arg in sys.argv if arg != "--slurm"])
        slurm = user.slurm
        if hasattr(config, "slurm"):
            slurm.update(config.slurm)
        slurm = [f"#SBATCH --{str(k)}={str(v)}" for k,v in slurm.items()]
        slurm = "\n".join(slurm)
        slurm = subprocess.run("sbatch", input=f"""#!/bin/bash
{slurm}

source {user.conda.path}/bin/activate {user.conda.environment}
export PROJ_LIB={user.conda.path}/envs/{user.conda.environment}/share/proj
cd {base_path}
python {command}""", text=True, capture_output=True)
        print("sbatch.stdout:", slurm.stdout.strip())
        print("sbatch.stderr:", slurm.stderr.strip())
        sys.exit(slurm.returncode)
    
    embedded_parameters = args.parameters
    if hasattr(config, "arguments"):
        command = [sys.executable, "-"]
        command.extend(args.parameters)
        env = os.environ.copy()
        if args.array_id >= 0:
            env["SLURM_ARRAY_TASK_ID"] = str(args.array_id)
        embedded_parameters = subprocess.run(command, input=f"""import sys
import os
import numpy as np

array_id = int(os.getenv(\"SLURM_ARRAY_TASK_ID\", default=-1))

def return_tuple(*args):
    args = ["\\"%s\\""%arg if type(arg)==str else str(arg) for arg in args]
    args = ",".join(args)
    print("(%s,)"%args, end="")

{config.arguments}""", env=env, text=True, capture_output=True)
        if embedded_parameters.returncode != 0:
            raise RuntimeError(embedded_parameters.stderr)
        embedded_parameters = eval(embedded_parameters.stdout)
        print(f"embedded code returned '{embedded_parameters}'")
        embedded_parameters = tuple([
            embedded_parameters[i] if i<len(embedded_parameters) else args.parameters[i] for i in range(max(len(embedded_parameters),len(args.parameters)))
        ])
        globals()["embedded_parameters"] = embedded_parameters
    config, config_yaml, full_config_yaml = parse_yaml(raw_yaml, defaults_yaml, embedded_parameters)
    
    if args.check_embedded_code:
        sys.exit(0)

    config.cache_path = cache_path
    if not (hasattr(config, "output_path") and config.output_path):
        config.output_path = f"{base_path}/tmp"
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    if getattr(config, "clear_output_path", False):
        os.system(f"rm -rf {config.output_path}/*")
    os.system(f"hostname > {config.output_path}/job_details.txt")
    with open(f"{config.output_path}/job_details.txt", "a") as f:
        f.write(" ".join(sys.argv) + "\n\n")
    os.system(f"nvidia-smi >> {config.output_path}/job_details.txt")
    os.system(f"echo >> {config.output_path}/job_details.txt")
    os.system(f"git log -12 >> {config.output_path}/job_details.txt")
    if "SLURM_JOB_ID" in os.environ:
        slurm_job_id = os.environ["SLURM_JOB_ID"]
        os.system(f"scontrol -d show job={slurm_job_id} > {config.output_path}/slurm_job_details.txt")
    with open(f"{config.output_path}/{args.configuration}.yaml", "w") as f:
        f.write(config_yaml)
    with open(f"{config.output_path}/{args.configuration}.raw.yaml", "w") as f:
        f.write(raw_yaml)        
    with open(f"{config.output_path}/{args.configuration}.full.yaml", "w") as f:
        f.write(full_config_yaml)

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config.gpu = args.gpu
    config.num_threads = args.threads if args.threads > 0 else len(os.sched_getaffinity(0))
    
    if args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
        os.environ["NUMBA_NUM_THREADS"] = str(args.threads)
        import torch
        torch.set_num_threads(args.threads)
    else:
        import torch
    config.device = torch.device("cpu" if args.cpu else "cuda:0")
    
    from concurrent.futures import ThreadPoolExecutor
    config.thread_pool = ThreadPoolExecutor(max_workers=config.num_threads)
    
    import rust
    rust.init(args.compile, cache_path)
    
    import numpy as np
    random_seeds = np.load(f"{base_path}/core/random_seeds.npy")
    if not (hasattr(config, "random_seeds") and config.random_seeds):
        config.random_seeds = (0, random_seeds.shape[0])
    config.random_seeds = random_seeds[config.random_seeds[0]:config.random_seeds[1]]

    for k, v in config.__dict__.items():
        if k[:2] == "__":
            continue
        globals()[k] = v
    globals()["args"] = args
    globals()["user"] = user
