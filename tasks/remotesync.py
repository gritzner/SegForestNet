import core
import yaml
import sys


def remotesync():
    assert len(core.args.parameters) > 1
    config, dataset = core.args.parameters[:2]
    with open(f"{core.base_path}/cfgs/{config}.yaml", "r") as f:
        config = core.parse_dict(yaml.safe_load(f.read()))
    target_path = f"{config.primary_path}/{dataset}"
    if len(core.args.parameters) == 2 or not core.str2bool(core.args.parameters[2]):
        dataset_paths = core.user.dataset_paths[dataset]
        returncode = core.call(f"ssh {config.servers.login} mkdir -p {target_path}")
        if returncode != 0:
            sys.exit(returncode)
        if type(dataset_paths) == str:
            dataset_paths = (dataset_paths,)
        for i, dataset_path in enumerate(dataset_paths):
            returncode = core.call(f"rsync -avz {dataset_path} {config.servers.transfer}:{target_path}/{i}")
            if returncode != 0:
                sys.exit(returncode)         
    else:
        sys.exit(core.call(f"ssh {config.servers.transfer} rsync -avz {target_path} {config.secondary_path}"))
