from ConfigSpace import Configuration, ConfigurationSpace, Integer, Float
from smac import HyperparameterOptimizationFacade, Scenario
from pathlib import Path
import glob
import bz2
import json
import numpy as np
import aethonproxy

aethonproxy.python_bin = "/home/gritzner/tmp/miniconda3/envs/torch2/bin/python"
aethonproxy.slurm_config = "luis", "HPO", "24:00:00", "9G", "hpo.tar.bz2", "hpo/hpo", "hpo"

def train(config: Configuration, seed: int = 0) -> float:
    path = aethonproxy.call((
        config["learning_rate"],
        config["momentum"],
        config["weight_decay"]
    ))
    history = next(glob.iglob(f"{path}/**/history.json.bz2", recursive=True))
    with bz2.open(history, "r") as f:
        data = json.load(f)
    return 1 - np.max(data["val_miou"])

if __name__ == "__main__":
    proxy = aethonproxy.init()
    configspace = ConfigurationSpace()
    configspace.add_hyperparameters([
        Float("learning_rate", bounds=(0.001, 0.1), log=True),
        Float("momentum", bounds=(0., .99)),
        Float("weight_decay", bounds=(0.00001, .00003))
    ])
    scenario = Scenario(
        configspace,
        name = "hpo",
        output_directory = Path("tmp/smac"),
        deterministic = True,
        n_trials = 20,
        n_workers = 5
    )
    smac = HyperparameterOptimizationFacade(scenario, train)
    print(smac.optimize())
    proxy.close()
