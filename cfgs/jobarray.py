# use 'python aethon.py @archive' to create PG.tar.bz2 and copy it to the proper location (see slurmconfig.yaml) first
# also, make sure 'python aethon.py @monitor slurmconfig' is running concurrently
#
# example call (runs @monitor concurrently): 'python aethon.py @array jobarray 0-55 @monitor slurmconfig @threads 16'
#

datasets = "hannover", "buxtehude", "nienburg", "potsdam", "vaihingen", "hameln_DA", "schleswig_DA", "toulouse_full"
n = len(datasets)

i = int(input())
i, j = i//n, i%n

dataset = datasets[j]
mem = (12, 12, 12, 44, 18, 12, 12, 25)[j]

models = "PFNet", "FCN", "DeepLabv3p", "RAFCN", "FarSeg", "UNet", "SegForestNet"
n = len(models)
i, j = i//n, i%n

model = models[j]
epochs, learning_rate = {
    "PFNet": (100, 0.005),
    "FCN": (120, 0.006),
    "DeepLabv3p": (80, 0.0035),
    "RAFCN": (120, 0.0002),
    "FarSeg": (120, 0.001),
    "UNet": (80, 0.00015),
    "SegForestNet": (200, 0.003)
}[model]

slurm_prefix = f"@slurm slurmconfig PG 30:00:00 {mem}G PG.tar.bz2 tmp/PG "
print(f"{slurm_prefix}PG {dataset} {model} {epochs} {learning_rate}")
