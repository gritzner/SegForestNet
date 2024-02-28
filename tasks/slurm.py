import core
import yaml
import subprocess
import re
import sys
import socket
import time


def slurm():
    assert len(core.args.parameters) > 6
    config, jobname, jobtime, mem, code, output_path = core.args.parameters[:6]
    
    with open(f"{core.base_path}/cfgs/{config}.yaml", "r") as f:
        config = core.parse_dict(yaml.safe_load(f.read()))
    prefix = ("ssh", config.servers.login) if hasattr(config, "servers") else ()
    config.slurm.options["job-name"] = jobname
    config.slurm.options["time"] = jobtime
    config.slurm.options["mem"] = mem
    config.slurm.options["chdir"] = config.slurm.jobs_path
    slurm_options = "\n".join([f"#SBATCH --{str(k)}={str(v)}" for k, v in config.slurm.options.items()])
    script = config.slurm.script.format(
        slurm_options = slurm_options,
        jobs_path = config.slurm.jobs_path,
        code = code,
        parameters = " ".join([str(p) for p in core.args.parameters[6:]])
    )
    
    process = subprocess.run(
        (*prefix, "sbatch"), input=script, text=True, capture_output=True
    )
    print("[stdout]", process.stdout.strip())
    print("[stderr]", process.stderr.strip())
    print()
    if process.returncode != 0:
        print("sbatch failed")
        sys.exit(process.returncode)

    m = re.search(r"Submitted batch job (?P<job_id>\d+)", process.stdout)
    if m is None:
        print("job ID not found")
        sys.exit(1)
    jobid = int(m["job_id"])
    print("job ID:", jobid)
    
    with socket.create_connection(("localhost", config.slurm.monitor.port)) as client:
        client.send(f"register:{jobid}".encode())
    while True:
        time.sleep(config.slurm.monitor.timer)
        with socket.create_connection(("localhost", config.slurm.monitor.port)) as client:
            client.send(f"query:{jobid}".encode())
            if core.str2bool(client.recv(256).decode()):
                break

    log_path = f"{config.slurm.jobs_path}/slurm-{jobid}.out"
    output_path = f"{output_path}/{jobid}"
    core.call(f"mkdir -p {output_path}")
    if hasattr(config, "servers"):
        core.call(f"ssh {config.servers.login} mv {log_path} {config.slurm.jobs_path}/{jobid}/tmp")
        core.call(f"rsync -aqz {config.servers.transfer}:{config.slurm.jobs_path}/{jobid}/tmp/ {output_path}")
        core.call(f"ssh {config.servers.login} rm -rf {config.slurm.jobs_path}/{jobid}")
    else:
        core.call(f"mv {log_path} {config.slurm.jobs_path}/{jobid}/tmp")
        core.call(f"rsync -aqz {config.slurm.jobs_path}/{jobid}/tmp/ {output_path}")
        core.call(f"rm -rf {config.slurm.jobs_path}/{jobid}")
