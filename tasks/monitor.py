import core
import yaml
import threading
import signal
import functools
import types
import socket
import subprocess


def monitor():
    assert len(core.args.parameters) > 0
    with open(f"{core.base_path}/cfgs/{core.args.parameters[0]}.yaml", "r") as f:
        config = core.parse_dict(yaml.safe_load(f.read()))
        
    def monitor(state):
        prefix = ("ssh", config.servers.login) if hasattr(config, "servers") else ()
        while True:
            try:
                process = subprocess.run(
                    (*prefix, "squeue", "-u", config.slurm.monitor.user, "-o", "%i"),
                    timeout=config.slurm.monitor.remote_timeout, text=True, capture_output=True
                )
            except subprocess.TimeoutExpired:
                continue
            if process.returncode == 0:
                jobs = set()
                ignored_jobs = set()
                for jobid in process.stdout.split("\n")[1:-1]:
                    if "_" in jobid:
                        ignored_jobs.add(jobid)
                    else:
                        jobs.add(int(jobid))
                if config.slurm.monitor.verbose:
                    print("jobs in queue:", jobs)
                    print("ignored jobs:", ignored_jobs)
                with state.lock:
                    state.jobs.clear()
                    state.jobs.update(jobs)
            if state.shutdown.wait(config.slurm.monitor.timer):
                return
    
    state = types.SimpleNamespace(
        shutdown = threading.Event(),
        lock = threading.Lock(),
        jobs = set()
    )
    def signal_handler(sig, frame, state):
        print("shutting down ...")
        state.shutdown.set()
    signal.signal(signal.SIGINT, functools.partial(signal_handler, state=state))

    monitor_thread = threading.Thread(target=monitor, args=(state,))
    monitor_thread.start()

    server = socket.create_server(("localhost", config.slurm.monitor.port))
    server.settimeout(config.slurm.monitor.timeout)
    
    print(f"server monitoring '{core.args.parameters[0]}' is running on port {config.slurm.monitor.port} ...")
    while not state.shutdown.is_set():
        try:
            conn, _addr = server.accept()
            data = conn.recv(256).decode()
            tokens = data.split(":")
            if len(tokens) != 2:
                print("ignoring invalid message:", data)
            else:
                msg, jobid = tokens[0].lower(), int(tokens[1])
                if msg == "register":
                    with state.lock:
                        state.jobs.add(jobid)
                elif msg == "query":
                    with state.lock:
                        done = not jobid in state.jobs
                    conn.send(str(done).encode())
                else:
                    print("ignoring unknown command:", data)
        except TimeoutError:
            pass
    
    server.close()
    monitor_thread.join()
