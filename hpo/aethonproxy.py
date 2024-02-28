import subprocess
import signal
import types
import threading
import time
import re


def monitor(shutdown):
    process = subprocess.Popen((python_bin, "aethon.py", "@monitor", slurm_config[0]), cwd="..")
    while not shutdown.wait(5):
        pass
    process.send_signal(signal.SIGINT)
    process.wait()

    
class init():
    def __init__(self):
        assert "python_bin" in globals()
        assert "slurm_config" in globals()
        self.shutdown = threading.Event()
        self.thread = threading.Thread(target=monitor, args=(self.shutdown,))
        self.thread.start()
        print("waiting for SLURM monitoring to start ...")
        time.sleep(15)
        
    def close(self):
        self.shutdown.set()
        self.thread.join()


def call(params):
    process = subprocess.run(
        (python_bin, "aethon.py", "@slurm", *slurm_config, *[str(p) for p in params]),
        cwd="..", text=True, capture_output=True
    )
    if process.returncode != 0:
        raise RuntimeError()
    r = re.compile(r"job ID: (?P<job_id>\d+)")
    for line in process.stdout.split("\n"):
        m = r.match(line.strip())
        if not m is None:
            return f"../{slurm_config[-2]}/" + m["job_id"]
    raise RuntimeError()
