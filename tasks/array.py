import core
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
import signal
import time
import argparse
import os
import json
import types
from itertools import repeat


def run_task(args, i):
    process = subprocess.run(
        (sys.executable, f"cfgs/{args.configuration}.py"), input=str(i[1]), text=True, capture_output=True
    )
    if process.returncode != 0:
        print(f"[error@{i}] {process.stderr.strip()}")
        return process, None
    s = process.stdout.strip()
    print(f"[{i}] {s}")
    if not args.print_only:
        if len(args.log) > 0:
            return process, subprocess.run(("python", "aethon.py", *s.split(" ")), text=True, capture_output=True)
        else:
            return process, types.SimpleNamespace(returncode=core.call(f"python aethon.py {s}"), stdout="", stderr="")
    return process, None


def proxy_func(args, log, lock, i, j):
    i = i, j
    process0, process1 = run_task(args, i)
    returncode = 0
    if process0.returncode != 0:
        returncode = process0.returncode
    elif (not process1 is None) and process1.returncode != 0:
        returncode = process1.returncode
    if len(args.log) == 0:
        return returncode

    results = {}
    for j, process in enumerate((process0, process1)):
        if process is None:
            continue
        results[f"process{j}"] = {
            "returncode": process.returncode,
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip()
        }
    
    lock.acquire()
    log[str(i)] = results
    with open(args.log, "w") as f:
        json.dump(log, f)
    lock.release()
    
    return returncode


def monitor(config, shutdown):
    process = subprocess.Popen((sys.executable, "aethon.py", "@monitor", config))
    while not shutdown.wait(5):
        pass
    process.send_signal(signal.SIGINT)
    process.wait()


def array():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0])+" @array", description="Task Array")
    parser.add_argument("configuration", type=str, help="configuration file")
    parser.add_argument("array_range", type=str, help="array range, e.g., 0-9 (end inclusive)")
    parser.add_argument("--threads", default=0, type=int, help="set the number of worker threads to use")
    parser.add_argument("--print-only", action="store_true", help="only print resulting Aethon parameters but do not execute")
    parser.add_argument("--monitor", default="", type=str, help="start SLURM monitor thread using the supplied configuration")
    parser.add_argument("--log", default="", type=str, help="capture results of tasks and log them to a JSON file")
    args = parser.parse_args([f"--{arg[1:]}" if arg[0]=="@" else arg for arg in core.args.parameters])
    
    if len(args.monitor) > 0:
        shutdown = threading.Event()
        monitor_thread = threading.Thread(target=monitor, args=(args.monitor, shutdown))
        monitor_thread.start()
        print("waiting for SLURM monitoring to start ...")
        time.sleep(15)
    
    if "," in args.array_range:
        temp = args.array_range.split(",")
    else:
        temp = [args.array_range]
    args.array_range = []
    for array_range in temp:
        if "x" in array_range:
            n, array_range = array_range.split("x")
            args.array_range.extend([array_range] * int(n))
        else:
            args.array_range.append(array_range)
    del temp
    for i, array_range in enumerate(args.array_range):
        if "-" in array_range:
            array_range = [int(x) for x in array_range.split("-")]
            assert len(array_range) == 2
        else:
            array_range = [int(array_range)] * 2
        array_range[1] += 1
        args.array_range[i] = range(*array_range)
    
    tasks = []
    log = {}
    returncode = 0
    try:
        if args.threads > 0:
            lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=args.threads) as thread_pool:
                tasks = [thread_pool.map(lambda i, j: proxy_func(args, log, lock, i, j), repeat(i), array_range) for i, array_range in enumerate(args.array_range)]
        else:
            lock = types.SimpleNamespace(acquire=lambda : None, release=lambda : None)
            tasks = [[proxy_func(args, log, lock, i, j) for j in array_range] for i, array_range in enumerate(args.array_range)]
        for task_set in tasks:
            for i in task_set:
                if i != 0:
                    returncode = i
    except KeyboardInterrupt:
        returncode = 1
    
    if len(args.monitor) > 0:
        shutdown.set()
        monitor_thread.join()
    
    sys.exit(returncode)
