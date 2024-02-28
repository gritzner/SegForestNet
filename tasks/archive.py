import core
from datetime import datetime
from tempfile import TemporaryDirectory
import subprocess
import sys
import glob
import os


def add_path_to_archive(archive_fn, path):
    for fn in glob.iglob(f"{path}*"):
        if fn in ("git_log.txt", "hpo", "tmp", "rust/Cargo.lock", "rust/src/exports.rs", "rust/target", archive_fn):
            continue
        if os.path.isfile(fn):
            core.call(f"tar -rf {archive_fn} {fn}")
        elif fn.split("/")[-1] != "__pycache__":
            add_path_to_archive(archive_fn, f"{fn}/")


def archive():
    dt = datetime.now()
    archive_fn = f"Aethon-{dt.year}_{dt.month:02}_{dt.day:02}-{dt.hour:02}_{dt.minute:02}_{dt.second:02}.tar"
        
    with TemporaryDirectory() as temp_dir:
        core.call(f"git log -{core.user.git_log_num_lines} > {temp_dir}/git_log.txt")
        core.call(f"tar -cf {archive_fn} -C {temp_dir} git_log.txt")
        core.call(f"mkdir -p {temp_dir}/.cargo")
        with open(".cargo/config.toml", "r") as f_in:
            with open(f"{temp_dir}/.cargo/config.toml", "w") as f_out:
                f_out.write(f'''[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "vendor"

{f_in.read()}
''')
        core.call(f"tar -rf {archive_fn} -C {temp_dir} .cargo/config.toml")
        ret_val = subprocess.call(f"cargo vendor --manifest-path {core.base_path}/rust/Cargo.toml", shell=True, cwd=temp_dir)
        if ret_val != 0:
            sys.exit(ret_val)
        for fn in glob.iglob(f"{temp_dir}/vendor/**", recursive=True):
            core.call(f"tar -rf {archive_fn} -C {temp_dir} {fn[len(temp_dir)+1:]}")
    
    add_path_to_archive(archive_fn, "")
    core.call(f"bzip2 -f {archive_fn}")
