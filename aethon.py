import core
core.init()
# IMPORTANT: Do not add any imports above here otherwise you may break the code in core.init() that limits the number of threads!

import utils
import sys
import atexit
import tasks
import os

if core.args.notify:
    argv = " ".join([arg for arg in sys.argv if arg != "--notify"])
    if hasattr(core, "embedded_parameters"):
        argv = f"{argv}\n\nembedded parameters: {str(core.embedded_parameters)}"
    atexit.register(utils.push_notification, core.args.configuration, argv)

task = core.create_object(tasks, core.task)
task.run()
os.system(f"touch {core.output_path}/done")

if core.args.notify and hasattr(task, "push_notification"):
    atexit.unregister(utils.push_notification)
    utils.push_notification(core.args.configuration, f"{argv}\n\n{task.push_notification}")
