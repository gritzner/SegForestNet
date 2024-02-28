import core
core.init()

import tasks
if core.args.configuration[0] != "@":
    task = core.create_object(tasks, core.task)
    task.run()
    core.call(f"touch {core.output_path}/done")
else:
    import os
    if "LD_LIBRARY_PATH" in os.environ:
        del os.environ["LD_LIBRARY_PATH"] # fixes an issue with calling ssh after cv2 has been imported
    getattr(tasks, core.args.configuration[1:])()
