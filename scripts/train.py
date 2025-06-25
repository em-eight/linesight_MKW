# =======================================================================================================================
# Handle configuration file before running the actual content of train.py
# =======================================================================================================================
"""
Two files named "config.py" and "config_copy.py" coexist in the same folder.

At the beginning of training, parameters are copied from config.py to config_copy.py
During training, config_copy.py will be reloaded at regular time intervals.
config_copy.py is NOT tracked with git, as it is essentially a temporary file.

Training parameters modifications made during training in config_copy.py will be applied on the fly
without losing the existing content of the replay buffer.

The content of config.py may be modified after starting a run: it will have no effect on the ongoing run.
This setup provides the possibility to:
  1) Modify training parameters on the fly
  2) Continue to code, use git, and modify config.py without impacting an ongoing run.
"""

import shutil
from pathlib import Path


def copy_configuration_file(): # save config file so we don't overwrite it and you can edit it while RL is happening
    base_dir = Path(__file__).resolve().parents[1]
    shutil.copyfile(
        base_dir / "config_files" / "config.py",
        base_dir / "config_files" / "config_copy.py",
    )


if __name__ == "__main__":
    copy_configuration_file()

# =======================================================================================================================
# Actual start of train.py, after copying config.py
# =======================================================================================================================

import ctypes
import os
import random
import signal
import sys, inspect
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from art import tprint
from torch.multiprocessing import Lock


"""cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"/config_files")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)"""


from config_files import config_copy
from MKW_rl.agents.iqn import make_untrained_iqn_network
from MKW_rl.multiprocess.collector_process import collector_process_fn
from MKW_rl.multiprocess.learner_process import learner_process_fn

# Set torch settings for RL
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
torch.set_float32_matmul_precision("high")
random_seed = 444
torch.cuda.manual_seed_all(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)


def signal_handler(sig, frame): # receive command to kill game instances
    print("Received SIGINT signal. Killing all open Dolphin instances.")
    clear_tm_instances()
    clear_port_files()

    for child in mp.active_children():
        child.kill()

    tprint("Bye bye!", font="tarty1")
    sys.exit()


def clear_tm_instances(): # stop all instances of the game
    if config_copy.is_linux:
        os.system("pkill -9 Dolphin.exe")
    else:
        os.system("taskkill /F /IM Dolphin.exe")

def clear_port_files(): # Remove port files
    # https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
    for file in os.listdir("dolphin_ports"):
        filename = os.fsdecode(file)
        if filename.startswith("pid_"):
            os.remove(config_copy.project_path / "dolphin_ports" / file)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    clear_tm_instances() # ensure we're starting fresh here, and old instances aren't used

    base_dir = Path(__file__).resolve().parents[1] # get base directory this program is running in
    save_dir = base_dir / "save" / config_copy.run_name # put save data within this directory
    save_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_base_dir = base_dir / "tensorboard" # save tensorboard information

    """# Copy Angelscript plugin to TMInterface dir
    shutil.copyfile(
        base_dir / "MKW_rl" / "MKW_interaction" / "Python_Link.as",
        config_copy.target_python_link_path,
    )"""

    print("Run:\n\n")
    tprint(config_copy.run_name, font="tarty4")
    print("\n" * 2)
    tprint("Linesight", font="tarty1")
    print("\n" * 2)
    print("Training is starting!")

    # Prepare multi process utilities
    shared_steps = mp.Value(ctypes.c_int64)
    shared_steps.value = 0
    rollout_queues = [mp.Queue(config_copy.max_rollout_queue_size) for _ in range(config_copy.gpu_collectors_count)]
    shared_network_lock = Lock()
    game_spawning_lock = Lock()
    _, uncompiled_shared_network = make_untrained_iqn_network(jit=config_copy.use_jit, is_inference=False) # initialize the network
    uncompiled_shared_network.share_memory() # share network memory for multi-threading

    # Start worker process
    collector_processes = [
        mp.Process(
            target=collector_process_fn,
            args=(
                rollout_queue,
                uncompiled_shared_network,
                shared_network_lock,
                game_spawning_lock,
                shared_steps,
                base_dir,
                save_dir,
                config_copy.base_tmi_port + process_number,
                process_number,
            ),
        )
        for rollout_queue, process_number in zip(rollout_queues, range(config_copy.gpu_collectors_count)) # create specified number of instances for RL
    ]
    for collector_process in collector_processes:
        collector_process.start()

    # Start learner process
    print("Train.py: Starting learner process")
    learner_process_fn(rollout_queues,uncompiled_shared_network,shared_network_lock,shared_steps,base_dir,save_dir,tensorboard_base_dir) #Turn main process into learner process instead of starting a new one, this saves 1 CUDA context

    for collector_process in collector_processes:
        collector_process.join() # combine processes for learning and/or completion?
