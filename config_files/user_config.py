"""
This file contains user-level configuration.
It is expected that the user fills this file once when setting up the project, and does not need to modify it after.
Note that there is an additional config file located in mkw-scripts/mkw_config.py to mark the region of your MKW game file.
""" # TODO: remove excess mkw_config.py file and port into user config

import os
from pathlib import Path
from sys import platform

is_linux = platform in ["linux", "linux2"]

# Path where Python_Link.as should be placed so that it can be loaded in TMInterface.
# Usually Path(os.path.expanduser("~")) / "Documents" / "TMInterface" / "Plugins" / "Python_Link.as"
# -Likely to be unused for this project
target_python_link_path = Path(os.path.expanduser("~")) / "Documents" / "TMInterface" / "Plugins" / "Python_Link.as"

project_path = Path(os.path.expanduser("~")) / "Documents" / "Python" / "MKW_linesight" / "linesight_MKW"

project_scripts_path = Path(os.path.expanduser("~")) / "Documents" / "Python" / "MKW_linesight" / "linesight_MKW" / "mkw_scripts"

dolphin_base_path = Path(os.path.expanduser("~")) / "Documents" / "Python" / "MKW_linesight" / "dolphin-stable"

# Communication port for the first TMInterface instance that will be launched.
# If using multiple instances, the ports used will be base_tmi_port + 1, +2, +3, etc...
base_tmi_port = 8478

# If on Linux, path of a shell script that launches the game, with the TMInterface port as first argument
linux_launch_game_path = "path_to_be_filled_only_if_on_linux"

# If on windows, name of the TMLoader profile that with launch TmForever + TMInterface
windows_TMLoader_profile_name = "default"

# Location of the game file for MKW
game_path = "C:\\Games\\Dolphin-x64\\Games\\MKW_Copies\\Mario_Kart_Wii.rvz"

# If on windows, path where the Dolphin exe can be found.
# Usually Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"
windows_dolphinexe_path = Path(os.path.expanduser("~")) / "Documents" / "Python" / "MKW_linesight" / "dolphin-stable" / "Dolphin.exe"

# Name of the Dolphin executable to run
windows_dolphinexe_filename = "Dolphin.exe"

# Video backend for Dolphin to use (see Dolphin Command Line Usage)
# Options include D3D, D3D12 (both Windows exclusive), OGL, Vulkan, Null (Game will not be rendered), and SoftwareRenderer
video_backend = "Vulkan"

# Dolphin emulation speed. Usually set to unlimited (0.0), but can be set to 0.5 for 100% or 1.0 for 200% and so forth. I do not know why it is multiplied by 2, just be aware that it is.
game_speed = "0.0"
