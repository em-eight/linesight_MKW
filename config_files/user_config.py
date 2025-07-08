"""
This file contains user-level configuration.
It is expected that the user fills this file once when setting up the project, and does not need to modify it after.
"""

import os
from pathlib import Path
from sys import platform

is_linux = platform in ["linux", "linux2"]

# Path where Python_Link.as should be placed so that it can be loaded in TMInterface.
# Usually Path(os.path.expanduser("~")) / "Documents" / "TMInterface" / "Plugins" / "Python_Link.as"
# --Unused for this project
target_python_link_path = Path(os.path.expanduser("~")) / "Documents" / "TMInterface" / "Plugins" / "Python_Link.as"

# Path to the folder of the cloned repository
project_path = Path(os.path.expanduser("~")) / "Documents" / "Python" / "Linesight" / "linesight_MKW"

project_scripts_path = project_path / "mkw_scripts"

# Directory that dolphin is working from (The main dolphin folder)
dolphin_base_path = Path(os.path.expanduser("~")) / "Programs" / "dolphin"

# If on Linux, path of the dolphin executable, starting at the dolphin_base_path
linux_launch_game_path = "/Build/Binaries/dolphin-emu"

# Communication port for the first Dolphin instance that will be launched.
# If using multiple instances, the ports used will be base_tmi_port + 1, +2, +3, etc...
base_tmi_port = 8478

# If on windows, name of the TMLoader profile that with launch TmForever + TMInterface
# --Unused for this project
windows_TMLoader_profile_name = "default"

# Location of the MKW game file/folder (usually a .rvz or .iso)
# Note that if this is a folder this should link to the boot file, not the folder itself (Maybe, I haven't tested).
game_path = Path(os.path.expanduser("~")) / "Programs" / "MKWii" / "Mario_Kart_Wii.rvz"

"""
The game_region defines the game ID to use for all scripts in mkw-scripts
The game ID is dependent on the region of your game file.

If unsure, check the flag/version Dolphin displays in the menu and match it with the value in quotations below.
USA: NTSC-U -- "RMCE01"
Europe: PAL -- "RMCP01"
Japan: NTSC-J -- "RMCJ01"
Republic of Korea: NTSC-K -- "RMCK01"
Note that Gecko codes are region specific. You must use/copy from the provided matching .ini file for them to work.
"""
game_region = "RMCE01"

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
