"""
This file defines the game ID to use for all scripts in mkw-scripts
The game ID is dependent on the region of your game file.

If unsure, check the flag Dolphin displays in the menu and match it with the value in quotations below.
USA: NTSC-U -- "RMCE01"
Europe: PAL -- "RMCP01"
Japan: NTSC-J -- "RMCJ01"
Republic of Korea: NTSC-K -- "RMCK01"
"""

from config_files.user_config import game_region
game_id_string = game_region