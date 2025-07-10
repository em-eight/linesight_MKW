# Dolphin test scripts
These files have been used for extensive testing of several parts of this repository.
As such, they are currently messy and have a large amount of unnecessary functionality

To use these scripts, launch the scripting dolphin, then in the scripts menu, add game_instance_hook.py.
Currently the script will throw errors until the game loads into a race. This is normal.
From there, it will track the players position on every frame, as well as displaying certain game values.
The user is encouraged to experiment with this file to develop an understanding of how MKWii works.

# Creating map.npy files
Once loaded into the race with the game_instance_hook running, it should stop printing errors.
Drive the race as normal, using the mushroom in the normal spot for the track you want to create the .npy file for.
Be sure not to respawn (fall out of bounds) during your run. Respawning messes with the position tracker and I have observed that it breaks the vcp logic in the training codebase. I have not fixed this yet.

Once the race is finished, you should see the file "map.npy" in the directory you specified. Rename this file to match the track and you are done. (In the future I may add an additional file to plot the generated .npy to confirm that the file is correct)