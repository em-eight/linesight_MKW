from dolphin import event, gui, controller, savestate # type: ignore
# Note that the program runs from the main linesight folder
from MKW_rl.MKW_interaction.MKW_interface import MKW_Interface
import os
import sys
sys.path.append(os.path.expanduser("~") + "\\AppData\\Local\\programs\\python\\python312\\lib\\site-packages")

from multiprocessing.connection import Listener
from config_files import config_copy
import inspect
source_file_path = inspect.getfile(inspect.currentframe())

from MKW_rl.MKW_interaction.MKW_data_translate import *
from mkw_scripts.Modules.mkw_classes.race_manager import RaceState

HOST = "127.0.0.1"

class GameInstanceHook():
    def __init__(self, port=8478):
        self.desired_inputs = {}
        self.last_desired_inputs = {}
        self.current_unprocessed_frame = None
        self.resized = None
        self.frame_counter = 0
        self.red = 0xffff0000
        self.listener = None
        self.conn = None
        self.game_data_initiated = False
        self.game_data_interface = MKW_Interface()
        self.port = port
        self.load_state_desired = False
        self.desired_savestate = None
        self.last_game_data = None
        self.restarting_race = False
        self.restarting_race_timer = 0

    def framedrawn_handler(self, width, height, data):
        if self.restarting_race:
            if self.restarting_race_timer < 3:
                self.desired_inputs = {
                    "A": False,
                    "B": False,
                    "Up": False,
                    "StickX": 0,
                    "StickY": 0,
                    "TriggerLeft": 0,
                    "TriggerRight": 0,
                    "Start": True,
                }
                self.restarting_race_timer += 1
            elif self.restarting_race_timer >= 3 and self.restarting_race_timer < 7:
                self.desired_inputs = {
                    "A": False,
                    "B": False,
                    "Up": False,
                    "StickX": 0,
                    "StickY": -1,
                    "TriggerLeft": 0,
                    "TriggerRight": 0,
                    "Start": False,
                }
                self.restarting_race_timer += 1
            elif self.restarting_race_timer >= 7 and self.restarting_race_timer < 12:
                self.desired_inputs = {
                    "A": True,
                    "B": False,
                    "Up": False,
                    "StickX": 0,
                    "StickY": 0,
                    "TriggerLeft": 0,
                    "TriggerRight": 0,
                    "Start": False,
                }
                self.restarting_race_timer += 1
            else:
                self.restarting_race_timer += 1
                # Reload game objects awaiting countdown start
                self.game_data_interface.initialize_race_objects()
                if self.game_data_interface.race_mgr.state() == RaceState.COUNTDOWN:
                    print("Restarting via countdown timer")
                    self.restarting_race = False
                    self.restarting_race_timer = 0
                # Skipped 1000 frames attempting to let the game load the race, so we continue instead.
                if self.restarting_race_timer > 1000:
                    # This behavior is likely unwanted but if it never runs then it never runs
                    print("ERROR: Restarting due to frame rule, track was not loaded within 1000 frames")
                    self.restarting_race = False
                    self.restarting_race_timer = 0
            return
        
        # Wait for data necessary to determine what we want to do
        self.current_unprocessed_frame = (height, width, data)

        # print("Now waiting for new request")
        # https://stackoverflow.com/questions/38412887/how-to-send-a-list-through-tcp-sockets-python
        socket_data = self.conn.recv()
        # print("Received:", socket_data)

        frame_data_request = socket_data[0]
        game_data_request = socket_data[1]
        new_inputs = socket_data[2]
        load_state_request = socket_data[3]

        if new_inputs is not None:
            self.desired_inputs = new_inputs

        if load_state_request is not None:
            # Funky way of avoiding loading a savestate for every rollout (bad for performance)
            if socket_data[3] == config_copy.restart_race_command:
                self.restarting_race = True
                self.desired_inputs = {
                    "A": False,
                    "B": False,
                    "Up": False,
                    "StickX": 0,
                    "StickY": 0,
                    "TriggerLeft": 0,
                    "TriggerRight": 0,
                    "Start": True,
                }
                return
            self.load_state_desired = True
            self.desired_savestate = socket_data[3]
            if frame_data_request:
                print("ERROR: Savestate file received but got unactionable frame data request")
            if game_data_request:
                print("ERROR: Savestate file received but got unactionable game data request")
            return

        if frame_data_request:
            self.conn.send_bytes(self.current_unprocessed_frame[2]) # unsure if i should pre-process frame or not...

            # width * height * 4, socket.MSG_WAITALL # server recv to receive frame data because it's big

            """# The following line brought to you by literal hours of trying to figure things out only to realize I just needed two functions that I could've just copied from the original code
            processed_frame = numpy.frombuffer(self.current_unprocessed_frame[2], dtype = numpy.uint8).reshape((self.current_unprocessed_frame[0], self.current_unprocessed_frame[1], 3))
            # https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
            resized_frame = processed_frame[::6,::6]
            resized_frame = numpy.expand_dims(cv2.cvtColor(resized_frame, cv2.COLOR_BGRA2GRAY), 0) # took me like 80 minutes to get to the solution that was already present in the original code
            # frame is a numpy array of shape (1, H, W) and dtype np.uint8
            print(processed_frame.shape, ":", resized_frame.shape)"""
        
        if game_data_request and self.desired_savestate is not None:
            if not self.game_data_initiated:
                self.game_data_interface.initialize_race_objects()
                self.game_data_initiated = True

            game_data = self.game_data_interface.get_game_data_object()
            for key in game_data["kart_data"].keys():
                value = game_data["kart_data"][key]
                if type(value) == vec3:
                    game_data["kart_data"][key] = [value.x, value.y, value.z]
            self.conn.send(game_data)
            self.last_game_data = game_data
        elif game_data_request:
            print("ERROR: game_data_request was sent before race state was loaded")
        # send the image data here, so we can set desired_inputs before we exit the function
            
    def frameadvance_handler(self):
        self.frame_counter += 1
        # gui.draw_text((10, 10), self.red, f"Frame: {self.frame_counter}")

        if self.load_state_desired:
            self.load_state_desired = False
            if self.desired_savestate.startswith("__slot__"):
                savestate.load_from_slot(int(self.desired_savestate[8:]))
            else:
                savestate.load_from_file(self.desired_savestate)
            self.game_data_initiated = False
            # print("Loaded new savestate:", self.desired_savestate)

        controller.set_gc_buttons(0, self.desired_inputs)

    def register(self):
        print("Initialize connection to Dolphin ")

        success = False
        fails = 0
        while not success:
            try:
                self.listener = Listener((HOST, self.port))
                print("Game hook socket listening on port", self.port)
                self.conn = self.listener.accept()
                success = True
            except OSError:
                self.port += 1 # Expert level port finding code
                fails += 1
                if fails > 10:
                    print("Error connecting to program.")
                    success = True # just let the puppy crash lol
        print("Connected accepted from:", self.listener.last_accepted)

# Use base tmi port in an attempt to find the right port lol
mymanager = GameInstanceHook(config_copy.base_tmi_port) #  + len(get_dolphin_pids()) - 1
print("Working from port", mymanager.port)
print("Working in directory", source_file_path)
"""
Register the socket, ensure it is connected
when framedrawn_handler is called, we read from the socket, waiting if necessary.
We process
    1. Whether a frame is requested, then sending data
    2. Whether any game data is requested, then sending data
    3. Whether to apply new inputs
Next, when frameadvance_handler is called
    1. Draw debug information to screen (inputs, framecount, fps?)
    2. Load savestate if necessary
    3. Apply inputs to the game
"""

mymanager.register()


event.on_framedrawn(mymanager.framedrawn_handler)
event.on_frameadvance(mymanager.frameadvance_handler)