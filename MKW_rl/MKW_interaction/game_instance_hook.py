from dolphin import event, gui, controller, savestate, memory # type: ignore
# Note that the program runs from the main linesight folder
import os
import sys

# TODO: add user config for python packages or just use virtual env for windows idk lol
# either way this step is not required either on linux or because I have global python modules available.
# sys.path.append(os.path.expanduser("~") + "\\AppData\\Local\\programs\\python\\python312\\lib\\site-packages")
import time
import socket
import pickle

source_directory = os.getcwd()
sys.path.append(source_directory)
from config_files import config_copy

from MKW_rl.MKW_interaction.MKW_interface import MKW_Interface
from MKW_rl.MKW_interaction.MKW_data_translate import *
from mkw_scripts.Modules.mkw_classes import common
from mkw_scripts.Modules.mkw_classes.race_manager import RaceState
from mkw_scripts.Modules import mkw_config, mkw_utils

HOST = "127.0.0.1" # config_copy.HOST

class GameInstanceHook():
    def __init__(self, port=config_copy.base_tmi_port): # 8478
        self.desired_inputs = {
                "A": False,
                "B": False,
                "Up": False,
                "StickX": 0,
                "StickY": 0,
                "TriggerLeft": 0,
                "TriggerRight": 0
            }
        self.last_desired_inputs = {}
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
        self.ghost_saved = False
        self.waiting_for_rkg = False
        self.rkg_timer = 0
        self.closing = False

    def end_framedrawn_handling(self, width, height, data):
        return
    def end_frameadvance_handling(self):
        return

    def framedrawn_handler(self, width, height, data):
        # print(width, height)
        if self.waiting_for_rkg:
            return
        try:
            self.game_data_interface.initialize_race_objects()
            if self.game_data_interface.race_mgr.state() == RaceState.INTRO_CAMERA:
                return
        except Exception: # failed to initialize game data
            pass
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
                    # print("Restarting via countdown timer")
                    self.restarting_race = False
                    self.restarting_race_timer = 0
                # Skipped 1000 frames attempting to let the game load the race, so we continue instead.
                if self.restarting_race_timer > 1000:
                    # This behavior is likely unwanted but if it never runs then it never runs
                    print("ERROR: Restarting due to frame rule, track was not loaded within 1000 frames")
                    self.restarting_race = False
                    self.restarting_race_timer = 0
            return

        try:
            socket_data = pickle.loads(self.conn.recv(256))
        except Exception as e:
            # print(e)
            if not self.closing:
                print("Closing socket and exiting")
                self.closing = True
            self.close()
            event.on_framedrawn(self.end_framedrawn_handling)
            event.on_frameadvance(self.end_frameadvance_handling)
            return

        frame_data_request = socket_data[0]
        game_data_request = socket_data[1]
        new_inputs = socket_data[2]
        load_state_request = socket_data[3]

        if new_inputs is not None:
            self.desired_inputs = new_inputs

        if len(load_state_request) > 0:
            # Funky way of avoiding loading a savestate for every rollout (bad for performance, I think)
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
            self.conn.sendall(data)
            self.conn.recv(128)
        
        if game_data_request and self.desired_savestate is not None:
            if not self.game_data_initiated:
                self.game_data_interface.initialize_race_objects()
                self.game_data_initiated = True

            game_data = self.game_data_interface.get_game_data_object()
            for key in game_data["kart_data"].keys():
                value = game_data["kart_data"][key]
                if type(value) == vec3:
                    game_data["kart_data"][key] = [value.x, value.y, value.z]
                elif type(value) == quatf:
                    game_data["kart_data"][key] = [value.x, value.y, value.z, value.w]
            self.conn.sendall(pickle.dumps(game_data))
            self.last_game_data = game_data
        elif game_data_request:
            print("ERROR: game_data_request was sent before race state was loaded")
        else:
            self.conn.sendall("nothing happened".encode()) # frame skipped.
            
    def frameadvance_handler(self):
        self.frame_counter += 1
        # gui.draw_text((10, 10), self.red, f"Frame: {self.frame_counter}")

        # Save ghost of completed race
        if (not self.ghost_saved) and self.game_data_initiated and self.game_data_interface.race_mgr_player.race_completion_max() >= 4:
            self.waiting_for_rkg = True
            region = mkw_config.game_id_string
            try:
                address = {"RMCE01": 0x809B8F88, "RMCP01": 0x809BD748,
                        "RMCJ01": 0x809BC7A8, "RMCK01": 0x809ABD88}
                rkg_addr = mkw_utils.chase_pointer(address[region], [0x18], 'u32')
            except KeyError:
                raise common.RegionError
            if not memory.read_u32(rkg_addr) == 0x524b4744: # Thank you Blounard for the hex number
                self.rkg_timer += 1
                if 60 * 4 < self.rkg_timer: # Failed to save ghost in 4 seconds
                    self.ghost_saved = True
                    self.waiting_for_rkg = False
                    self.rkg_timer = 0
                return
            gui.add_osd_message("Saving ghost")
            fin_timer = self.game_data_interface.race_mgr_player.inst_race_finish_time()
            racetime = fin_timer.minutes() * 60 + fin_timer.seconds() + fin_timer.milliseconds() / 1000

            base_dir = config_copy.project_path # get base directory this program is running in
            save_dir = base_dir / "save" / config_copy.run_name / "all_runs" # put save data within this directory
            if not os.path.isdir(base_dir / "save" / config_copy.run_name / "all_runs"):
                os.mkdir(save_dir)
            filename = save_dir / f"{self.game_data_interface.race_settings.course_id().name}_{racetime:.3f}.rkg"
            with open(filename, 'wb') as f:
                f.write(common.read_bytes(rkg_addr, 0x2800))

            self.ghost_saved = True
            self.waiting_for_rkg = False
            self.rkg_timer = 0 # reset rkg_timer as we succeeded in saving the ghost

        if self.load_state_desired:
            self.load_state_desired = False
            if self.desired_savestate.startswith("__slot__"):
                savestate.load_from_slot(int(self.desired_savestate[8:]))
                self.conn.sendall("state_loaded".encode()) # maintain baton-passing of send and recv calls
            else:
                savestate.load_from_file(self.desired_savestate)
                self.conn.sendall("state_loaded".encode()) # maintain baton-passing of send and recv calls
                """
                If multiple sends occur on one side of the connection before the other side has called a recv,
                both programs will hang waiting in a recv as the following send calls are lost in the buffer somehow.
                Rather than trying to understand that, we always alternate sending and receiving on both sides.
                """
            self.game_data_initiated = False
            self.ghost_saved = False
            self.waiting_for_rkg = False
            self.rkg_timer = 0
            # print("Loaded new savestate:", self.desired_savestate)

        controller.set_gc_buttons(0, self.desired_inputs)

    def register(self):
        print("Initialize connection from Dolphin")

        success = False
        fails = 0
        while not success:
            try:
                self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.listener.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.listener.bind((HOST, self.port))
                self.listener.listen(1)
                self.conn, _ = self.listener.accept()
                success = True
            except Exception as e:
                print(e)
                fails += 1
                if fails > 10:
                    print("Error connecting to program.")
                    success = True # just let the puppy crash lol
        print("Connection accepted")

    def close(self):
        self.listener.close()

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

read_counter = 0
connected = False
while not connected:
    try:
        with open(str(config_copy.project_path) + "/dolphin_ports/pid_" + str(os.getpid())) as f:
            connection_port = int(f.read())
            mymanager = GameInstanceHook(config_copy.base_tmi_port + connection_port)
            connected = True
    except Exception as e:
        print(e)
        read_counter += 1
        if read_counter > 100:
            raise
        time.sleep(0.1)

mymanager.register()

event.on_framedrawn(mymanager.framedrawn_handler)
event.on_frameadvance(mymanager.frameadvance_handler)