from MKW_rl.MKW_interaction import MKW_data_translate
from config_files import config_copy, user_config

import math
import os
import socket
import subprocess
import time
from typing import Callable, Dict, List
import pickle

from MKW_rl.MKW_interaction.MKW_data_translate import *
from MKW_rl import map_loader
if config_copy.use_pynoko:
    import pynoko

import cv2
import numba
import numpy as np
import numpy.typing as npt
import psutil

HOST = "127.0.0.1"
FRAME_WIDTH = 611
FRAME_HEIGHT = 456

# @numba.njit
def update_current_zone_idx(
    current_zone_idx: int,
    zone_centers: npt.NDArray,
    sim_state_position: npt.NDArray,
    max_allowable_distance_to_virtual_checkpoint: float,
):
    d1 = np.linalg.norm(zone_centers[current_zone_idx + 1] - sim_state_position)
    d2 = np.linalg.norm(zone_centers[current_zone_idx] - sim_state_position)
    d3 = np.linalg.norm(zone_centers[current_zone_idx - 1] - sim_state_position)
    while (
        d1 <= d2
        and d1 <= max_allowable_distance_to_virtual_checkpoint
        and current_zone_idx
        < len(zone_centers) - 1 - config_copy.n_zone_centers_extrapolate_after_end_of_map  # We can never enter the final virtual zone
    ):
        # Move from one virtual zone to another
        current_zone_idx += 1
        d2, d3 = d1, d2
        d1 = np.linalg.norm(zone_centers[current_zone_idx + 1] - sim_state_position)
    while current_zone_idx >= 2 and d3 < d2 and d3 <= max_allowable_distance_to_virtual_checkpoint:
        current_zone_idx -= 1
        d1, d2 = d2, d3
        d3 = np.linalg.norm(zone_centers[current_zone_idx - 1] - sim_state_position)
    return current_zone_idx

class GameManager:
    def __init__(
        self,
        game_spawning_lock,
        running_speed=1,
        run_steps_per_action=10,
        max_overall_duration_f=2000,
        max_minirace_duration_f=2000,
        tmi_port=None,
        process_number=None
    ):
        # Create TMInterface we will be using to interact with the game client
        self.iface = None
        self.sock = None
        self.latest_tm_engine_speed_requested = 1
        self.running_speed = running_speed
        self.run_steps_per_action = run_steps_per_action
        self.max_overall_duration_f = max_overall_duration_f
        self.max_minirace_duration_f = max_minirace_duration_f
        self.timeout_has_been_set = False
        self.msgtype_response_to_wakeup_TMI = None
        self.latest_savestate_path_requested = -2
        self.last_rollout_crashed = False
        self.last_game_reboot = time.perf_counter()
        self.UI_disabled = False
        self.tmi_port = tmi_port
        self.dolphin_process_id = None
        self.start_states = {} # oh hey I might want to use this for starting later on in the track
        self.game_spawning_lock = game_spawning_lock
        self.game_activated = False
        self.process_number = process_number
        self.pynoko_system = None
        self.pynoko_race_completion_max = 0

    def register(self, timeout=None):
        # https://stackoverflow.com/questions/6920858/interprocess-communication-in-python
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((HOST, self.tmi_port))
        """# signal.signal(signal.SIGINT, self.signal_handler) # Handle close game signal
        # https://stackoverflow.com/questions/45864828/msg-waitall-combined-with-so-rcvtimeo
        # https://stackoverflow.com/questions/2719017/how-to-set-timeout-on-pythons-socket-recv-method
        if timeout is not None:
            if config_copy.is_linux:  # https://stackoverflow.com/questions/46477448/python-setsockopt-what-is-worng
                timeout_pack = struct.pack("ll", timeout, 0)
            else:
                timeout_pack = struct.pack("q", timeout * 1000)
            # Set the maximum amount for time the socket will wait for a response and/or attempt to send data.
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, timeout_pack)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, timeout_pack)
        # Ensure packets are sent immediately instead of waiting for larger batches to be created
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((HOST, self.tmi_port))"""
        self.registered = True
        print("Connected")

    # Launch program and return pids
    def launch_game(self):
        if config_copy.use_pynoko:
            self.pynoko_launch_game()
            return
        # See Dolphin Command Line Usage for more information (https://github.com/dolphin-emu/dolphin) (https://wiki.dolphin-emu.org/index.php?title=GameINI)
        self.dolphin_process_id = None
        dolphin_process_number = ""
        if self.process_number >= 1:
            dolphin_process_number = self.process_number + 1

        if config_copy.is_linux:
            self.game_spawning_lock.acquire()
            pid_before = self.get_dolphin_pids()
            launch_string = str(user_config.dolphin_base_path) + str(dolphin_process_number) + (f"{user_config.linux_launch_game_path}"
                        f" --video_backend='{config_copy.video_backend}'"
                        f" --config=Dolphin.Core.EmulationSpeed={config_copy.game_speed}"
                        f" --batch --script 'MKW_rl/MKW_interaction/game_instance_hook.py'"
                        f" --no-python-subinterpreters --exec='{config_copy.game_path}'")
            os.system(launch_string + " &") # continue execution after launching game
            while True:
                pid_after = self.get_dolphin_pids()
                dolphin_pid_candidates = set(pid_after) - set(pid_before)
                if len(dolphin_pid_candidates) > 0:
                    assert len(dolphin_pid_candidates) == 1
                    break
            self.dolphin_process_id = list(dolphin_pid_candidates)[0]
            self.game_spawning_lock.release()

        else:
            # print("Dolphin process number assigned as:", dolphin_process_number, "Derived from received process number of", self.process_number)
            launch_string = (
                'powershell -executionPolicy bypass -command "& {'
                f" $process = start-process -FilePath '{config_copy.dolphin_base_path}{dolphin_process_number}\\{config_copy.windows_dolphinexe_filename}'" # Launch .exe file
                " -PassThru -ArgumentList " # Assign arguments for .exe
                f'\'--video_backend="{config_copy.video_backend}" --config=Dolphin.Core.EmulationSpeed={config_copy.game_speed} --batch'
                f' --script "MKW_rl\\MKW_interaction\\game_instance_hook.py" --no-python-subinterpreters --exec="{config_copy.game_path}"\';'
                ' echo exit $process.id}"' # push process_id to stdout to read later
            )
            # --batch
            # print(launch_string)
            
            self.dolphin_process_id = int(subprocess.check_output(launch_string).decode().split("\r\n")[1]) # locate the pid from the program

        # print(f"Found Dolphin process id: {self.dolphin_process_id=}")
        with open("dolphin_ports/pid_" + str(self.dolphin_process_id), "w") as f:
            f.write(str(self.process_number))
        self.last_game_reboot = time.perf_counter() # set counter to know how old the process is
        self.latest_map_path_requested = -1
        self.msgtype_response_to_wakeup_TMI = None
        while not self.is_game_running(): # wait for the program to launch fully
            time.sleep(0)

    def is_game_running(self):
        if config_copy.use_pynoko:
            return self.pynoko_system is not None
        return (self.dolphin_process_id is not None) and (self.dolphin_process_id in (p.pid for p in psutil.process_iter()))

    def is_dolphin_process(self, process: psutil.Process) -> bool:
        try:
            # Dolphin.exe in windows and dolphin-emu on linux, hoo ray.
            return "dolphin" in process.name() or "Dolphin" in process.name()
        except psutil.NoSuchProcess:
            return False

    def get_dolphin_pids(self) -> List[int]:
        return [process.pid for process in psutil.process_iter() if self.is_dolphin_process(process)]

    def close_game(self):
        if config_copy.use_pynoko:
            self.pynoko_close_game()
            return
        self.timeout_has_been_set = False
        self.game_activated = False
        assert self.dolphin_process_id is not None
        if config_copy.is_linux:
            os.system("kill -9 " + str(self.dolphin_process_id))
        else:
            os.system(f"taskkill /PID {self.dolphin_process_id} /f")
        # Remove the temporary port file
        os.remove(config_copy.project_path / ("dolphin_ports/pid_" + str(self.dolphin_process_id)))
        self.sock.close()
        self.registered = False
        while self.is_game_running(): # wait for process to fully close
            time.sleep(0)

    def ensure_game_launched(self):
        if not self.is_game_running():
            print("Game not found. Starting Dolphin.")
            self.launch_game()

    """def select_map_savestate(self, savestate_path: str, zone_centers: npt.NDArray):
    # TODO: Convert to loading a specific savestate of chosen track one frame before countdown
    # This method also allows different vehicle combinations by proxy.
        self.latest_savestate_path_requested = savestate_path
        if user_config.is_linux:
            savestate_path = savestate_path.replace("\\", "/")
        else:
            savestate_path = savestate_path.replace("/", "\\")
    # TODO: Include a dictionary of track savestates somewhere
        savestate.load_from_file(savestate_path)
        self.UI_disabled = False
        (
            self.next_real_checkpoint_positions,
            self.max_allowable_distance_to_real_checkpoint,
        ) = map_loader.sync_virtual_and_real_checkpoints(zone_centers, savestate_path)"""

    def pynoko_launch_game(self):
        self.pynoko_system = pynoko.KHostSystem()
        # Activate a dummy time trial to call init()
        self.pynoko_system.configureTimeTrial(pynoko.Course.GCN_Mario_Circuit, pynoko.Character.Funky_Kong, pynoko.Vehicle.Flame_Runner, False)
        self.pynoko_system.init() # initialization so all rollout restarts can use .reset
        self.last_game_reboot = time.perf_counter()
        self.latest_map_path_requested = -1
        self.msgtype_response_to_wakeup_TMI = None
        print("pynoko initialized")

    def pynoko_close_game(self):
        self.pynoko_system = None
        self.timeout_has_been_set = False
        self.game_activated = False

    # TODO: move to MKW_data_translate
    def pynoko_read_game_data(self, frame_count):
        proxy = self.pynoko_system.kartObjectProxy()
        rm = self.pynoko_system.raceManager()
        player = rm.player()
        status = proxy.status()
        move = proxy.move()

        pos = proxy.pos().to_numpy().tolist()
        rot = proxy.full_rot().to_numpy().tolist()
        ext_vel = proxy.ext_vel().to_numpy().tolist()
        int_vel = proxy.int_vel().to_numpy().tolist()

        race_timer = player.raceTimer()
        race_time_s = (frame_count - 240 if frame_count > 240 else 0) / 60
        # print(race_time_s)

        mt_charge = move.mtCharge()
        # DriftState: NotDrifting / ChargingMt / ChargedMt / ChargedSmt
        drift_state_name = move.driftState().name

        inv = self.pynoko_system.itemDirector().itemInventory(0)
        race_completion = player.raceCompletion()
        self.pynoko_race_completion_max = max(self.pynoko_race_completion_max, race_completion)

        respawn = proxy.is_in_respawn() or status.onBit(pynoko.Status.InRespawn)
        in_trick = status.onBit(pynoko.Status.InATrick)
        trickable = status.onBit(pynoko.Status.Trickable)
        wheelie = proxy.is_bike() and status.onBit(pynoko.Status.Wheelie)

        return {
            "boost_data": {
                "mt_charge": mt_charge,
                "mt_charge_full": 1 if mt_charge >= 270 else 0,
                "smt_charge": 0,
                "smt_charge_full": 0,
                "ssmt_charge": 0,
                "ssmt_charge_full": 0,
                "mt_boost": 1 if status.onBit(pynoko.Status.Boost) else 0,
                "trick_boost": 1 if status.onBit(pynoko.Status.ZipperBoost) else 0,
                "shroom_boost": 1 if status.onBit(pynoko.Status.MushroomBoost) else 0,
            },
            "kart_data": {
                "character": 0,
                "vehicle": 0,
                "position": pos,
                "rotation": rot,
                "speed": proxy.speed(),
                "external_velocity": ext_vel,
                "internal_velocity": int_vel,
                "moving_road_velocity": [0.0, 0.0, 0.0],
                "moving_water_velocity": [0.0, 0.0, 0.0],
                "wheelie_cooldown": 1 if wheelie else 0,
                "trick_cooldown": 1 if in_trick else 0,
                "respawn_timer": 1 if respawn else 0,
                "time_in_respawn": 1 if respawn else 0,
            },
            "race_data": {
                "lap_completion": race_completion % 1.0,
                "race_completion": race_completion,
                "race_completion_max": self.pynoko_race_completion_max,
                "checkpoint_id": player.checkpointId(),
                "current_key_checkpoint": 0,
                "max_key_checkpoint": 0,
                "driving_direction": 0,
                "item_count": inv.currentCount(),
                "race_time": race_time_s,
                "state": 2,
            },
            "start_boost_charge": 0.0,
            "start_boost_full": 0,
            "trickable_timer": 1 if trickable else 0,
            "surface_properties": [
                1 if status.onBit(pynoko.Status.WallCollision) else 0,
                0,  # is_solid_oob (not bound)
                1 if status.onBit(pynoko.Status.RampBoost) else 0,
                1 if status.onBit(pynoko.Status.CollidingOffroad) else 0,
                0,  # is_boost_panel_or_ramp (not bound)
                1 if status.onBit(pynoko.Status.Trickable) else 0,
            ],
            "airtime": 0 if status.onBit(pynoko.Status.TouchingGround) else 1,
        }

    def pynoko_translate_action(self, computed_action: dict):
        """Convert GCInputs dict to pynoko setInput arguments."""
        btn_list = []
        if computed_action.get("A", 0):
            btn_list.append(pynoko.KPAD_BUTTON_A)
        if computed_action.get("TriggerRight", 0):
            # TriggerRight (GC R-trigger) = drift -> maps to KPAD_BUTTON_B (Wiimote B = drift)
            btn_list.append(pynoko.KPAD_BUTTON_B)
        if computed_action.get("TriggerLeft", 0):
            btn_list.append(pynoko.KPAD_BUTTON_ITEM)
        buttons = pynoko.buttonInput(btn_list)
        stickX_raw = int(round(7 + computed_action.get("StickX", 0) * 7))
        stickY_raw = int(round(7 + computed_action.get("StickY", 0) * 7))
        trick = pynoko.Trick.Up if computed_action.get("Up", 0) else pynoko.Trick.NoTrick
        return buttons, stickX_raw, stickY_raw, trick
    
    def pynoko_skip_intro(self, inputs):
        while self.pynoko_system.raceManager().stage() == pynoko.RaceManager.Stage.Intro:
            self.pynoko_system.setInput(inputs[0], inputs[1], inputs[2], inputs[3])
            self.pynoko_system.calc()

    def rollout(self, exploration_policy: Callable, savestate_path, zone_centers: npt.NDArray, update_network: Callable, last_loop_finished: bool):
        """
        exploration_policy: Function that returns ratio of exploration vs exploitation runs
        savestate_path: file path to current track to run
        update_network: function to send network information to update itself with
        """
        (
            zone_transitions,
            distance_between_zone_transitions,
            distance_from_start_track_to_prev_zone_transition,
            normalized_vector_along_track_axis,
        ) = map_loader.precalculate_virtual_checkpoints_information(zone_centers)

        self.ensure_game_launched()
        if time.perf_counter() - self.last_game_reboot > config_copy.game_reboot_interval: # stale instance of game
            self.close_game()
            self.sock = None
            self.launch_game()

        end_race_stats = {
            "cp_time_ms": [0],
        }

        instrumentation__answer_normal_step = 0
        instrumentation__answer_action_step = 0
        instrumentation__between_run_steps = 0
        instrumentation__grab_frame = 0
        instrumentation__request_inputs_and_speed = 0
        instrumentation__exploration_policy = 0
        instrumentation__convert_frame = 0
        instrumentation__grab_floats = 0

        rollout_results = {
            "current_zone_idx": [], # Current zone we are in based off the map.npy file
            "frames": [], # frame data
            "input_w": [], # Whether or not w is pressed
            "actions": [], # list of actions
            "action_was_greedy": [], # whether action is greedy as defined by the exploration policy
            "q_values": [],
            "race_completion": [],
            "state_float": [], # Game_Data object
            "furthest_zone_idx": 0, # based off map.npy file
        }

        if ((self.sock is None) or (not self.registered)) and not config_copy.use_pynoko: # Game was not connected to the program
            assert self.msgtype_response_to_wakeup_TMI is None
            print("Initialize connection to Dolphin from game_manager using port number", (self.tmi_port))
            # self.iface = TMInterface(self.tmi_port) # reset the interface

            connection_attempts_start_time = time.perf_counter()
            last_connection_error_message_time = time.perf_counter()
            while True:
                try:
                    self.register(config_copy.tmi_protection_timeout_s) # connect to the interface
                    break
                except ConnectionRefusedError as e:
                    current_time = time.perf_counter()
                    if current_time - last_connection_error_message_time > 1:
                        print(f"Connection to Dolphin unsuccessful for {current_time - connection_attempts_start_time:.1f}s")
                        last_connection_error_message_time = current_time
        """else:
            assert self.msgtype_response_to_wakeup_TMI is not None or self.last_rollout_crashed # Game is running and connected

            self.request_speed(self.running_speed) # FPS?
            if self.msgtype_response_to_wakeup_TMI is not None:
                self.iface._respond_to_call(self.msgtype_response_to_wakeup_TMI)
                self.msgtype_response_to_wakeup_TMI = None"""

        self.last_rollout_crashed = False

        frames_processed = 0 # track how long this rollout has been going
        current_zone_idx = 0

        # Insert values for the start of a race
        computed_action = {}
        if config_copy.use_pynoko:
            inputs = self.pynoko_translate_action(computed_action)
        this_rollout_is_finished = False
        n_th_action_we_compute = 0

        pc = time.perf_counter_ns() # performance counter
        pc5 = 0
        floats = None

        # distance_since_track_begin = 0.99 # Beginning lap completion percentage is usually about 0.999, depending on the track
        last_progress_improvement_f = 0
        game_data = None
        
        # Load the savestate if we have not done so (we are on the wrong map)
        # print("loading savestate")
        if (self.latest_map_path_requested != savestate_path or last_loop_finished) or not config_copy.use_race_restart:
            # We have to load the savestate we want
            # print("Loading savestate")
            if config_copy.use_pynoko:
                # TODO: rework savestatepath to give vehicle, character, and track combo instead of str
                self.pynoko_system.configureTimeTrial(savestate_path[0], savestate_path[1], savestate_path[2], False)
                self.pynoko_system.reset()
                self.pynoko_skip_intro(inputs)
                self.latest_map_path_requested = savestate_path
            else:
                self.sock.sendall(pickle.dumps([False, False, computed_action, savestate_path]))
                self.sock.recv(128) # wait for dolphin to load the state before requesting actions
                self.latest_map_path_requested = savestate_path # this seems backwards... TODO
        else: # map hasn't changed
            # Send signal to restart race manually instead of reloading savestate to save overhead
            # Note that this doesn't actually save any time as loading a savestate is generally just as fast
            # print("Restarting manually")
            if config_copy.use_pynoko:
                # TODO: rework savestatepath to give vehicle, character, and track combo instead of str
                self.pynoko_system.configureTimeTrial(savestate_path[0], savestate_path[1], savestate_path[2], False)
                self.pynoko_system.reset()
                self.pynoko_skip_intro(inputs)
            else:
                self.sock.sendall(pickle.dumps([False, False, computed_action, config_copy.restart_race_command]))
                self.sock.recv(128) # wait for dolphin to load the state before requesting actions

        self.pynoko_race_completion_max = 0

        disable_rewards = False
        while not this_rollout_is_finished:
            """
            This loop performs these essential functions:
            1. Load race savestate (enforce mini-races and race finishes)
            2. Read game state (floats and frame)
            3. Send data to exploration policy
            4. Send inputs received from exploration policy to the game
            5. update the network

            The flow of the loop consists of these steps:
            1. Ensure correct savestate is loaded
            2. Skip frame if needed
            3. Send current requested action
            4. Receive new game state after action
            5. Disable rewards if currently in a respawn
            6. Update maximum zone reached
            7. Convert game data to 1d list
            8. Send frame and game data to the exploration_policy to get next action
            9. Set new requested action
            10. Update the network
            11. Save rollout results, end race results, or both depending on finish state.
            """

            if self.latest_map_path_requested != savestate_path:
                # We have to load the savestate we want
                print("loading savestate")
                if config_copy.use_pynoko:
                    self.pynoko_system.configureTimeTrial(savestate_path[0], savestate_path[1], savestate_path[2], False)
                    self.pynoko_system.reset()
                    self.pynoko_skip_intro(inputs)
                    self.latest_map_path_requested = savestate_path
                    self.pynoko_race_completion_max = 0
                    frames_processed = 0
                else:
                    self.sock.sendall(pickle.dumps([False, False, computed_action, savestate_path]))
                    self.sock.recv(128) # wait for dolphin to load the state before requesting actions
                    self.latest_map_path_requested = savestate_path # this seems backwards... TODO
                    frames_processed = 0
                    continue

            frames_processed += 1
            if (frames_processed % self.run_steps_per_action != 0):
                if not config_copy.use_pynoko:
                    self.sock.sendall(pickle.dumps([False, False, computed_action, ""]))
                    self.sock.recv(128)
                else:
                    self.pynoko_system.setInput(inputs[0], inputs[1], inputs[2], inputs[3])
                    self.pynoko_system.calc()
                continue

            pc2 = time.perf_counter_ns()
            instrumentation__between_run_steps += pc2 - pc

            pc3 = time.perf_counter_ns()
            # send requested action
            if config_copy.use_pynoko:
                self.pynoko_system.setInput(inputs[0], inputs[1], inputs[2], inputs[3])
                self.pynoko_system.calc()
                # frame_data = self.pynoko_system.getFrame()[:FRAME_HEIGHT, :FRAME_WIDTH, 2::-1]
                frame_data = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            else:
                self.sock.sendall(pickle.dumps([True, True, computed_action, ""]))
                
                raw_frame_data = bytearray()
                data_length = FRAME_WIDTH * FRAME_HEIGHT * 3
                while len(raw_frame_data) < data_length:
                    try:
                        packet = self.sock.recv(data_length - len(raw_frame_data))
                    except Exception as e:
                        print("Error receiving frame data:", e)
                    if not packet:
                        print("Error receiving frame data")
                        break
                    raw_frame_data.extend(packet)
                frame_data = np.frombuffer(raw_frame_data, dtype = np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
                self.sock.sendall("frame_read".encode()) # maintain baton-passing
            pc4 = time.perf_counter_ns()
            instrumentation__grab_frame += pc4 - pc3
            # https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
            resized_frame = frame_data[::4,::4]
            """if frame_counter % 240 == 0:
                cv2.imshow("Greyscale", cv2.cvtColor(resized_frame, cv2.COLOR_BGRA2GRAY))
                cv2.waitKey(0)""" # Image is collected properly, save to file for display.
            resized_frame = np.expand_dims(cv2.cvtColor(resized_frame, cv2.COLOR_BGRA2GRAY), 0) # took me like 80 minutes to get to the solution that was already present in the original code
            # frame is a numpy array of shape (1, H, W) and dtype np.uint8
            pc5 = time.perf_counter_ns()
            instrumentation__convert_frame += pc5 - pc4
            if config_copy.use_pynoko:
                game_data = self.pynoko_read_game_data(frames_processed)
            else:
                game_data = pickle.loads(self.sock.recv(65535))
            if game_data["race_data"]["state"] == 0:
                # Race has not started, so skip frames until we enter countdown
                print("ERROR: Attempted to process intro camera state during rollout")
                continue
            # print("Game manager rollout() :: race time is", game_data["race_data"]["race_time"])

            # game_data["race_data"]["item_count"] = manual_item_count
            # print("Game data converted to:", network_inputs.get_flattened_game_data())
            race_time = max([game_data["race_data"]["race_time"], 1e-12]) # Epsilon trick to avoid division by zero

            if game_data["kart_data"]["respawn_timer"] > 0 or game_data["kart_data"]["time_in_respawn"] > 0:
                # Do not update current zone once a respawn occurs (disabling progress reward)
                disable_rewards = True
            elif not disable_rewards:
                current_zone_idx = update_current_zone_idx(
                    current_zone_idx,
                    zone_centers,
                    game_data["kart_data"]["position"],
                    config_copy.max_allowable_distance_to_virtual_checkpoint,
                )

            if current_zone_idx > rollout_results["furthest_zone_idx"]:
                last_progress_improvement_f = frames_processed
                rollout_results["furthest_zone_idx"] = current_zone_idx

            """distance_since_track_begin = game_data["race_data"]["race_completion"]
            if distance_since_track_begin > last_progress_improvement + config_copy.required_progress_per_cutoff_rollout: # Force the ai to improve by non-micro steps
                # print("Game manager rollout race_completion_max value updated:", game_data["race_data"]["race_completion"], "Now updating frame:", frames_processed)
                last_progress_improvement = distance_since_track_begin
                last_progress_improvement_f = frames_processed"""

            pc6 = time.perf_counter_ns()
            instrumentation__grab_floats += pc6 - pc5
            # TODO: Check relative zone center positions to ensure reasonable data is being sent
            # TODO: Convert zone center positions to be relative to the vehicle's rotation
            state_zone_center_coordinates_in_car_reference_system = zone_centers[
                current_zone_idx : current_zone_idx
                + config_copy.one_every_n_zone_centers_in_inputs
                * config_copy.n_zone_centers_in_inputs : config_copy.one_every_n_zone_centers_in_inputs,
                :,
            ] - game_data["kart_data"]["position"]
            game_data["relative_zone_centers"] = state_zone_center_coordinates_in_car_reference_system.ravel().tolist()

            floats = MKW_data_translate.get_1d_state_floats(game_data, rollout_results["actions"])
                # print("Floats generated:", len(floats))
            pc7 = time.perf_counter_ns()
            instrumentation__answer_action_step += pc7 - pc6

            # print("game_manager rollout(): Shape of img:", rollout_results["frames"][-1].shape, ":: floats:", floats)
            (
                action_idx,
                action_was_greedy,
                q_value,
                q_values,
            ) = exploration_policy(resized_frame, floats)
            pc8 = time.perf_counter_ns()
            instrumentation__exploration_policy += pc8 - pc7

            computed_action = config_copy.inputs[action_idx].copy()  # determine next input

            if computed_action["TriggerLeft"] != 0:
                # pressing item button
                # print("Items left:", manual_item_count, "While race is:", game_data["race_data"]["race_completion_max"])
                if game_data["race_data"]["item_count"] <= math.floor(-(game_data["race_data"]["race_completion_max"] - config_copy.Mushroom_point)):
                    computed_action["TriggerLeft"] = 0 # Disable item button if mushroom usage is bad
                    # print("Prevented item:", manual_item_count, "While max is:", math.floor(-(game_data["race_data"]["race_completion_max"] - 4)))
            if config_copy.use_pynoko:
                inputs = self.pynoko_translate_action(computed_action)

            # Save the estimated Q-value of the starting state (start of the track)
            if n_th_action_we_compute == 0:
                end_race_stats["value_starting_frame"] = q_value
                for i, val in enumerate(np.nditer(q_values)):
                    end_race_stats[f"q_value_{i}_starting_frame"] = val

            n_th_action_we_compute += 1
            if (n_th_action_we_compute % config_copy.update_inference_network_every_n_actions == 0):
                # print("Updating network")
                update_network()

            if not self.timeout_has_been_set:
                # reset the ai for doing bad things for too long?
                self.timeout_has_been_set = True

            race_time_for_ratio = race_time + 4 # include starting countdown time
            # print(game_data["start_boost_charge"], " And race time is", race_time)
            # Failed to finish race in time. Note that race_time is used to prevent resetting during the countdown
            if ((frames_processed > self.max_overall_duration_f or frames_processed > last_progress_improvement_f + self.max_minirace_duration_f) 
                and not this_rollout_is_finished and race_time > 2.5):
                # print("Failed at:", current_zone_idx, "Max completion:", rollout_results["furthest_zone_idx"], "Race completion:", rollout_results["race_completion"][-20:])
                
                end_race_stats["race_finished"] = False
                end_race_stats["race_time_for_ratio"] = race_time_for_ratio
                end_race_stats["race_time"] = config_copy.cutoff_rollout_if_race_not_finished_within_duration_f / config_copy.game_running_fps

                end_race_stats["instrumentation__answer_normal_step"] = (instrumentation__answer_normal_step / race_time_for_ratio * 50)
                end_race_stats["instrumentation__answer_action_step"] = (instrumentation__answer_action_step / race_time_for_ratio * 50)
                end_race_stats["instrumentation__between_run_steps"] = (instrumentation__between_run_steps / race_time_for_ratio * 50)
                end_race_stats["instrumentation__grab_frame"] = instrumentation__grab_frame / race_time_for_ratio * 50
                end_race_stats["instrumentation__convert_frame"] = (instrumentation__convert_frame / race_time_for_ratio * 50)
                end_race_stats["instrumentation__grab_floats"] = instrumentation__grab_floats / race_time_for_ratio * 50
                end_race_stats["instrumentation__exploration_policy"] = (instrumentation__exploration_policy / race_time_for_ratio * 50)
                end_race_stats["instrumentation__request_inputs_and_speed"] = (instrumentation__request_inputs_and_speed / race_time_for_ratio * 50)

                end_race_stats["tmi_protection_cutoff"] = False
            
                this_rollout_is_finished = True
            elif game_data["race_data"]["race_completion_max"] >= 4:
                print("Finished race in:", race_time)
                end_race_stats["race_finished"] = True
                end_race_stats["race_time_for_ratio"] = race_time_for_ratio
                end_race_stats["race_time"] = race_time

                end_race_stats["instrumentation__answer_normal_step"] = (instrumentation__answer_normal_step / race_time_for_ratio * 50)
                end_race_stats["instrumentation__answer_action_step"] = (instrumentation__answer_action_step / race_time_for_ratio * 50)
                end_race_stats["instrumentation__between_run_steps"] = (instrumentation__between_run_steps / race_time_for_ratio * 50)
                end_race_stats["instrumentation__grab_frame"] = instrumentation__grab_frame / race_time_for_ratio * 50
                end_race_stats["instrumentation__convert_frame"] = (instrumentation__convert_frame / race_time_for_ratio * 50)
                end_race_stats["instrumentation__grab_floats"] = instrumentation__grab_floats / race_time_for_ratio * 50
                end_race_stats["instrumentation__exploration_policy"] = (instrumentation__exploration_policy / race_time_for_ratio * 50)
                end_race_stats["instrumentation__request_inputs_and_speed"] = (instrumentation__request_inputs_and_speed / race_time_for_ratio * 50)

                end_race_stats["tmi_protection_cutoff"] = False

                rollout_results["race_time"] = race_time
                rollout_results["frames"].append(np.nan)
                rollout_results["input_w"].append(np.nan)
                rollout_results["actions"].append(np.nan)
                rollout_results["action_was_greedy"].append(np.nan)
                rollout_results["state_float"].append(np.nan)
                rollout_results["race_completion"].append(game_data["race_data"]["race_completion_max"])
                rollout_results["current_zone_idx"].append(len(zone_centers) - config_copy.n_zone_centers_extrapolate_after_end_of_map) # insert the last zone center for map completion

                this_rollout_is_finished = True
            else: # rollout continues
                rollout_results["current_zone_idx"].append(current_zone_idx)
                rollout_results["frames"].append(resized_frame)
                rollout_results["race_completion"].append(game_data["race_data"]["race_completion"])
                rollout_results["input_w"].append(config_copy.inputs[action_idx]["A"])
                rollout_results["actions"].append(action_idx)
                rollout_results["action_was_greedy"].append(action_was_greedy)
                rollout_results["q_values"].append(q_values)
                rollout_results["state_float"].append(game_data)

        return rollout_results, end_race_stats

