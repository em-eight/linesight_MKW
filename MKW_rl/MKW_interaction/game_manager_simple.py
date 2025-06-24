from multiprocessing import process
from MKW_rl.MKW_interaction import MKW_data_translate
from config_files import config_copy, user_config

import math
import os
from multiprocessing.connection import Client
import subprocess
import time
from typing import Callable, Dict, List
import flatdict

from MKW_rl.MKW_interaction.MKW_data_translate import *
from MKW_rl import map_loader

import cv2
import numba
import numpy as np
import numpy.typing as npt
import psutil

# import warnings ignored for cross-platform neatness
if config_copy.is_linux:
    import Xdo # type: ignore
else:
    import win32.lib.win32con as win32con # type: ignore
    import win32com.client # type: ignore
    import win32gui # type: ignore
    import win32process # type: ignore

HOST = "127.0.0.1"
FRAME_WIDTH = 611
FRAME_HEIGHT = 456

# Assuming that this function will work unmodified despite changes to the magnitude of zone spacings
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
        self.dolphin_window_id = None
        self.start_states = {} # oh hey I might want to use this for starting later on in the track
        self.game_spawning_lock = game_spawning_lock
        self.game_activated = False
        self.process_number = process_number
    
    def get_window_id(self):
        assert self.dolphin_process_id is not None
        if config_copy.is_linux:
            self.dolphin_window_id = None
            while self.dolphin_window_id == None:
                window_search_depth = 1
                while True:
                    c1 = set(Xdo().search_windows(winname=b"Dolphin", max_depth=window_search_depth + 1))
                    c2 = set(Xdo().search_windows(winname=b"Dolphin", max_depth=window_search_depth))
                    c1 = {w_id for w_id in c1 if Xdo().get_pid_window(w_id) == self.dolphin_process_id}
                    c2 = {w_id for w_id in c2 if Xdo().get_pid_window(w_id) == self.dolphin_process_id}
                    c1_diff_c2 = c1.difference(c2)
                    if len(c1_diff_c2) == 1:
                        self.dolphin_window_id = c1_diff_c2.pop()
                        break
                    elif (
                        len(c1_diff_c2) == 0 and len(c1) > 0
                    ) or window_search_depth >= 10:  # 10 is an arbitrary cutoff in this search we do not fully understand
                        print(
                            "Warning: Worker could not find the window of the game it just launched, stopped at window_search_depth",
                            window_search_depth,
                        )
                        break
                    window_search_depth += 1
        else:
            def get_hwnds_for_pid(pid):
                def callback(hwnd, hwnds):
                    _, found_pid = win32process.GetWindowThreadProcessId(hwnd)

                    if found_pid == pid:
                        hwnds.append(hwnd)
                    return True

                hwnds = []
                win32gui.EnumWindows(callback, hwnds)
                return hwnds

            while True:
                for hwnd in get_hwnds_for_pid(self.dolphin_process_id):
                    if "Dolphin" in win32gui.GetWindowText(hwnd):
                        self.dolphin_window_id = hwnd
                        return
                    # else:
                    #     raise Exception("Could not find TmForever window id.")

    def register(self, timeout=None):
        # https://stackoverflow.com/questions/6920858/interprocess-communication-in-python
        print(self.tmi_port + (self.dolphin_process_id % (65535 - self.tmi_port)))
        self.sock = Client((HOST, (self.tmi_port + (self.dolphin_process_id % (65535 - self.tmi_port))))) # Client((HOST, self.tmi_port))
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
        # See Dolphin Command Line Usage for more information (https://github.com/dolphin-emu/dolphin) (https://wiki.dolphin-emu.org/index.php?title=GameINI)
        self.dolphin_process_id = None
        dolphin_process_number = ""
        if self.process_number >= 1:
            dolphin_process_number = self.process_number + 1

        if config_copy.is_linux:
            self.game_spawning_lock.acquire()
            pid_before = self.get_dolphin_pids()
            # dolphin.exe path --no python subinterpreters
            os.system(str(user_config.dolphin_base_path) + str(user_config.linux_launch_game_path) + (f"{dolphin_process_number} --video_backend='{config_copy.video_backend}'"
                       f"--config=Dolphin.Core.EmulationSpeed={config_copy.game_speed}"
                       "--batch --script MKW_rl/MKW_interaction/game_instance_hook.py"
                       f"--no-python-subinterpreters --exec='{config_copy.game_path}' + str(self.tmi_port))"))
            while True:
                pid_after = self.get_dolphin_pids()
                dolphin_pid_candidates = set(pid_after) - set(pid_before)
                if len(dolphin_pid_candidates) > 0:
                    assert len(dolphin_pid_candidates) == 1
                    break
            self.dolphin_process_id = list(dolphin_pid_candidates)[0]

        else:
            # print("Dolphin process number assigned as:", dolphin_process_number, "Derived from received process number of", self.process_number)
            launch_string = (
                'powershell -executionPolicy bypass -command "& {'
                f" $process = start-process -FilePath '{config_copy.dolphin_base_path}{dolphin_process_number}\\{config_copy.windows_dolphinexe_filename}'" # Launch .exe file
                " -PassThru -ArgumentList " # Assign arguments for .exe
                f'\'--video_backend="{config_copy.video_backend}" --config=Dolphin.Core.EmulationSpeed={config_copy.game_speed} --batch --script MKW_rl\\MKW_interaction\\game_instance_hook.py --no-python-subinterpreters --exec="{config_copy.game_path}"\';'
                ' echo exit $process.id}"' # push process_id to stdout to read later
            )
            # --batch
            # print(launch_string)
            
            self.dolphin_process_id = int(subprocess.check_output(launch_string).decode().split("\r\n")[1]) # locate the pid from the program
            # We do not need the parent of returned process id for this fork
            """while self.dolphin_process_id is None:
                dolphin_processes = list(
                    filter(
                        lambda s: s.startswith("Dolphin"), # confirm this is a Dolphin process
                        subprocess.check_output("wmic process get Caption,ParentProcessId,ProcessId").decode().split("\r\n"),
                    )
                ) # create a list of Dolphin processes by filtering out unmatching ones and checking their output
                for process in dolphin_processes:
                    name, parent_id, process_id = process.split() # extract information from process
                    parent_id = int(parent_id)
                    process_id = int(process_id)
                    if parent_id == dolphin_process_id: # confirm we have our Dolphin process and assign it
                        self.dolphin_process_id = process_id
                        break"""

        print(f"Found Dolphin process id: {self.dolphin_process_id=}")
        self.last_game_reboot = time.perf_counter() # set counter to know how old the process is
        self.latest_map_path_requested = -1
        self.msgtype_response_to_wakeup_TMI = None
        while not self.is_game_running(): # wait for the program to launch fully
            time.sleep(0)

        self.get_window_id() # locate window ID for the process

    def is_game_running(self):
        return (self.dolphin_process_id is not None) and (self.dolphin_process_id in (p.pid for p in psutil.process_iter()))

    def is_dolphin_process(self, process: psutil.Process) -> bool:
        try:
            return "Dolphin" in process.name()
        except psutil.NoSuchProcess:
            return False

    def get_dolphin_pids(self) -> List[int]:
        return [process.pid for process in psutil.process_iter() if self.is_dolphin_process(process)]

    def close_game(self):
        self.timeout_has_been_set = False
        self.game_activated = False
        assert self.dolphin_process_id is not None
        if config_copy.is_linux:
            os.system("kill -9 " + str(self.dolphin_process_id))
        else:
            os.system(f"taskkill /PID {self.dolphin_process_id} /f")
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

    def rollout(self, exploration_policy: Callable, savestate_path: str, zone_centers: npt.NDArray, update_network: Callable, last_loop_finished: bool):
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

        if (self.sock is None) or (not self.registered): # Game was not connected to the program
            assert self.msgtype_response_to_wakeup_TMI is None
            print("Initialize connection to Dolphin from game_manager")
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
        computed_action = None
        give_up_signal_has_been_sent = False
        this_rollout_has_seen_t_negative = False
        this_rollout_is_finished = False
        n_th_action_we_compute = 0
        compute_action_asap = True
        compute_action_asap_floats = True
        frame_expected = False
        map_change_requested_time = math.inf

        last_known_simulation_state = None
        pc = time.perf_counter_ns() # performance counter
        pc5 = 0
        floats = None

        # distance_since_track_begin = 0.99 # Beginning lap completion percentage is usually about 0.999, depending on the track
        last_progress_improvement = 0
        last_progress_improvement_f = 0

        sim_state_car_gear_and_wheels = None

        game_data = None
        
        # Load the savestate if we have not done so (we are on the wrong map)
        # print("loading savestate")
        if (self.latest_map_path_requested != savestate_path or last_loop_finished) or not config_copy.use_race_restart:
            # We have to load the savestate we want
            # print("Loading savestate")
            self.sock.send([False, False, computed_action, savestate_path])
            self.latest_map_path_requested = savestate_path # this seems backwards... TODO
        else:
            # Send signal to restart race manually instead of reloading savestate to save overhead
            # Note that this may run into some serious issues regarding load times and reward functions
            # These issues also will be hard to debug... oh joy. IDK if this is worth it xd
            # print("Restarting manually")
            self.sock.send([False, False, computed_action, config_copy.restart_race_command])
        
        
        # self.sock.send([False, False, computed_action, savestate_path])
        # self.latest_map_path_requested = savestate_path # this seems backwards... TODO

        manual_item_count = 3
        while not this_rollout_is_finished:
            """
            This loop needs to perform these essential functions:
            1. Load race savestate (enforce mini-races and race finishes)
            2. Read game state (floats and frame)
            3. Send data to exploration policy
            4. Send inputs received from exploration policy to the game
            5. update the network
            """

            if self.latest_map_path_requested != savestate_path:
                # We have to load the savestate we want
                # print("loading savestate")
                self.sock.send([False, False, computed_action, savestate_path])
                self.latest_map_path_requested = savestate_path # this seems backwards... TODO
                continue
            pc2 = time.perf_counter_ns()
            instrumentation__between_run_steps += pc2 - pc
            if (frames_processed % self.run_steps_per_action != 0):
                self.sock.send([False, False, computed_action, None])
                compute_action_asap_floats = True
                frames_processed += 1
                continue
            pc3 = time.perf_counter_ns()
            
            self.sock.send([True, True, computed_action, None])
            # The following line brought to you by literal hours of trying to figure things out only to realize I just needed two functions that I could've just copied from the original code
            frame_data = np.frombuffer(self.sock.recv_bytes(FRAME_WIDTH * FRAME_HEIGHT * 3), dtype = np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
            pc4 = time.perf_counter_ns()
            instrumentation__grab_frame += pc4 - pc3
            frames_processed += 1
            # https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
            resized_frame = frame_data[::4,::4]
            """if frame_counter % 240 == 0:
                cv2.imshow("Greyscale", cv2.cvtColor(resized_frame, cv2.COLOR_BGRA2GRAY))
                cv2.waitKey(0)""" # Image is collected properly, save to file for display.
            resized_frame = np.expand_dims(cv2.cvtColor(resized_frame, cv2.COLOR_BGRA2GRAY), 0) # took me like 80 minutes to get to the solution that was already present in the original code
            # frame is a numpy array of shape (1, H, W) and dtype np.uint8
            pc5 = time.perf_counter_ns()
            instrumentation__convert_frame += pc5 - pc4
            game_data = self.sock.recv()
            if game_data["race_data"]["state"] == 0:
                # Race has not started, so skip frames until we enter countdown
                print("ERROR: Attempted to process intro camera state during rollout")
                continue
            # print("Game manager rollout() :: race time is", game_data["race_data"]["race_time"])
            if game_data["boost_data"]["shroom_boost"] > 86:
                manual_item_count -= 1

            game_data["race_data"]["item_count"] = manual_item_count
            # print("Game data converted to:", network_inputs.get_flattened_game_data())
            race_time = max([game_data["race_data"]["race_time"], 1e-12]) # Epsilon trick to avoid division by zero

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
            state_zone_center_coordinates_in_car_reference_system = zone_centers[
                current_zone_idx : current_zone_idx
                + config_copy.one_every_n_zone_centers_in_inputs
                * config_copy.n_zone_centers_in_inputs : config_copy.one_every_n_zone_centers_in_inputs,
                :,
            ] - game_data["kart_data"]["position"]
            game_data["relative_zone_centers"] = state_zone_center_coordinates_in_car_reference_system.ravel().tolist()

            if compute_action_asap_floats:
                compute_action_asap_floats = False
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
                if manual_item_count <= math.floor(-(game_data["race_data"]["race_completion_max"] - config_copy.Mushroom_point)):
                    computed_action["TriggerLeft"] = 0 # Disable item button if mushroom usage is bad
                    # print("Prevented item:", manual_item_count, "While max is:", math.floor(-(game_data["race_data"]["race_completion_max"] - 4)))

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
            
            if frames_processed == 0 and (savestate_path != self.latest_map_path_requested):
                map_change_requested_time = frames_processed
                give_up_signal_has_been_sent = True

            race_time_for_ratio = race_time + 4 # include starting countdown time
            # print(game_data["start_boost_charge"], " And race time is", race_time)
            # Failed to finish race in time. Note that race_time is used to prevent resetting during the countdown
            if ((frames_processed > self.max_overall_duration_f or frames_processed > last_progress_improvement_f + self.max_minirace_duration_f) 
                and not this_rollout_is_finished and race_time > 2.5):
                #print("Failed at:", current_zone_idx, "Max completion:", rollout_results["furthest_zone_idx"])
                
                end_race_stats["race_finished"] = False
                end_race_stats["race_time_for_ratio"] = race_time_for_ratio
                end_race_stats["race_time"] = config_copy.cutoff_rollout_if_race_not_finished_within_duration_f

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

