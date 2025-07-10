from http import client

from dolphin import event, gui, controller # type: ignore
# from game_data_interface import Game_Data_Interface
import time
import os
import sys

# SET THIS PATH TO WHERE THESE SCRIPTS ARE LOCATED
# Another path you need to set is located in the mkw_interface script
# Trying to import the project's config files will likely not work. It is HIGHLY recommended to use the full path.
# sys.path.append(os.path.expanduser("~") + "\\AppData\\Local\\programs\\python\\python312\\lib\\site-packages")
sys.path.append(os.path.expanduser("~") + "/Programs/dolphin/Build/Binaries")

# Where the map.npy file should be saved
map_save_folder = os.path.expanduser("~") + "/Programs/dolphin/Build/Binaries/maps"

from MKW_interface import Game_Data, MKW_Interface
import geometry
import numpy as np
from pathlib import Path
import math

import socket

HOST = "127.0.0.1"

# 2 - ((time_taken / lap_length) * 4) / 3

# 30 45 100

# 60 - 30 = 30 now 0 = 2, 30 = -2
2 / 30

expected_time_per_lap = 27 # in seconds
expected_f_per_lap = expected_time_per_lap * 60 # calculate frames
expected_progress_per_f_per_lap = 1 / expected_f_per_lap # Total race_completion we earned from a lap divided by expected time gives us the average progress per frame per lap

# 0.0005
# 0.0008 for 20 second time
# frame 1: progressed 0.001
# frame 60: progressed 0.060 - 6/100 in 1 second
# 16.666 seconds to complete the lap


# If it takes us 30 seconds to complete the lap, our reward should be 0 as that is the expected time.
# So, it would be useful to have a secondary target time for rewards = 2/3rds per lap, or 2 for a total race on the target time, and increasing for better times.
# zero_reward_ratio = race_completion_percentage per frame expected_frames_per_lap


constant_reward_per_f = -expected_progress_per_f_per_lap
# complete in 30s = 0 reward

reward_per_lap_completed = 2 # reward for completing the lap
reward_per_m_advanced_along_centerline = 4/3 # total reward per lap completed
final_speed_reward_as_if_duration_s = 0.0 # times speed (80) times reward_per_m (2000) = 
final_speed_reward_per_f_per_s = reward_per_m_advanced_along_centerline * final_speed_reward_as_if_duration_s
engineered_item_usage_reward = 0
engineered_button_A_pressed_reward = 0

# data_points = np.array(np.load("maps/rMC3.npy"), dtype=np.float32)

distance_between_checkpoints = 300
# Old values were 0.5 dbc and 90 rw. I have set the road_width to roughly match those values' ratios (90/0.5 = 180, 30000/300 = 100) taking 'closer to 24' into account (24/0.5 = 48)
road_width = 30000  ## a little bit of margin, could be closer to 24 probably ? Don't take risks there are curvy roads
max_allowable_distance_to_virtual_checkpoint = np.sqrt((distance_between_checkpoints / 2) ** 2 + (road_width / 2) ** 2)

class GameInstanceHook():
    def __init__(self, port=65432):
        self.desired_inputs = None
        self.last_desired_inputs = {}
        self.current_unprocessed_frame = None
        self.resized = None
        self.frame_counter = 0
        self.red = 0xffff0000
        self.listener = None
        self.conn = None
        # self.game_data_interface = Game_Data_Interface()
        self.port = port
        self.last_rewards = 0
        self.rewards = 0
        self.game_data_initiated = False
        self.game_data_interface = MKW_Interface()
        self.game_data: Game_Data = None
        self.last_race_completion = 0.995
        self.setting_time_period = False
        self.expected_time_per_lap = 29

        self.positions = []
        self.positions_saved = False
        self.current_zone_idx = 0
        self.vcp_reward = 0

    def update_rewards(self):
        self.last_rewards = self.rewards

        # reward for 3lap completion at target time = 2
        # const reward for total time = -4
        # reward for 3lap completion at target time = 6

        # reward for 3lap completion at target time * 2 = -2

        # const_reward: (-2/(seconds_per_lap*2*actions_per_second))/3 # at seconds_per_lap*2 / 3 seconds (double total 3lap time), the reward is -2. reward is -1 when on target
        # added_race_completion * actions_per_second * seconds_per_lap
        # (-2 + (added_race_completion * actions_per_second * seconds_per_lap)  * 4)) / (seconds_per_lap * actions_per_second) / 3

        # -4 / time_per_lap * lap_count * 15
        # added_race_completion * 2
        self.rewards += -2 / (7 * 15)
        total_expected = 3 / (self.expected_time_per_lap * 3 / 7)
        self.rewards += ((self.game_data["race_data"]["race_completion"] - self.last_race_completion) * (4 / total_expected))
        """self.rewards += (
            self.game_data["race_data"]["race_completion"] - self.last_race_completion # meters progressed (negative if backwards)
        ) * reward_per_m_advanced_along_centerline"""

        if final_speed_reward_per_f_per_s != 0:
            # car is driving forward at a decent speed
            self.rewards += final_speed_reward_per_f_per_s * (
                self.game_data["kart_data"]["speed"]
            )


        buttons = controller.get_gc_buttons(0)
        if engineered_button_A_pressed_reward != 0 and buttons["A"] == True:
            self.rewards += engineered_button_A_pressed_reward

        if engineered_item_usage_reward != 0 and buttons["TriggerLeft"] == True:
            self.rewards += engineered_item_usage_reward

        # self.current_zone_idx = geometry.update_current_zone_idx(self.current_zone_idx, data_points, self.game_data["kart_data"]["position"], max_allowable_distance_to_virtual_checkpoint)



        """# TODO: Recalculate this value to match -0.5, 0.5 over 7 seconds
        self.rewards += (2 / (7 * (15))) * np.interp(max(900,
                min(5000, np.linalg.norm(data_points[self.current_zone_idx] - self.game_data["kart_data"]["position"])),
            ),
            [900, 5000],
            [0.5, -1])
        
        self.vcp_reward += (2 / (7 * (15))) * np.interp(max(900,
                min(5000, np.linalg.norm(data_points[self.current_zone_idx] - self.game_data["kart_data"]["position"])),
            ),
            [900, 5000],
            [0.5, -1])"""
        
        
        if buttons["CStickY"] > 0.5:
            self.rewards = 0

        if buttons["CStickX"] < -0.5:
            if not self.setting_time_period:
                self.expected_time_per_lap -= 1
                self.setting_time_period = True
        elif buttons["CStickX"] > 0.5:
            if not self.setting_time_period:
                self.expected_time_per_lap += 1
                self.setting_time_period = True
        elif self.setting_time_period:
            self.setting_time_period = False
        

    def framedrawn_handler(self, width, height, data):
        # Wait for data necessary to determine what we want to do
        self.current_unprocessed_frame = (height, width, data)

        """# https://stackoverflow.com/questions/38412887/how-to-send-a-list-through-tcp-sockets-python
        socket_data = pickle.loads(self.conn.recv(4096))
        print("Received:", socket_data)

        frame_data_request = socket_data[0]
        game_data_request = socket_data[1]
        set_new_inputs = socket_data[2]

        if set_new_inputs and len(socket_data) >= 4:
            # TODO: Use input-list index to assign inputs in leiu of sending the full dicti
            self.desired_inputs = socket_data[3]
        elif set_new_inputs and not len(socket_data) >= 4:
            print("ERROR: New inputs set to be delivered but none were provided")"""
        
        if not self.game_data_initiated:
            self.game_data_interface.initialize_race_objects()
            self.game_data_initiated = True

        if self.frame_counter % 4 == 0:
            self.game_data = self.game_data_interface.get_game_data_object()
            if self.game_data["boost_data"]["shroom_boost"] > 110:
                pass
            # print("updated game data:", self.game_data["race_data"]["race_completion"], "last:", self.last_race_completion)
            self.update_rewards()
        # process and send the image data here, so we can set desired_inputs before we exit the function"""
            
    def frameadvance_handler(self):
        self.frame_counter += 1
        violated_item = False
        if (math.floor(-(self.game_data["race_data"]["race_completion_max"] - 4))) > self.game_data["race_data"]["item_count"]:
            violated_item = True
        # draw on screen
        gui.draw_text((10, 10), self.red, 
                    f"Position: {self.game_data["kart_data"]["position"]}, Current_zone: {self.current_zone_idx}\n"
                    + f"Rewards: {self.rewards}\nvcp_rewards: {self.vcp_reward}\n"
                    + f"Expected time: {self.expected_time_per_lap}\n"
                    + f"race completion: {self.game_data['race_data']['race_completion']}\n"
                    + f"State: {self.game_data["race_data"]["state"]}\n"
                    + f"Items: {self.game_data["race_data"]["item_count"]}")

        self.last_race_completion = float(self.game_data["race_data"]["race_completion"])
        if self.desired_inputs and self.desired_inputs != self.last_desired_inputs:
            controller.set_gc_buttons(0, self.desired_inputs)
            self.last_desired_inputs = self.desired_inputs

        if self.game_data["race_data"]["race_time"] > 0 and self.game_data["race_data"]["race_completion_max"] < 4:
            new_position = self.game_data["kart_data"]["position"]
            self.positions.append(np.array(new_position))

        if not self.positions_saved and self.game_data["race_data"]["race_completion_max"] >= 4: # completed one lap
            # save positions to file
            geometry.extract_cp_distance_interval(self.positions, 300, map_save_folder)
            self.positions_saved = True

    def register(self, port):
        print("Initialize connection to Dolphin ")
        # self.iface = TMInterface(self.tmi_port) # reset the interface

        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(HOST, self.port)
        print("Game hook socket binded on port", self.port)

        self.listener.listen(1)
        self.conn, addr = self.listener.accept()
        print("Connected. Address:", addr)


mymanager = GameInstanceHook()

# mymanager.register()

event.on_framedrawn(mymanager.framedrawn_handler)
event.on_frameadvance(mymanager.frameadvance_handler)