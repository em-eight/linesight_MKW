"""
This file contains classes that facilitate passing game data from the instance hook to the game manager.
As the game_instance_hook.py file runs within dolphin, it has access to the functions defined in the dolphin stubs.
Due to this, the game manager cannot import the classes which define datatypes for the game.
Thus, this file copies implementations of mat34 and vec3 without using dolphin stubs, and defines TypedDicti classes to assist data transfer.


"""

from dataclasses import dataclass
import math
from typing import TypedDict
import os, sys
sys.path.append(os.path.expanduser("~") + "\\AppData\\Local\\programs\\python\\python312\\lib\\site-packages")
import flatdict
from config_files import config_copy
import numpy as np
from enum import Enum

class RaceState(Enum):
    INTRO_CAMERA = 0  # Course preview
    COUNTDOWN = 1  # not including starting pan # yes the original comment is wrong.
    RACE = 2
    FINISHED_RACE = 3
    FINISHED_RACEv2 = 4

@dataclass
class mat34:
    e00: float = 0.0
    e01: float = 0.0
    e02: float = 0.0
    e03: float = 0.0
    e10: float = 0.0
    e11: float = 0.0
    e12: float = 0.0
    e13: float = 0.0
    e20: float = 0.0
    e21: float = 0.0
    e22: float = 0.0
    e23: float = 0.0

@dataclass
class vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def length_xz(self) -> float:
        return math.sqrt(self.x**2 + self.z**2)
    
@dataclass
class quatf:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0

class SurfaceProperties():
    def __init__(self, value):
        self.value = value

    WALL = 0x1
    SOLID_OOB = 0x2
    BOOST_RAMP = 0x10
    OFFROAD = 0x40
    BOOST_PANEL_OR_RAMP = 0x100
    TRICKABLE = 0x800

class Boosts(TypedDict, total=False):
    mt_charge: int
    smt_charge: int # smt exclusive to karts
    ssmt_charge: int
    mt_boost: int
    trick_boost: int
    shroom_boost: int

class Kart_Data(TypedDict, total=False):
    position: vec3
    rotation: quatf
    angle: float

    external_velocity: vec3
    internal_velocity: vec3
    moving_road_velocity: vec3
    moving_water_velocity: vec3

    wheelie_cooldown: int
    trick_cooldown: int

class Race_Data(TypedDict, total=False):
    lap_completion: float
    race_completion: float
    race_completion_max: float
    checkpoint_id: int
    current_key_checkpoint: int
    max_key_checkpoint: int

    respawn_point: int
    driving_direction: int # 0, 1, 2, or 3.
    item_count: int
    item_type: int



class Game_Data(TypedDict, total=False):
    boost_data: Boosts
    kart_data: Kart_Data
    race_data: Race_Data

    start_boost_charge: float
    trickable_timer: int
    surface_properties: list
    airtime: int

float_input_mean = [
    config_copy.temporal_mini_race_duration_actions / 2,
    0.9, # A
    0.1, # B
    0.1, # Dpad Up
    0, # StickX
    0, # StickY
    0.5, # TriggerLeft
    0.5, # TriggerRight
    0.9, # A
    0.1, # B
    0.1, # Dpad Up
    0, # StickX
    0, # StickY
    0.5, # TriggerLeft
    0.5, # TriggerRight
    0.9, # A
    0.1, # B
    0.1, # Dpad Up
    0, # StickX
    0, # StickY
    0.5, # TriggerLeft
    0.5, # TriggerRight
    0.9, # A
    0.1, # B
    0.1, # Dpad Up
    0, # StickX
    0, # StickY
    0.5, # TriggerLeft
    0.5, # TriggerRight
    0.9, # A
    0.1, # B
    0.1, # Dpad Up
    0, # StickX
    0, # StickY
    0.5, # TriggerLeft
    0.5, # TriggerRight
    # End ugly ugly input listing
    75, # mt_charge
    20, # smt_charge
    3, # ssmt_charge
    30, # mt_boost
    0, # trick_boost
    30, # shroom_boost
    # Begin kart_data
    23, # character
    18, # vehicle
    0.0, # position.x
    1000.0, # position.y
    0.0, # position.z
    0.0, # rotation.x
    0.0, # rotation.y
    0.0, # rotation.z
    0.0, # rotation.w
    60.0, # speed_norm
    20.0, # external_velocity.x
    10.0, # external_velocity.y
    20.0, # external_velocity.z
    60.0, # internal_velocity.x
    10.0, # internal_velocity.y
    60.0, # internal_velocity.z
    0.0, # moving_road_velocity.x
    0.0, # moving_road_velocity.y
    0.0, # moving_road_velocity.z
    0.0, # moving_water_velocity.x
    0.0, # moving_water_velocity.y
    0.0, # moving_water_velocity.z
    5, # wheelie_cooldown
    5, # trick_cooldown
    0, # respawn_timer
    0.5, # lap_completion
    2.0, # race_completion
    2.0, # race_completion_max
    10, # checkpoint_id
    3, # current_key_checkpoint
    3, # max_key_checkpoint
    2, # driving_direction
    1, # item_count
    60, # race_time
    2, # state
    0.8, # start_boost_charge -- average value over an entire rollout
    2, # trickable_timer
    # SURFACE PROPERTIES
    0, # is_wall
    0, # is_solid_oob
    0, # is_boost_ramp
    0, # is_offroad
    0, # is_boost_panel_or_ramp
    0, # surface_properties
    0, # is_trickable
    # ==================== BEGIN 40 CP =====================
    # -6.423e+03 1.434e+03 1.9811e+04
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    0,
    3000,
    0,
    # ====================  END  40 CP =====================
]

float_input_deviation = [
    config_copy.temporal_mini_race_duration_actions / 2,
    1, # A
    1, # B
    1, # Dpad Up
    2, # StickX
    2, # StickY
    1, # TriggerLeft
    1, # TriggerRight
    1, # A
    1, # B
    1, # Dpad Up
    2, # StickX
    2, # StickY
    1, # TriggerLeft
    1, # TriggerRight
    1, # A
    1, # B
    1, # Dpad Up
    2, # StickX
    2, # StickY
    1, # TriggerLeft
    1, # TriggerRight
    1, # A
    1, # B
    1, # Dpad Up
    2, # StickX
    2, # StickY
    1, # TriggerLeft
    1, # TriggerRight
    1, # A
    1, # B
    1, # Dpad Up
    2, # StickX
    2, # StickY
    1, # TriggerLeft
    1, # TriggerRight
    # End ugly ugly input list
    270, # mt_charge
    270, # smt_charge
    75, # ssmt_charge
    141, # mt_boost
    100, # trick_boost
    90, # shroom_boost
    # Begin kart_data
    47, # character
    35, # vehicle
    100_000, # position.x
    5_000, # position.y
    100_000, # position.z
    2.0, # rotation.x
    2.0, # rotation.y
    2.0, # rotation.z
    2.0, # rotation.w
    180.0, # speed_norm
    120.0, # external_velocity.x
    120.0, # external_velocity.y
    120.0, # external_velocity.z
    120.0, # internal_velocity.x
    120.0, # internal_velocity.y
    120.0, # internal_velocity.z
    # Toad's factory conveyers
    120.0, # moving_road_velocity.x
    120.0, # moving_road_velocity.y
    120.0, # moving_road_velocity.z
    # Koopa Cape's water stream
    120.0, # moving_water_velocity.x
    120.0, # moving_water_velocity.y
    120.0, # moving_water_velocity.z
    20, # wheelie_cooldown
    30, # trick_cooldown
    50, # respawn_timer
    1.0, # lap_completion
    4.0, # race_completion
    4.0, # race_completion_max
    100, # checkpoint_id
    15, # current_key_checkpoint
    15, # max_key_checkpoint
    4, # driving_direction
    3, # item_count
    360, # race_time
    3, # state
    1.0, # start_boost_charge
    23, # trickable_timer
    # SURFACE PROPERTIES
    1, # is_wall
    1, # is_solid_oob
    1, # is_boost_ramp
    1, # is_offroad
    1, # is_boost_panel_or_ramp
    1, # is_trickable
    240, # airtime -- note that this may be higher if you get more than 4 seconds of airtime but eh.
    # ==================== BEGIN 40 CP =====================
    # 4.0e+04  5.0e+03  6.0e+04
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    100_000,
    10_000,
    100_000,
    # ==================== END   40 CP =====================
]

class Network_Inputs():
    def __init__(self, game_data: Game_Data, previous_actions_idx):
        self.game_data = game_data
        self.__flat_game_data = None
        self.previous_actions_idx = previous_actions_idx

    def get_input_dimensions(self):
        previous_inputs_length = config_copy.n_prev_actions_in_inputs * 18
        game_data_length = len(self.get_flattened_game_data())
        
        input_dimensions = previous_inputs_length + game_data_length
        # Sanity check in case of bad programmers
        assert game_data_length == len(float_input_deviation)
        return input_dimensions
    
    def get_input_data(self):
        previous_actions = [
            # Get inputs from k for every k that we have not processed yet
            config_copy.inputs[self.previous_actions_idx[k] if k >= 0 else config_copy.action_forward_idx]
            for k in range(
                len(self.previous_actions_idx) - config_copy.n_prev_actions_in_inputs, len(self.previous_actions_idx)
            )
        ]
        unwrapped_actions = []
        for dictionary in previous_actions:
            for value in dictionary.values():
                unwrapped_actions.append(value)
        
        return unwrapped_actions

    def get_flattened_game_data(self):
        if not self.__flat_game_data:
            temp_game_data = self.game_data.copy()
            for key in temp_game_data["kart_data"].keys():
                value = temp_game_data["kart_data"][key]
                if type(value) == vec3:
                    temp_game_data["kart_data"][key] = [value.x, value.y, value.z]
                elif type(value) == quatf:
                    temp_game_data["kart_data"][key] = [value.x, value.y, value.z, value.w]
            self.__flat_game_data = flatdict.FlatterDict(temp_game_data).values()
        return self.__flat_game_data
    

def get_1d_state_floats(game_data, previous_actions_idx):
    network_inputs = Network_Inputs(game_data, previous_actions_idx)

    return np.hstack(
        (
            config_copy.temporal_mini_race_duration_actions / 2,
            np.array(
                network_inputs.get_input_data()
            ),
            np.array(
                network_inputs.get_flattened_game_data() # includes relative zone centers
            )
        ),
        dtype=np.float32
    )
