"""
This file's main entry point is the function fill_buffer_from_rollout_with_n_steps_rule().
Its main inputs are a rollout_results object (obtained from a GameInstanceManager object), and a buffer to be filled.
It reassembles the rollout_results object into transitions, as defined in /trackmania_rl/experience_replay/experience_replay_interface.py
"""

import math
import random

import numpy as np
from numba import jit
from torchrl.data import ReplayBuffer

from config_files import config_copy
from MKW_rl.experience_replay.experience_replay_interface import Experience
from MKW_rl.MKW_interaction.MKW_data_translate import get_1d_state_floats


# @jit(nopython=True)
def get_potential(state_float):

    # distance_traveled_potential = 0
    # distance_traveled_potential += config_copy.constant_reward_per_f
    """distance_traveled_potential = (
        state_float["race_data"]["race_completion"] - (state_float["race_data"]["race_completion_max"]) # meters of backwards-progression
    ) * config_copy.reward_per_m_advanced_along_centerline"""
    # distance_traveled_potential += (state_float["kart_data"]["speed"] - 70) / config_copy.average_lap_increment_per_action

    # https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
    vector_vcp_to_vcp_further_ahead = state_float["relative_zone_centers"][1] - state_float["relative_zone_centers"][0]
    vector_vcp_to_vcp_further_ahead_normalized = vector_vcp_to_vcp_further_ahead / np.linalg.norm(vector_vcp_to_vcp_further_ahead)
    # current vcp is target ahead of us, previous vcp is what we've just passed
    # Punish ai for being farther away from current vcp by ratio of how far we are from previous vcp
    # -0.1 * max(2, min(25, previous_vector_vc_dist)) + (0 * n)
    return (-config_copy.constant_reward_per_action / 2) * np.interp(max(config_copy.engineered_reward_min_dist_to_cur_vcp,
            min(config_copy.engineered_reward_max_dist_to_cur_vcp, np.linalg.norm(state_float["relative_zone_centers"][0])),
        ),
        [config_copy.engineered_reward_min_dist_to_cur_vcp, config_copy.engineered_reward_max_dist_to_cur_vcp],
        [0.5, -1])
    return (
        config_copy.shaped_reward_dist_to_cur_vcp
        * max(
            config_copy.shaped_reward_min_dist_to_cur_vcp,
            min(config_copy.shaped_reward_max_dist_to_cur_vcp, np.linalg.norm(state_float["relative_zone_centers"][0])), # max of 25
        )
    ) + (config_copy.shaped_reward_point_to_vcp_ahead * (vector_vcp_to_vcp_further_ahead_normalized[2] - 1))


def fill_buffer_from_rollout_with_n_steps_rule(
    buffer: ReplayBuffer,
    buffer_test: ReplayBuffer,
    rollout_results: dict,
    n_steps_max: int,
    gamma: float,
    discard_non_greedy_actions_in_nsteps: bool,
    engineered_item_usage_reward=0.0,
    engineered_button_A_pressed_reward=2,
    engineered_supergrinding_reward=0.0,
    engineered_close_to_vcp_reward=0,
    engineered_start_boost_reward=0,
):
    assert len(rollout_results["frames"]) == len(rollout_results["current_zone_idx"])
    n_frames = len(rollout_results["frames"])

    number_memories_added_train = 0
    number_memories_added_test = 0
    Experiences_For_Buffer = []
    Experiences_For_Buffer_Test = []
    list_to_fill = Experiences_For_Buffer_Test if random.random() < config_copy.buffer_test_ratio else Experiences_For_Buffer

    gammas = (gamma ** np.linspace(1, n_steps_max, n_steps_max)).astype(
        np.float32
    )  # Discount factor that will be placed in front of next_step in Bellman equation, depending on n_steps chosen

    reward_into = np.zeros(n_frames)
    # reward_into_constant = np.zeros(n_frames)
    # reward_into_progress = np.zeros(n_frames)
    # reward_into_ev = np.zeros(n_frames)
    for i in range(1, n_frames): # run for each frame of the rollout
        """reward_into[i] += config_copy.constant_reward_per_ms * (
            config_copy.f_per_action
            if (i < n_frames - 1 or ("race_time" not in rollout_results)) # We haven't generated any frames, or the race has not started
            else rollout_results["race_time"] - (n_frames - 2) * config_copy.f_per_action
        )"""
        if type(rollout_results["state_float"][i]) != float and rollout_results["state_float"][i]["race_data"]["state"] == 2: # Apply these rewards during the race (Not race finished, not during countdown timer)
            reward_into[i] += config_copy.constant_reward_per_action
            # reward_into_constant[i] += config_copy.constant_reward_per_action

            if rollout_results["state_float"][i]["race_data"]["item_count"] < rollout_results["state_float"][i - 1]["race_data"]["item_count"]:
                # used item punish
                reward_into[i] += config_copy.constant_reward_per_action * engineered_item_usage_reward

            """ # LUIGI CIRCUIT FORCE SHORTCUT
            if rollout_results["state_float"][i]["kart_data"]["position"][2] > config_copy.LC_punish_line:
                reward_into[i] += config_copy.constant_reward_per_action * config_copy.LC_punish_rate # TODO: Set this value in the map cycle? """

            """ if i < n_frames - 1:
                if engineered_close_to_vcp_reward != 0:
                    reward_into[i] += engineered_close_to_vcp_reward * np.interp(max(config_copy.engineered_reward_min_dist_to_cur_vcp,
                            min(config_copy.engineered_reward_max_dist_to_cur_vcp, np.linalg.norm(rollout_results["state_float"][i]["relative_zone_centers"][0])),
                        ),
                        [config_copy.engineered_reward_min_dist_to_cur_vcp, config_copy.engineered_reward_max_dist_to_cur_vcp],
                        [0.5, -1]
                    ) # normalizing to 1, -1 using np.interp so when we multiply by engineered reward we are reasonable """
                
        if i < n_frames - 1: # apply these rewards unless this is the finish frame
            """if config_copy.final_speed_reward_per_f_per_s != 0:
                # car is driving forward
                reward_into[i] += config_copy.final_speed_reward_per_f_per_s * (
                    rollout_results["state_float"][i]["kart_data"]["speed"]
                )
            if engineered_button_A_pressed_reward != 0 and config_copy.inputs[rollout_results["actions"][i]]["A"] > 0 and config_copy.inputs[rollout_results["actions"][i]]["TriggerRight"] == 0:
                reward_into[i] += engineered_button_A_pressed_reward

            if engineered_item_usage_reward != 0 and config_copy.inputs[rollout_results["actions"][i]]["TriggerLeft"] > 0:
                reward_into[i] += engineered_item_usage_reward"""
            if type(rollout_results["state_float"][i]) != float and rollout_results["state_float"][i]["race_data"]["state"] == 2: # Only apply these rewards during the actual race (Not race finished, not during countdown timer)

                temp_completion_reward = (
                    rollout_results["current_zone_idx"][i] - rollout_results["current_zone_idx"][i - 1]
                ) * config_copy.reward_per_vcp_passed

                # temp_completion_reward = (
                #     rollout_results["race_completion"][i] - rollout_results["race_completion"][i - 1] # meters progressed (negative if backwards)
                # ) * config_copy.reward_per_m_advanced_along_centerline # Based on estimated time to lap completion

                # discourage mushroom usage according to speed increase for the duration of the boost
                # 83 > 120 = 40 speed increase. 1/3rd of progression. so, discount roughly 40% (?) of progression
                if (rollout_results["state_float"][i]["boost_data"]["shroom_boost"] > 60
                    and temp_completion_reward > 0):
                    temp_completion_reward = 0 # 0.6 (40% discount for speed increase) divided by 30/90 as we can't confirm source of boost outside that range
            
                reward_into[i] += temp_completion_reward
                # reward_into_progress[i] += temp_completion_reward
                # reward based on external velocity
                # external_velocity_reward = config_copy.external_velocity_reward_per_f * math.sqrt(rollout_results["state_float"][i]["kart_data"]["external_velocity"][0]**2 + 
                                                                                # rollout_results["state_float"][i]["kart_data"]["external_velocity"][2]**2)
                # reward_into[i] += external_velocity_reward
                # reward_into_ev[i] += external_velocity_reward
            elif type(rollout_results["state_float"][i]) != float and rollout_results["state_float"][i]["race_data"]["state"] == 1:
                # continue using race completion for countdown reward as VCPs are too sparse to encourage moving forward
                completion_reward = (
                    rollout_results["race_completion"][i] - rollout_results["race_completion"][i - 1]
                ) * config_copy.reward_per_m_advanced_along_centerline # Based on estimated time to lap completion
                reward_into[i] += completion_reward
                # reward_into_progress[i] += completion_reward

            """ if (engineered_start_boost_reward != 0
                and rollout_results["state_float"][i]["race_data"]["state"] == 1): # only reward start boost during countdown
                # reward_into[i] += engineered_start_boost_reward * (rollout_results["state_float"][i]["start_boost_charge"] - .3)
                if rollout_results["state_float"][i]["start_boost_charge"] > rollout_results["state_float"][i - 1]["start_boost_charge"]:
                    reward_into[i] += engineered_start_boost_reward if rollout_results["state_float"][i]["start_boost_charge"] < 0.95 else -engineered_start_boost_reward
                else:
                    reward_into[i] += -engineered_start_boost_reward if rollout_results["state_float"][i]["start_boost_charge"] <= 0.925 else 0 """

    # print("Rewards for progress:", np.sum(reward_into_progress))
    # print("Constant reward:", np.sum(reward_into_constant))
    # print("Rewards for EV:", np.sum(reward_into_ev))
    # print("Total:", np.sum(reward_into))
    # print("Seconds run:", n_frames / (config_copy.game_running_fps / config_copy.f_per_action))

    for i in range(n_frames - 1):  # Loop over all frames that were generated
        # Switch memory buffer sometimes
        if random.random() < 0.1:
            list_to_fill = Experiences_For_Buffer_Test if random.random() < config_copy.buffer_test_ratio else Experiences_For_Buffer

        n_steps = min(n_steps_max, n_frames - 1 - i)
        if discard_non_greedy_actions_in_nsteps:
            try:
                first_non_greedy = rollout_results["action_was_greedy"][i + 1 : i + n_steps].index(False) + 1
                n_steps = min(n_steps, first_non_greedy)
            except ValueError:
                pass

        rewards = np.empty(n_steps_max).astype(np.float32)
        for j in range(n_steps):
            rewards[j] = (gamma**j) * reward_into[i + j + 1] + (rewards[j - 1] if j >= 1 else 0)

        state_img = rollout_results["frames"][i]
        state_float = rollout_results["state_float"][i]
        #state_potential = get_potential(rollout_results["state_float"][i])

        # Get action that was played
        action = rollout_results["actions"][i]
        terminal_actions = float((n_frames - 1) - i) if "race_time" in rollout_results else math.inf
        next_state_has_passed_finish = ((i + n_steps) == (n_frames - 1)) and ("race_time" in rollout_results) # last frame in the buffer and race_time exists, thus we finished

        if not next_state_has_passed_finish:
            next_state_img = rollout_results["frames"][i + n_steps]
            next_state_float = rollout_results["state_float"][i + n_steps]
            # next_state_potential = get_potential(rollout_results["state_float"][i + n_steps])
        else:
            # It doesn't matter what next_state_img and next_state_float contain, as the transition will be forced to be final
            next_state_img = state_img
            next_state_float = state_float
            #next_state_potential = 0

        state_float = get_1d_state_floats(state_float, rollout_results["actions"][:i])
        next_state_float = get_1d_state_floats(next_state_float, rollout_results["actions"][:i + n_steps])

        list_to_fill.append(
            Experience(
                state_img,
                state_float,
                # state_potential,
                action,
                n_steps,
                rewards,
                next_state_img,
                next_state_float,
                # next_state_potential,
                gammas,
                terminal_actions,
            )
        )

    number_memories_added_train += len(Experiences_For_Buffer)
    if len(Experiences_For_Buffer) > 1:
        buffer.extend(Experiences_For_Buffer)
    elif len(Experiences_For_Buffer) == 1:
        buffer.add(Experiences_For_Buffer[0])

    number_memories_added_test += len(Experiences_For_Buffer_Test)

    if len(Experiences_For_Buffer_Test) > 1:
        buffer_test.extend(Experiences_For_Buffer_Test)
    elif len(Experiences_For_Buffer_Test) == 1:
        buffer_test.add(Experiences_For_Buffer_Test[0])

    return buffer, buffer_test, number_memories_added_train, number_memories_added_test
