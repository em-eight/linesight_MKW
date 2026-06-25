"""
This file contains a run's configuration.
It is expected that this file contains all relevant information about a run.

Two files named "config.py" and "config_copy.py" coexist in the same folder.

At the beginning of training, parameters are copied from config.py to config_copy.py
During training, config_copy.py will be reloaded at regular time intervals.
config_copy.py is NOT tracked in git, as it is essentially a temporary file.
You must manually copy config.py into config_copy.py to apply changes mid-training

Training parameter modifications made during training in config_copy.py will be applied on the fly
without losing the existing content of the replay buffer.

The content of config.py may be modified after starting a run: it will have no effect on the ongoing run.
This setup provides the possibility to:
1) Modify training parameters on the fly
2) Continue to code, use git, and modify config.py without impacting an ongoing run.

"""
import numpy as np

from config_files.inputs_list import *
from config_files.user_config import *

W_downsized = 150
H_downsized = 100

run_name = "rMC_dolphin_flow_test2"

use_pynoko = True
if use_pynoko:
    import pynoko # used for data-types

# gpu_collectors_count is the number of Dolphin instances that will be launched in parallel.
# It is recommended that users adjust this number depending on the performance of their machine.
# We recommend trying different values and finding the one that maximises the number of batches done per unit of time.
# Note that each additional instance requires a separate folder containing a full Dolphin installation, and should be named sequentially.
# (Dolphin's game save files cannot be shared between instances)
# For instance, if the original install is called 'dolphin_folder', installations 2 and 3 should be named 'dolphin_folder2' and 'dolphin_folder3'.
gpu_collectors_count = 4

global_schedule_speed = 1.5
n_steps = 3

f_per_action = 4
game_running_fps = 60
use_miniraces = True

"""
NETWORK SETTINGS
"""

# Zone centers passed to the network
n_zone_centers_in_inputs = 40
one_every_n_zone_centers_in_inputs = 5
n_zone_centers_extrapolate_after_end_of_map = 500
n_zone_centers_extrapolate_before_start_of_map = 20

n_prev_actions_in_inputs = 5
n_prev_drift_actions_in_inputs = 10

# Numper of game_data points + (7 (number of input buttons in each input) * number of previous actions) + number of previous drift inputs + (3 * n zone centers)
float_input_dim = 56 + 7 * n_prev_actions_in_inputs + n_prev_drift_actions_in_inputs + 3 * n_zone_centers_in_inputs

float_hidden_dim = 256
conv_head_output_dim = 4320
dense_hidden_dimension = 512
iqn_embedding_dimension = 64
iqn_n = 8  # must be an even number because we sample tau symmetrically around 0.5
iqn_k = 32  # must be an even number because we sample tau symmetrically around 0.5
iqn_kappa = 5e-3
use_ddqn = False
use_MINTO = False

# TODO: convert to dictionary or list of layers to apply spectral norm to, and whether it is active in those layers
use_spectral_norm = True

use_munchausen_reward_augmentation = False
munchausen_alpha = 0.9
munchausen_temperature = 0.05
munchausen_clip = -0.05

prio_alpha = np.float32(0.0)  # Rainbow-IQN paper: 0.2, Rainbow paper: 0.5, PER paper 0.6
prio_epsilon = np.float32(2e-3)  # Defaults to 10^-6 in stable-baselines
prio_beta = np.float32(1)
prio_uper_lam = np.float32(1.0)

# State-action pair processed in action exploration will be discarded after randomly selected this amount of times
number_times_single_memory_is_used_before_discard = 32  # 32 // 4

epsilon_schedule = [
    (0, 1),
    (50_000, 1),
    (300_000, 0.15),
    (3_000_000 * global_schedule_speed, 0.07),
    (7_000_000 * global_schedule_speed, 0.03),
]
epsilon_boltzmann_schedule = [
    (0, 0.15),
    (3_000_000 * global_schedule_speed, 0.03),
    (5_000_000 * global_schedule_speed, 0.02),
]
tau_epsilon_boltzmann = 0.01
discard_non_greedy_actions_in_nsteps = True
buffer_test_ratio = 0.05

batch_size = 512
weight_decay_lr_ratio = 1 / 50
adam_epsilon = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999

single_reset_flag = 0
reset_every_n_frames_generated = 400_000_00000000
additional_transition_after_reset = 1_600_000
last_layer_reset_factor = 0.8  # 0 : full reset, 1 : nothing happens
overall_reset_mul_factor = 0.01  # 0 : nothing happens ; 1 : full reset

clip_grad_value = 1000
clip_grad_norm = 30

# Number of state-action pairs we train before updating the Target Network as defined by DQN.
number_memories_trained_on_between_target_network_updates = 2048
soft_update_tau = 0.02

# Every n batches, each collection process updates it's network to match the current Online Network as defined by DQN
send_shared_network_every_n_batches = 10
update_inference_network_every_n_actions = 20

target_self_loss_clamp_ratio = 2

# Schedule how many state-action pairs we save in memory at specific sections of training
memory_size_schedule = [
    (0, (50_000, 20_000)),
    (1_000_000 * global_schedule_speed, (125_000, 50_000)),
    (3_000_000 * global_schedule_speed, (250_000, 125_000)),
    (5_000_000 * global_schedule_speed, (400_000, 250_000)),
    (7_500_000 * global_schedule_speed, (700_000, 400_000)),
    (10_000_000 * global_schedule_speed, (1_000_000, 400_000)),
]
lr_schedule = [
    (0, 1e-3),
    (3_000_000 * global_schedule_speed, 2e-4),
    (12_000_000 * global_schedule_speed, 2e-4),
    (15_000_000 * global_schedule_speed, 1e-4),
    (52_000_000, 1e-4),
    (55_000_000, 5e-5),
    (100_000_000, 5e-5),
    (103_000_000, 1e-5),
]
"""
    (0, 1e-3),
    (3_000_000 * global_schedule_speed, 5e-5),
    (12_000_000 * global_schedule_speed, 5e-5),
    (15_000_000 * global_schedule_speed, 1e-5),

    (0, 1e-3),
    (3_000_000 * global_schedule_speed, 2e-4),
    (12_000_000 * global_schedule_speed, 2e-4),
    (15_000_000 * global_schedule_speed, 1e-4),
    (57_000_000, 1e-4),
    (60_000_000, 5e-5),
    (114_000_000, 5e-5),
    (117_000_000, 1e-5),
    (212_000_000, 1e-5),
    (215_000_000, 5e-6),"""

tensorboard_suffix_schedule = [
    (0, ""),
    (6_000_000 * global_schedule_speed, "_2"),
    (15_000_000 * global_schedule_speed, "_3"),
    (30_000_000 * global_schedule_speed, "_4"),
    (45_000_000 * global_schedule_speed, "_5"),
    (80_000_000 * global_schedule_speed, "_6"),
    (150_000_000 * global_schedule_speed, "_7"),
]
# Weight of future reward scaling (1 = optimize returns, 0 = optimize immediate reward)
gamma_schedule = [
    (0, 0.999),
    (1_500_000, 0.999),
    (2_500_000, 1),
] if use_miniraces else [
    (0, 0.993),
]
# Mini-race disable commit:
# https://github.com/Linesight-RL/linesight/commit/c171384c086714f465a7f71949dd047e497875a8

munchausen_gamma = 0.993

"""
RACE SETTINGS
"""

temporal_mini_race_duration_s = 7 # Mini-race duration

LC_punish_line = 44700 # Punish the AI for having a position.x value greater than this value
DC_punish_line_z = -1300
DC_punish_line_x = -10000
# Punish AI for entering illegal area (forcing shortcuts/correct mushroom usage)
illegal_zone_punish_rate = 0

rMC_drift_bonus_start = 0.77
rMC_drift_bonus_end = 0.87

Mushroom_point = 4.24
# mushroom points for each track (because these are annoying to collect, and will be useful in the future for multi-track training)
all_mushroom_points = {
    "LC": 4.65,
    "rBC":4.17,
    "rMC3":4.63,
    "rGV2":4.30,
    "rMR":4.29,
    "DC": 4.18,
    "MMM": 4.38,
    "MC": 4.69,
    "CM": 4.27,
    "DKS": 4.32,
    "rMC": 4.24,
    "rYF": 4.06,
    "rDH": 4.65,
    "rWS": 4.36,
}

game_reboot_interval = 3600 * 10  # Restart dolphin every x seconds
running_speed = 80 # TODO: UNUSED; Remove from GameManager instantiation

# race not completed in time
cutoff_rollout_if_race_not_finished_within_duration_f = game_running_fps * 180 # in seconds
# No progress has been made
cutoff_rollout_if_no_vcp_passed_within_duration_f = game_running_fps * 10 # in seconds

# TODO: Do not save runs more than 10% slower than PB time. (save percentage value as variable here)
# Do not save runs until after we start getting roughly human-level results (i.e. prevent saving 1000s of extra bad runs) -- Currently disabled
frames_before_save_best_runs = 250_000

# Values to detect the agent going the wrong way
distance_between_checkpoints = 300
road_width = 30000
max_allowable_distance_to_virtual_checkpoint = np.sqrt((distance_between_checkpoints / 2) ** 2 + (road_width / 2) ** 2)

# Restart intervals in case of lost connection or game crash
timeout_during_run_ms = 10_100
timeout_between_runs_ms = 600_000_000
tmi_protection_timeout_s = 500

oversample_long_term_steps = 40
oversample_maximum_term_steps = 5

temporal_mini_race_duration_f = game_running_fps * temporal_mini_race_duration_s
temporal_mini_race_duration_actions = temporal_mini_race_duration_f // f_per_action
min_horizon_to_update_priority_actions = temporal_mini_race_duration_actions - 40
# If mini_race_time == mini_race_duration this is the end of the minirace
margin_to_announce_finish_meters = 700

use_race_restart = False
restart_race_command = "restart_race" # can use basically anything so long as it doesn't conflict with a savestate filename.

"""
REWARD SETTINGS
"""

"""
It is recommended to work with standardized values for the inputs of the neural network.
The same recommendation holds for the output of the neural network.
Since the neural network outputs returns, you want your returns to be somewhat normalized.
What does this mean?

It means that when the agent plays well in a favorable scenario, returns should be somewhere around 2.
If the agent plays bad in an unfavorable scenario, returns should be somewhere around -2.
If the agent plays neither good neither bad, returns should be around 0.

For Trackmania, we could say that
"playing well" is being able to achieve 300km/h on average.  We can calculate the theoretical returns obtained by the agent and make sure it's around 2.
"playing bad" is achieving 100km/h on average. We can calculate the theoretical returns obtained by the agent and make sure it's around -2.

However, we had a bump in performance when the range was -3 to -7 than when the range was +2 to -2 in Trackmania.
We still do not understand why. There's no theoretical reason for this...
"""
# Lower bound of rewards
constant_reward_per_action = -7 / (temporal_mini_race_duration_s * (game_running_fps / f_per_action))

# Expected average speed of 100 units per frame (u/f)
# VCP distance of 300 units, normalized by vcp generation code ( Thank you pb4 & Agade :] )
# 60fps (internal, unaffected by game_running_speed)
# 6000 u/s for doing well, 6000/300 = 20 VCPs per s (Note that VCPs are not on the optimal lines, thus effective speed is higher than 100u/f)
expected_vcp_passed_per_s = 20

# +4 reward over the course of a minirace for driving well (passing VCPs)
reward_per_vcp_passed = 4 / (expected_vcp_passed_per_s * temporal_mini_race_duration_s)

expected_lap_duration_s = 45 # This value controls the inverse amount of reward given for the start slide
expected_lap_duration_per_action = expected_lap_duration_s * (game_running_fps / f_per_action)
average_lap_increment_per_action = 1 / expected_lap_duration_per_action
total_second_increment_expected = 3 / (expected_lap_duration_s * 3 / temporal_mini_race_duration_s)

reward_per_m_advanced_along_centerline = 4 / total_second_increment_expected # Value is used only for start-slide

reward_per_drift_boost_frame = 0 / temporal_mini_race_duration_actions # add reward for being in a drift boost

# Shaped reward for being close to VCPs (Guaranteed to equal 0 over the course of the race)
shaped_reward_dist_to_cur_vcp = -0.000
shaped_reward_min_dist_to_cur_vcp = 1500
shaped_reward_max_dist_to_cur_vcp = 9000
engineered_reward_min_dist_to_cur_vcp = 900 # min distance
engineered_reward_max_dist_to_cur_vcp = 9000 # max distance
shaped_reward_point_to_vcp_ahead = 0

engineered_close_to_vcp_reward_schedule = [
    (0, 0),
]
"""(0, 5),
    (1_000_000, 5),
    (5_000_000, 3),
    (8_000_000, 1),
    (10_000_000, 0),"""

# Punish A.I. for using an item as a ratio multiplier of progress made during boost
engineered_item_usage_reward_schedule = [
    (0, 0),
]
engineered_supergrinding_reward_schedule = [
    (0, 0),
]

# Average EV expected for average performance
expected_average_external_velocity = 40
# give +4 reward total for the duration of a mini-race for average performance
external_velocity_reward_per_f = (0 / temporal_mini_race_duration_actions) / expected_average_external_velocity
# Reward maintaining ev based on an average of 60 as being optimized
engineered_external_velocity_reward_schedule = [
    (0, external_velocity_reward_per_f),
    (15_000_000, external_velocity_reward_per_f),
    (25_000_000, external_velocity_reward_per_f),
    (50_000_000, external_velocity_reward_per_f),
    (60_000_000, external_velocity_reward_per_f),
]

expected_ev_punish_line = 0

shaped_ev_reward_multiplier = 0

# ---------------------------------------------------
airtime_reward_per_action = 0

# Reward/Punish AI for being in the air
engineered_airtime_reward_schedule = [
    (0, airtime_reward_per_action),
    (50_000, airtime_reward_per_action),
    (15_000_000 * global_schedule_speed, airtime_reward_per_action),
    (40_000_000 * global_schedule_speed, airtime_reward_per_action),
    (50_000_000 * global_schedule_speed, airtime_reward_per_action),
]

# Reward AI for performing a trick
engineered_trick_reward_schedule = [
    (0, 0),
]


# Tensorboard settings
plot_race_time_left_curves = False
n_transitions_to_plot_in_distribution_curves = 1000
make_highest_prio_figures = False
apply_randomcrop_augmentation = False
n_pixels_to_crop_on_each_side = 2

max_rollout_queue_size = 1 # Unknown

use_jit = True

threshold_to_save_all_runs_ms = -1 # unused

""" 
============================================      MAP CYCLE     =======================================================

In this section we define the map cycle.

It is a list of iterators, each iterator must return tuples with the following information:
    - short map name        (string):   for logging purposes
    - map path (savestate file path):   (string): to automatically load the map in game. 
    - reference line path   (string):   where to find the reference line for this map
    - is_explo              (boolean):  whether the policy when running on this map should be exploratory
    - fill_buffer           (boolean):  whether the memories generated during this run should be placed in the buffer 

The map cycle may seem complex at first glance, but it provides a large amount of flexibility:
    - can train on some maps, test blindly on others
    - can train more on some maps, less on others
    - can define multiple reference lines for a given map
    - etc...

The example below defines a simple cycle where the agent alternates between four exploratory runs on Luigi Circuit (LC), and one 
evaluation run on the same map.

map_cycle = [
    [("LC", "linesight_savestates\\LC_F_Sp.sav", "LC.npy", True, True)] * 4,
    [("LC", "linesight_savestates\\LC_F_Sp.sav", "LC.npy", False, True)] * 1,
]
"""

nadeo_maps_to_train_and_test = [
    "A01-Race",
    # "A02-Race",
    "A03-Race",
    # "A04-Acrobatic",
    "A05-Race",
    # "A06-Obstacle",
    "A07-Race",
    # "A08-Endurance",
    # "A09-Race",
    # "A10-Acrobatic",
    "A11-Race",
    # "A12-Speed",
    # "A13-Race",
    "A14-Race",
    "A15-Speed",
    "B01-Race",
    "B02-Race",
    "B03-Race",
    # "B04-Acrobatic",
    "B05-Race",
    # "B06-Obstacle",
    # "B07-Race",
    # "B08-Endurance",
    # "B09-Acrobatic",
    "B10-Speed",
    # "B11-Race",
    # "B12-Race",
    # "B13-Obstacle",
    "B14-Speed",
    # "B15-Race",
]

map_cycle = []
# for map_name in nadeo_maps_to_train_and_test:
#   short_map_name = map_name[0:3]
#   map_cycle.append([(short_map_name, f'"Official Maps\{map_name}.Challenge.Gbx"', f"{map_name}_0.5m_cl2.npy", True, True)] * 4)
#   map_cycle.append([(short_map_name, f'"Official Maps\{map_name}.Challenge.Gbx"', f"{map_name}_0.5m_cl2.npy", False, True)] * 1)


map_cycle += [
    # [("rGV2", "linesight_savestates/rGV2_F_FR_linux.sav", "rGV2.npy", True, True)] * 4,
    # [("rGV2", "linesight_savestates/rGV2_F_FR_linux.sav", "rGV2.npy", False, True)] * 1,
    # [("rMC3", "linesight_savestates\\rMC3_D_MB.sav", "rMC3.npy", True, True)] * 4,
    # [("rMC3", "linesight_savestates\\rMC3_D_MB.sav", "rMC3.npy", False, True)] * 1,
    # [("rMC3", "__slot__2", "rMC3.npy", True, True)] * 4, # Using __slot__X for dolphin save slots. Not recommended, as the save state depends on the dolphin save.
    # [("rMC3", "__slot__2", "rMC3.npy", False, True)] * 1,
    # [("rMC", "linesight_savestates/rMC_F_FR_linux.sav", "rMC.npy", True, True)] * 4,
    # [("rMC", "linesight_savestates/rMC_F_FR_linux.sav", "rMC.npy", False, True)] * 1,
    [("rMC", [pynoko.Course.GCN_Mario_Circuit, pynoko.Character.Funky_Kong, pynoko.Vehicle.Flame_Runner], "rMC.npy", True, True)] * 4,
    [("rMC", [pynoko.Course.GCN_Mario_Circuit, pynoko.Character.Funky_Kong, pynoko.Vehicle.Flame_Runner], "rMC.npy", False, True)] * 1,
]
