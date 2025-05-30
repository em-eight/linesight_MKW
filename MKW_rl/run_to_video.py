"""
This file contains various utilities to:
    - make .inputs file that can be loaded in TMInterface to replay a set of inputs
    - make a video-widget showing the agent's inputs and beliefs (ie: state value and action advantage)
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np

from config_files import config


# ===============================================================
#       Load run into TMI
# ===============================================================
def write_actions_from_disk_in_tmi_format(infile_path: Path, outfile_path: Path):
    """
    Input : path to a file on disk containing a list of action indices.
    Output: write a text file on disk containing the corresponding inputs, readable by TMI to load the replay
    """
    write_actions_in_csv_format(joblib.load(infile_path), outfile_path)


def write_actions_in_csv_format(action_idxs: List[int], outfile_path: Path):
    """
    Input : list of action indices.
    Output: write a text file on disk containing the corresponding inputs, readable by TMI to load the replay
    """
    outfile = open(outfile_path, "w")
    # Stick range keys must match values present in inputs file.
    convert_range = {
        1: 7,
        0.57: 6,
        0.5 : 5,
        0.43: 4,
        0.36: 3,
        0.29: 2,
        0.22: 1,
        0: 0,
        -0.156: -1,
        -0.22: -2,
        -0.29: -3,
        -0.36: -4,
        -0.43: -5,
        -0.5: -6,
        -1: -7
        }
    for action_idx in action_idxs[:-1]:
        action = config.inputs[action_idx]
        writestr = ""
        writestr += str(1 if action["A"] else 0) + "," # Acceleration
        writestr += str(action["TriggerRight"]) + "," # Brake (Drift)
        writestr += str(action["TriggerLeft"]) + "," # Item
        writestr += str(convert_range[action["StickX"]]) + ","
        writestr += str(convert_range[action["StickY"]]) + ","
        writestr += str(1 if action["Up"] else 0) + "," # Trick
        writestr += str(-1) # 1 if action["B"] else 0 # True Brake
        outfile.write(
            writestr
            + "\n"
        )
    outfile.close()


# ===============================================================
#       Q-values Widget
# ===============================================================


def make_widget_video_from_q_values_on_disk(q_values_path: Path, video_path: Path, q_value_gap):
    make_widget_video_from_q_values(joblib.load(q_values_path), video_path, q_value_gap)


def make_widget_video_from_q_values(q_values: List, video_path: Path, q_value_gap):
    with tempfile.TemporaryDirectory() as zou_dir:
        path_str = "C:\\Users\\chopi\\projects\\trackmania_rl\\temp" # TODO: for input visualization, possible that this is arrow keys only.
        temp_dir = Path(path_str)
        # Place the keys where they should be
        key_reorder = [1, 0, 2, 4, 3, 5, 10, 9, 11, 7, 6, 8] # input list ordering

        def plot_q_and_key(ax, q_values_one_frame, key_number_one_frame):
            ax.axis("off")
            ax.set_ylim([-0.05, 2.1])
            ax.set_xlim([-0.55, 2.55])

            alpha = np.clip(1 + (q_values_one_frame[key_number_one_frame] - q_values_one_frame.max()) / q_value_gap, 0, 1)

            if key_number_one_frame == 3:
                # Brake
                ax.bar(
                    x=1, height=1, width=1.9, bottom=0, align="center", hatch="////", color=(1, 1, 1, 0.5), linewidth=0.5, edgecolor="black"
                )

                # Brake
                ax.bar(
                    x=1,
                    height=alpha * ((key_number_one_frame == 3) | config.inputs[key_number_one_frame]["brake"]),
                    width=1.9,
                    bottom=0,
                    align="center",
                    color=(
                        1 * (alpha < 1.0),
                        1 * (alpha >= 1.0),
                        0,
                        alpha * ((key_number_one_frame == 3) | config.inputs[key_number_one_frame]["brake"]),
                    ),
                    linewidth=0.5,
                    edgecolor="black",
                )

                ax.text(
                    x=1,
                    y=0.5,
                    s="No\nKeypress",
                    ha="center",
                    va="center",
                    # backgroundcolor='w',
                    bbox=dict(boxstyle="square,pad=0.1", fc=(1, 1, 1, 0.7), ec="none"),
                )

            else:
                # Left
                ax.bar(
                    x=0,
                    height=1,
                    width=0.9,
                    bottom=0,
                    align="center",
                    hatch="////" if not config.inputs[key_number_one_frame]["left"] else "",
                    color=(1, 1, 1, 0.5),
                    linewidth=0.5,
                    edgecolor="black",
                )

                # Brake
                ax.bar(
                    x=1,
                    height=1,
                    width=0.9,
                    bottom=0,
                    align="center",
                    hatch="////" if not config.inputs[key_number_one_frame]["brake"] else "",
                    color=(1, 1, 1, 0.5),
                    linewidth=0.5,
                    edgecolor="black",
                )

                # Right
                ax.bar(
                    x=2,
                    height=1,
                    width=0.9,
                    bottom=0,
                    align="center",
                    hatch="////" if not config.inputs[key_number_one_frame]["right"] else "",
                    color=(1, 1, 1, 0.5),
                    linewidth=0.5,
                    edgecolor="black",
                )

                # Accelerate
                ax.bar(
                    x=1,
                    height=1,
                    width=0.9,
                    bottom=1.1,
                    align="center",
                    hatch="////" if not config.inputs[key_number_one_frame]["accelerate"] else "",
                    color=(1, 1, 1, 0.5),
                    linewidth=0.5,
                    edgecolor="black",
                )

                # Left
                ax.bar(
                    x=0,
                    height=alpha * ((key_number_one_frame == 3) | config.inputs[key_number_one_frame]["left"]),
                    width=0.9,
                    bottom=0,
                    align="center",
                    color=(
                        1 * (alpha < 1.0),
                        1 * (alpha >= 1.0),
                        0,
                        alpha * ((key_number_one_frame == 3) | config.inputs[key_number_one_frame]["left"]),
                    ),
                    linewidth=0.5,
                    edgecolor="black",
                )

                # Brake
                ax.bar(
                    x=1,
                    height=alpha * ((key_number_one_frame == 3) | config.inputs[key_number_one_frame]["brake"]),
                    width=0.9,
                    bottom=0,
                    align="center",
                    color=(
                        1 * (alpha < 1.0),
                        1 * (alpha >= 1.0),
                        0,
                        alpha * ((key_number_one_frame == 3) | config.inputs[key_number_one_frame]["brake"]),
                    ),
                    linewidth=0.5,
                    edgecolor="black",
                )

                # Right
                ax.bar(
                    x=2,
                    height=alpha * ((key_number_one_frame == 3) | config.inputs[key_number_one_frame]["right"]),
                    width=0.9,
                    bottom=0,
                    align="center",
                    color=(
                        1 * (alpha < 1.0),
                        1 * (alpha >= 1.0),
                        0,
                        alpha * ((key_number_one_frame == 3) | config.inputs[key_number_one_frame]["right"]),
                    ),
                    linewidth=0.5,
                    edgecolor="black",
                )

                # Accelerate
                ax.bar(
                    x=1,
                    height=alpha * ((key_number_one_frame == 3) | config.inputs[key_number_one_frame]["accelerate"]),
                    width=0.9,
                    bottom=1.1,
                    align="center",
                    color=(
                        1 * (alpha < 1.0),
                        1 * (alpha >= 1.0),
                        0,
                        alpha * ((key_number_one_frame == 3) | config.inputs[key_number_one_frame]["accelerate"]),
                    ),
                    linewidth=0.5,
                    edgecolor="black",
                )

        for frame_id in range(min(40000000, len(q_values))):
            print(f"{(1+frame_id) / len(q_values):.1%}", end="\r")
            fig, axes = plt.subplots(nrows=len(q_values[0]) // 3, ncols=3)
            for key_number in range(len(q_values[0])):
                plot_q_and_key(axes.ravel()[key_reorder[key_number]], q_values[frame_id], key_number)
            fig.suptitle(frame_id)
            plt.savefig(Path(temp_dir) / f"frame_{frame_id:09}.png", transparent=True)
            plt.close(fig)

        print("")
        print("Individual frames done.")
        print("Begin encoding video.")

        subprocess.call(
            [
                "ffmpeg",
                "-framerate",
                "20",
                "-pattern_type",
                "sequence",
                "-start_number",
                "000000000",
                "-i",
                path_str + "\\frame_%09d.png",
                "-pix_fmt",
                "yuva420p",
                "-vcodec",
                "png",
                str(video_path),
            ]
        )
        print("Encoding done, script finished.")
