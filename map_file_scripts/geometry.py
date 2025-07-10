from pathlib import Path
from typing import List

import numpy as np
import numpy.typing as npt
from scipy.interpolate import make_interp_spline


def line_plane_collision_point(plane_normal, plane_point, ray_direction, ray_point, epsilon=1e-6):
    # https://gist.github.com/TimSC/8c25ca941d614bf48ebba6b473747d72
    # All inputs: 3D numpy arrays. No need for them to be normalized.
    # Output : the intersection point between the line and the plane
    ndotu = plane_normal.dot(ray_direction)

    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = ray_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    intersection_point = ray_point + si * ray_direction
    return intersection_point


def fraction_time_spent_in_current_zone(
    current_zone_center: npt.NDArray, next_zone_center: npt.NDArray, current_pos: npt.NDArray, next_pos: npt.NDArray
) -> float:
    # All inputs: 3D numpy arrays. No need for them to be normalized.
    # Output : the intersection point between the line and the plane
    plane_normal = next_zone_center - current_zone_center
    si = -plane_normal.dot(current_pos - (next_zone_center + current_zone_center) / 2) / plane_normal.dot(next_pos - current_pos)
    return max(0, min(1, si))


def extract_cp_distance_interval(raw_position_list: List, target_distance_between_cp_m: float, base_dir: Path):
    """
    :param raw_position_list:               a list of 3D coordinates.

    This function saves on disk a 2D numpy array of shape (N, 3) with the following properties.
    - The first point of the array is raw_position_list[0]
    - The middle of the last and second to last points of the array is raw_position_list[-1]
    - All points in the 2D array are distant of approximately target_distance_between_cp_m from their neighbours.
    - All points of the array lie on the path defined by raw_position_list

    In short, this function resamples a path given in input to return regularly spaced checkpoints.

    It is highly likely that there exists a one-liner in numpy to do all this, but I have yet to find it...
    """
    interpolation_function = make_interp_spline(x=range(len(raw_position_list)), y=raw_position_list, k=1)
    raw_position_list = interpolation_function(np.arange(0, len(raw_position_list) - 1 + 1e-6, 0.01))
    a = np.array(raw_position_list)
    b = np.linalg.norm(a[:-1] - a[1:], axis=1)  # b[i] : distance traveled between point i and point i+1, for i > 0
    c = np.pad(b.cumsum(), (1, 0))  # c[i] : distance traveled between point 0 and point i
    number_zones = round(c[-1] / target_distance_between_cp_m - 0.5) + 0.5  # half a zone for the end
    zone_length = c[-1] / number_zones
    index_first_pos_in_new_zone = np.unique(c // zone_length, return_index=True)[1][1:]
    index_last_pos_in_current_zone = index_first_pos_in_new_zone - 1
    w1 = 1 - (c[index_last_pos_in_current_zone] % zone_length) / zone_length
    w2 = (c[index_first_pos_in_new_zone] % zone_length) / zone_length
    zone_centers = a[index_last_pos_in_current_zone] + (a[index_first_pos_in_new_zone] - a[index_last_pos_in_current_zone]) * (
        w1 / (1e-4 + w1 + w2)
    ).reshape((-1, 1))
    zone_centers = np.vstack(
        (
            raw_position_list[0][None, :],
            zone_centers,
            (2 * raw_position_list[-1] - zone_centers[-1])[None, :],
        )
    )
    np.save(base_dir / "map.npy", np.array(zone_centers).round(0))
    return zone_centers

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
        < len(zone_centers) - 1 - 0  # We can never enter the final virtual zone
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