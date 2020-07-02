import numpy as np

import traj.discrete_time_parameterization

def simple_is_valid(position, velocity, acceleration, jerk):
    """
    Checks values against constant bounds.
    """
    p_end = 10.0
    v_max = 10.0
    a_max = 10.0
    j_max = 10.0
    if position < 0.0 or position > p_end + traj.discrete_time_parameterization.POSITION_THRESHOLD:
        return False
    if velocity > v_max + traj.discrete_time_parameterization.VELOCITY_THRESHOLD:
        return False
    if np.abs(acceleration) > a_max + traj.discrete_time_parameterization.ACCELERATION_THRESHOLD:
        return False
    if np.abs(jerk) > j_max + traj.discrete_time_parameterization.JERK_THRESHOLD:
        return False
    return True


def test_stop_from_stopped():
    assert traj.discrete_time_parameterization.compute_stopping_trajectory(
        0.0, 0.0, 0.0, simple_is_valid, j_max=10.0, delta_t=0.001) is not None
