import numpy as np

MAX_TIME_STEPS = 10000

# How close we have to be to a given position/velocity/acceleration to consider it "reached".
# These are needed because we are using a discrete approximation of the trajectory. These
# should all be small enough that if you commanded zero velocity starting now, the resulting
# deceleration would be less than the robot's limits.
POSITION_THRESHOLD = 0.01
VELOCITY_THRESHOLD = 0.001
ACCELERATION_THRESHOLD = 0.01
JERK_THRESHOLD = 0.01

def pvaj_to_pppp(p0, v0, a0, j0, delta_t):
    """
    Convert position, velocity, acceleration, and jerk at one timestep to positions for
    the next 4 timesteps.
    """
    a_arr = [a0, a0 + j0 * delta_t]
    v_arr = [v0]
    for a in a_arr:
        v_arr.append(v_arr[-1] + delta_t * a)
    p_arr = [p0]
    for v in v_arr:
        p_arr.append(p_arr[-1] + delta_t * v)
    return p_arr

def pppp_to_pvaj(pppp):
    pass

def integrate(p, v, a, j, delta_t):
    """
    Propagate the trajectory forward one timestep.
    """
    return p + v * delta_t, v + a * delta_t, a + j * delta_t


def smooth_stop_fine_adjustment(trajectory, is_valid, j_max, delta_t, increments=10):
    """
    """
    smoothed_trajectory = np.tile(np.nan, (MAX_TIME_STEPS, 4))
    smoothed_trajectory[:len(trajectory)] = trajectory

    if len(trajectory) == 0:
        return trajectory

    for j in np.linspace(-j_max, ):
        for time_i in range(len(trajectory)):
            pass


def smooth_stop(trajectory, is_valid, j_max, delta_t):
    """
    We start with a trajectory which reaches zero velocity if use the most negative valid jerk
    at every timestep. Unfortunately we probably reach zero velocity with a large negative
    acceleration. We need to reach zero velocity and zero acceleration at the same time,
    and so need to switch from max negative jerk to max positive jerk at some timestep.
    """
    if trajectory[-1, 2] > -ACCELERATION_THRESHOLD:
        # No Need to add a positive jerk section.
        return trajectory

    smoothed_trajectory = np.tile(np.nan, (MAX_TIME_STEPS, 4))
    smoothed_trajectory[:len(trajectory)] = trajectory

    # We need to add a positive jerk section.
    for positive_jerk_start_time_i in range(len(trajectory) - 1, -1, -1):
        for time_i in range(positive_jerk_start_time_i, len(smoothed_trajectory) - 1):
            if not is_valid(*smoothed_trajectory[time_i, :3], j_max):
                return None

            if smoothed_trajectory[time_i, 1] < -VELOCITY_THRESHOLD:
                # We weren't reduce acceleration magnitude to zero before velocity hit zero.
                break

            if smoothed_trajectory[time_i, 2] > -ACCELERATION_THRESHOLD:
                return smoothed_trajectory[:time_i + 1]

            smoothed_trajectory[time_i, 3] = j_max
            smoothed_trajectory[time_i + 1, :3] = integrate(*smoothed_trajectory[time_i], delta_t)

    # We were unable to decelerate.
    return None


def compute_stopping_trajectory(p_start, v_start, a_start, is_valid, j_max, delta_t):
    """
    The jerk returned by this function during the final timestep is meaningless, since we've
    already stopped at that point.
    """
    trajectory = np.tile(np.nan, (MAX_TIME_STEPS, 4))
    trajectory[0, :3] = p_start, v_start, a_start

    # Decelerate until our velocity drops to zero.
    for time_i in range(MAX_TIME_STEPS):
        # Invariant: positions, velocities, and accelerations up to and including index time_i have
        # been defined. Jerks up to and including index time_i - 1 have been defined. positions,
        # velocities, accelerations, and jerks up to and including index time_i - 1 are set to
        # valid values. No guarantees for the validity of position, velocity, and acceleration
        # values at index time_i.

        found_valid_jerk = False
        for j in (-j_max, 0.0):
            if not is_valid(*trajectory[time_i, :3], j):
                continue

            trajectory[time_i, 3] = j
            trajectory[time_i + 1, :3] = integrate(*trajectory[time_i], delta_t)

            # Position, velocity, and acceleration in the next timestep depend on the jerk
            # from this timestep, so we have to look one timestep into the future. In particular,
            # if we don't do this check, the acceleration might become too negative.
            if not is_valid(*trajectory[time_i + 1, :3], 0.0):
                continue

            if trajectory[time_i + 1, 1] < VELOCITY_THRESHOLD:
                return smooth_stop(trajectory[:time_i + 2], is_valid, j_max, delta_t)

            # We try the most desirable jerk (the one that will slow us down the fastest) first.
            # Because of this, we can stop as soon as we find a valid jerk - it is definitely
            # the best one.
            found_valid_jerk = True
            break

        if not found_valid_jerk:
            return None

    raise RuntimeError('Failed to find a solution after {} trajectory points'.format(
        MAX_TIME_STEPS))


def parameterize_path_discrete(p_start, p_end, is_valid, j_max, delta_t):
    # Each row of the trajectory array is one timestep. The columns contain values for position,
    # velocity, acceleration, and jerk, in that order. We start off with all values as NaN so
    # that we can tell if we accidentally use an uninitialized value.
    trajectory = np.tile(np.nan, (MAX_TIME_STEPS, 4))
    trajectory[0][:3] = p_start, 0.0, 0.0

    stopping_trajectory = None
    for time_i in range(MAX_TIME_STEPS):
        next_stopping_trajectory = None

        if is_valid(*trajectory[time_i, :3], j_max):
            # Integrate trajectory forward to the next timestep using this jerk.
            p_next, v_next, a_next = integrate(*trajectory[time_i, :3], j_max, delta_t)

            next_stopping_trajectory = compute_stopping_trajectory(
                p_next, v_next, a_next, is_valid, j_max, delta_t)

        if next_stopping_trajectory is None:
            if stopping_trajectory is None:
                raise RuntimeError("No valid trajectory at start")
            else:
                # Use the jerk from last timestep's stopping trajectory.
                trajectory[time_i, 3] = stopping_trajectory[0, 3]
                trajectory[time_i + 1, :3] = integrate(*trajectory[time_i], delta_t)
                stopping_trajectory = stopping_trajectory[1:]
        else:
            trajectory[time_i, 3] = j_max
            stopping_trajectory = next_stopping_trajectory
            trajectory[time_i + 1, :3] = p_next, v_next, a_next

        if stopping_trajectory[-1][0] >= p_end - POSITION_THRESHOLD:
            # Reached our goal.
            return np.vstack((trajectory[:time_i], stopping_trajectory))

    raise RuntimeError(
        'Failed to find a solution after {} trajectory points'.format(MAX_TIME_STEPS))
