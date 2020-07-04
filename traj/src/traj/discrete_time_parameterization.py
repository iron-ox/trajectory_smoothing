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


def integrate(trajectory_point, delta_t):
    """
    Propagate the trajectory forward one timestep.
    """
    p, v, a, j = trajectory_point
    a_next = a + j * delta_t
    v_next = v + a * delta_t
    p_next = p + v * delta_t
    return p_next, v_next, a_next


def smooth_end_of_trajectory():
    # Invariant: positions, velocities, and accelerations, and jerks are set to valid values up to
    # and including index time_i - 1.

    # This is the timestep at which we reach zero velocity if use the most negative valid jerk
    # at every timestep. Unfortunately we probably reach zero velocity with a large negative
    # acceleration. We need to reach zero velocity and zero acceleration at the same time,
    # and so need to switch from max negative jerk to max positive jerk at some timestep.
    # the next loop searches for that timestep.
    soonest_velocity_zero_time = time_i - 1

    if accelerations[soonest_velocity_zero_time] > -ACCELERATION_THRESHOLD:
        return (positions[1:time_i],
                velocities[1:time_i],
                accelerations[1:time_i],
                jerks[1:time_i])

    # We need to add a positive jerk section.
    for positive_jerk_start_time_i in range(soonest_velocity_zero_time, 0, -1):
        for time_i in range(positive_jerk_start_time_i, MAX_TIME_STEPS):
            if accelerations[time_i - 1] > -ACCELERATION_THRESHOLD:
                return (positions[1:time_i], velocities[1:time_i], accelerations[1:time_i],
                        jerks[1:time_i])
            elif velocities[time_i - 1] < -VELOCITY_THRESHOLD:
                # We weren't reduce acceleration magnitude to zero before velocity hit zero.
                break

            jerks[time_i] = j_max
            accelerations[time_i] = accelerations[time_i - 1] + jerks[time_i] * delta_t
            velocities[time_i] = velocities[time_i - 1] + accelerations[time_i] * delta_t
            positions[time_i] = positions[time_i - 1] + velocities[time_i] * delta_t

            if not is_valid(positions[time_i], velocities[time_i],
                            accelerations[time_i], jerks[time_i]):
                break

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
            # Check instantaneous limits.
            if not is_valid(*trajectory[time_i, :3], j):
                continue

            trajectory[time_i, 3] = j
            trajectory[time_i + 1, :3] = integrate(trajectory[time_i], delta_t)

            if trajectory[time_i + 1, 1] < VELOCITY_THRESHOLD:
                return trajectory[:time_i + 2]

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

    # Since we start with velocity and acceleration at zero, an empty stopping trajectory will do
    # the trick.
    stopping_trajectory = np.zeros((0, 4))

    for time_i in range(MAX_TIME_STEPS):
        found_valid_jerk = False
        for j in [j_max, -j_max]:
            # Check whether this jerk is instantaneously valid.
            if not is_valid(*trajectory[time_i, :3], j):
                continue

            # Integrate trajectory forward to the next timestep using this jerk.
            trajectory[time_i, 3] = j
            p_next, v_next, a_next = integrate(trajectory[time_i], delta_t)

            next_stopping_trajectory = compute_stopping_trajectory(
                p_next, v_next, a_next, is_valid, j_max, delta_t)
            if next_stopping_trajectory is None:
                # There will be no valid way to stop if we apply this jerk.
                continue

            # We start from the best (highest) jerk and work our way down from there, so
            # we are done as soon as we find a jerk that lets us have a valid stopping
            # trajectory.
            stopping_trajectory = next_stopping_trajectory
            found_valid_jerk = True
            trajectory[time_i + 1, :3] = p_next, v_next, a_next
            break

        if not found_valid_jerk:
            raise RuntimeError('No valid jerk found for timestep {}'.format(time_i))

        if stopping_trajectory[-1][0] >= p_end - POSITION_THRESHOLD:
            # Reached our goal.
            return np.vstack((trajectory[:time_i], stopping_trajectory))

    raise RuntimeError(
        'Failed to find a solution after {} trajectory points'.format(MAX_TIME_STEPS))
