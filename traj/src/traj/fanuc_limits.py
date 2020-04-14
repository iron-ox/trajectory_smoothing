
# Copyright (c) 2020, G.A. vd. Hoorn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author G.A. vd. Hoorn

import numpy as np

from scipy.interpolate import interp2d


def gen_limit_interpolation_func(no_load_thresh, max_load_thresh, payload_max,
                                 cart_vmax=4.0, num_steps=20,
                                 interp_func='linear'):
    """Generates a function which can be used to lookup the scaled (positive)
    limit for a joint based on the NO and MAX load threshold tables, payload
    and Cartesian velocity of the TCP of a Fanuc with J519.

    The returned function object wraps the function returned by a call to
    scipy.interp2d(..), taking in current Cartesian velocity of the TCP
    (in m/s) and current weight of the payload (in Kg) and returns the 2D
    interpolated velocity, acceleration or jerk limit based on the information
    in the provided threshold tables (see below).

    The payload argument is optional and will default to 'payload_max', as
    provided in the call to 'gen_limit_interpolation_func(..)' (ie: this
    function).

    This will result in the slowest limit being returned, which would result
    in a conservative estimate of the capabilities of the robot for the given
    Cartesian velocity (but should not result in motion execution errors due
    to violating a joint limit).

    Depending on whether threshold tables for velocity, acceleration or jerk
    are passed in, the limits returned are velocity, acceleration or jerk
    limits.

    The threshold tables are expected to conform to the format as returned by
    a controller with J519 upon receipt of a 'Type 3' packet (ie: Request/Ack).
    Assumptions are: 20 elements per table, elements of type float, sorted in
    descending order (ie: max limit -> fully scaled down limit).

    Args:
        no_load_thresh: threshold table for NO load configuration
          list(float)
        max_load_thresh: threshold table for MAX load configuration
          list(float)
        payload_max: maximum payload supported by the robot (Kg)
          float
        cart_vmax: maximum Cartesian velocity supported by the robot (m/s)
          default: 4.0
          float
        num_steps: number of entries in a single threshold table
          default: 20
          int
        interp_func: order of interpolation used. Passed on to interp2d(..)
          default: 'linear'
          str

    Returns:
        Function wrapping the return value of scipy.interp2d(..).

        Args:
          cart_vel: the Cartesian velocity of the TCP (m/s)
            float
          payload: the weight of the current payload of the robot (Kg)
            default: payload_max
            float

        Returns:
          2D interpolated joint limit for the given Cartesian velocity and
          payload.

    Example:

      # create interpolation function for the acceleration limits of J1, with
      # a maximum payload of 25 Kg, and the default maximum Cartesian velocity
      # (of 4.0 m/s), default number of elements in the threshold tables (20)
      # and the default interpolation strategy (linear).
      j1_acc_limit_func = gen_limit_interpolation_func(
          no_load_thresh=[2050.00, 2050.00, ..],
          max_load_thresh=[1601.56, 1601.56, ..],
          payload_max=25.0
      )

      # determine acceleration limit for J1 with TCP moving at 1.5 m/s and
      # with a current payload of 6.3 Kg
      j1_curr_acc_limit = j1_acc_limit_func(cart_vel=1.5, payload=6.3)[0]

      # determine acceleration limits for J1 with TCP moving at 1.1, 1.35 and
      # 1.47 m/s and the default (ie: max) payload
      j1_acc_limits = j1_acc_limit_func(cart_vel=[1.1, 1.35, 1.47])
    """
    len_nlt = len(no_load_thresh)
    len_mlt = len(max_load_thresh)
    if len_nlt != num_steps or len_mlt != num_steps:
        raise ValueError(
            "Threshold table should contain {} elements (got: {} and {} "
            "elements for NO and MAX load respectively)"
            .format(num_steps, len_nlt, len_mlt))

    # TODO: check for negative max payloads
    # TODO: check for negative max cart vel
    # TODO: check for negative num steps

    # TODO: this sets up a full 2D interpolation. Not sure that is what
    # we want. Perhaps we do need to consider the 'binning' on the X-axis
    # (ie: percentage of max cart velocity)
    x = np.linspace(cart_vmax/num_steps, cart_vmax, num_steps)
    y = [0.0, payload_max]
    z = [no_load_thresh, max_load_thresh]
    limit_interp2d = interp2d(x, y, z, kind=interp_func)

    # create function object for caller to use for lookups
    # note: similar to the robot controller, we assume maximum payload
    # if nothing else has been provided
    # TODO: check whether we should optimise for single lookups (ie: instead
    # of mesh/multiple lookups at once): https://stackoverflow.com/a/47233198
    def func(cart_vel, payload=payload_max):
        # TODO: check for negative payload
        # TODO: check for negative cart vel
        return limit_interp2d(cart_vel, payload)
    return func
