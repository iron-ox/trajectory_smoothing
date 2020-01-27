#!/usr/bin/env python
"""
Simple example that fits general trajectory segment, given the start and
end positions/velocities. Start and end acceleration/jerk are assumed to be zero.
"""
from matplotlib import pyplot as plt
import traj


def test_seven_segment(p_start, p_end, v_start, v_end, p_max, v_max, a_max, j_max):
    position, velocity, acceleration, jerk = traj.fit_traj_segment(p_start, p_end, v_start, v_end, p_max, v_max, a_max, j_max)
    traj.plot.plot_trajectory(plt, position, velocity, acceleration, jerk, n_points=100, v_max=v_max, a_max=a_max, j_max=j_max)
    plt.show()
    
p_max=30
v_max=3.0
a_max=4.0
j_max=10.0   

############### CASE A: v_start = v_end = 0    
#case A1:  no limit is reached     
test_seven_segment(0.0, 1.0,      0.0, 0.0,     p_max, v_max, a_max, j_max)    
#case A2:  acc_limit is reached     
test_seven_segment(0.0, 3.0,      0.0, 0.0,     p_max, v_max, a_max, j_max)    
#case A3:  vel_limit and acc_limit are reached    
test_seven_segment(0.0, 5.0,      0.0, 0.0,     p_max, v_max, a_max, j_max)    


############## CASE B: v_start = v_end !=0
#case B1:  no limit is reached
test_seven_segment(0.0, 2.0,      1.0, 1.0,     p_max, v_max, a_max, j_max)    
#case B2:  vel_limit is reached 
test_seven_segment(0.0, 3.0,      2.5, 2.5,     p_max, v_max, a_max, j_max)    
#case B3:  acc_limit is reached
test_seven_segment(0.0, 3.0,      0.5, 0.5,     p_max, v_max, a_max, j_max)    
#case B4:  vel_limit and acc_limit are reached   
test_seven_segment(0.0, 10.0,     0.5, 0.5,     p_max, v_max, a_max, j_max)    



############## CASE C: v_start != v_end !=0, and v_start < v_end [acceleration]
#case C1:  no limit is reached
test_seven_segment(0.0, 2.0,       0.5, 1.5,    p_max, v_max, a_max, j_max)    
#case C2:  vel_limit is reached 
test_seven_segment(0.0, 5.0,      2.0, 2.5,     p_max, v_max, a_max, j_max)    
#case C3:  acc_limit is reached
test_seven_segment(0.0, 3.0,      0.5, 2.5,     p_max, v_max, a_max, j_max)    
#case C4:  vel_limit and acc_limit are reached   
test_seven_segment(0.0, 5.0,       0.5, 2.5,    p_max, v_max, a_max, j_max)    



############## CASE D: v_start != v_end !=0, and v_start > v_end [deceleration]
#case D1:  no limit is reached
test_seven_segment(0.0, 2.0,       1.5, 0.5,    p_max, v_max, a_max, j_max)    
#case D2:  vel_limit is reached 
test_seven_segment(0.0, 5.0,       2.5, 2.0,    p_max, v_max, a_max, j_max)    
#case D3:  acc_limit is reached
test_seven_segment(0.0, 3.0,       2.5, 0.5,    p_max, v_max, a_max, j_max)    
#case D4:  vel_limit and acc_limit are reached   
test_seven_segment(0.0, 5.0,       2.5, 0.5,    p_max, v_max, a_max, j_max)    




############## CASE E: not feasible values considering monotonic segment; can't reach v_end from v_start considering value of p_start and p_end
#case E1: unfeasible acceleration
test_seven_segment(0.0, 0.2,      0.5, 2.5,     p_max, v_max, a_max, j_max)    
#case E2: unfeasible deceleration
test_seven_segment(0.0, 0.2,      2.5, 0.5,     p_max, v_max, a_max, j_max)  



############## CASE F: not feasible values considering start and end values; start and end values beyond the limits     
#case F1: unfeasible v_start 
test_seven_segment(0.0, 0.2,      3.5, 2.5,    p_max, v_max, a_max, j_max)    
#case F2: unfeasible v_end 
test_seven_segment(0.0, 0.2,      0.5, 4.5,    p_max, v_max, a_max, j_max)    
#case F3: unfeasible p_start 
test_seven_segment(40.0, 0.2,      0.5, 2.5,   p_max, v_max, a_max, j_max)    
#case F4: unfeasible p_end 
test_seven_segment(0.0, 40.2,      0.5, 2.5,   p_max, v_max, a_max, j_max) 







