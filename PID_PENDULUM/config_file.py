import numpy as np
import os

# Simulation config:
T_SIMULATION = 1.
dt = 0.001                    
ndt = 10

# Controller params:
kd = 10.

# Robot limits:
tau_max = 0.1
tau_min = - 0.1
q_max = np.pi/2
q_min = 0.
v_max = 10.
v_min = - 10.

ee_frame_name = "joint1"

# Other dynamics params:
simulate_coulomb_friction = 0
simulation_type = 'timestepping' 
tau_coulomb_max = 0*np.ones(1) 
randomize_robot_model = 0
model_variation = 30.0

# Viewer:
use_viewer = 0
simulate_real_time = 0
show_floor = False
PRINT_T = 0.1                  # print every PRINT_N time steps
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_N time steps
CAMERA_TRANSFORM = [1.0568891763687134, 0.7100808024406433, 0.39807042479515076, 
                    0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]
#CAMERA_TRANSFORM = [2.582354784011841, 1.620774507522583, 1.0674564838409424, 0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]
