import numpy as np
import os

# Simulation config:
T_SIMULATION = 1.
dt = 0.0001
ndt = 10

# Controller params:
kd = 10

# Robot limits:
tau_max = [0., 0.]
tau_min = [-0.75, -0.35]
q_max = [np.pi/2, np.pi/2]
q_min = [0., 0.]
v_max = [10., 10]
v_min = [-10., -10.]

ee_frame_name = "joint2"

# Other dynamics params:
simulate_coulomb_friction = 0
simulation_type = 'timestepping'
tau_coulomb_max = 0*np.ones(2)
randomize_robot_model = 0
model_variation = 30.0

# Viewer:
use_viewer = 0
simulate_real_time = 0
show_floor = False
PRINT_T = 0.1                  # print every PRINT_N time steps
# update robot configuration in viwewer every DISPLAY_N time steps
DISPLAY_T = 0.02
CAMERA_TRANSFORM = [2.582354784011841, 1.620774507522583, 1.0674564838409424,
                    0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]
