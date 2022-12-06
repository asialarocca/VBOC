
import numpy as np
import os

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

N_SIMULATION = 1             # number of time steps simulated
dt = 0.01                   # controller time step

w_ee = 0.0                      # weight of end-effector task
w_posture = 1.0                 # weight of joint posture task
w_torque_bounds = 1.0           # weight of the torque bounds
w_joint_bounds = 1.0            # weight of the joint bounds

kp_ee = 100.0                   # proportional gain of end-effector constraint
kp_posture = 1.0                # proportional gain of joint posture task

tau_max_scaling = 1.           # scaling factor of torque bounds
v_max_scaling = 1.             # scaling factor of velocity bounds
q_max_scaling = 1.             # scaling factor of position bounds

ee_frame_name = "ee_fixed_joint"        # end-effector frame name
ee_task_mask = np.array([1, 1, 1, 1, 1, 1]).T

PRINT_N = 500                   # print every PRINT_N time steps
DISPLAY_N = 20                  # update robot configuration in viwewer every DISPLAY_N time steps
CAMERA_TRANSFORM = [2.582354784011841, 1.620774507522583, 1.0674564838409424, 0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]

ERROR_MSG = 'You should set the environment variable UR5_MODEL_DIR to something like "$DEVEL_DIR/install/share"\n';
path      = os.environ.get('UR5_MODEL_DIR', ERROR_MSG)
#path      = '/opt/openrobots/share/'
#urdf      = path + "example-robot-data/robots/ur_description/urdf/ur5_robot.urdf";
#srdf      = path + 'example-robot-data/robots/ur_description/srdf/ur5_robot.srdf'
