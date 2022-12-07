import time
from double_pendulum_ocp_class_mpc_try import OCPdoublependulumNN, OCPdoublependulumINIT
import numpy as np
import random
import scipy.linalg as lin
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template import AcadosModel
from casadi import SX, vertcat, exp, fmax, tanh, Function
import matplotlib.pyplot as plt
import os
import casadi as cs
import urdf2casadi.urdfparser as u2c
import pinocchio as pin
from scipy.stats import entropy, qmc
import cProfile, pstats, io
from pstats import SortKey
import torch

# ocp = OCPdoublependulumINIT()

# start_time = time.time()
# ocp.compute_problem([-0.7805794638446351, -2.5675506591796875], [0., 0.])
# print("Execution time: %s seconds" % (time.time() - start_time))

# ERROR_MSG = 'You should set the environment variable LOCOSIM_DIR"\n'
# path = os.environ.get('LOCOSIM_DIR', ERROR_MSG)
# urdf = path + "/robot_urdf/generated_urdf/" + "ur5_fix.urdf"
# robot = RobotWrapper.BuildFromURDF(urdf, [path, ])
# frame_name = conf.robot_params['ur5']["ee_frame"]
# x_ee = robot.framePlacement(
#     np.array([0., 0.]), robot.model.getFrameId(frame_name)
# ).translation
# print(x_ee)

with cProfile.Profile() as pr:

    # Ocp initialization:
    model = torch.load('model_save')
    ocp = OCPdoublependulumNN(model,-0.8750016093254089,1.6528178453445435) # double pendulum MPC
    #mpc = OCPdoublependulumINIT()
    
    status = ocp.compute_problem(np.array([-0.80624, -2.50447]), np.array([ 0.02808, -0.88789]))
    print('done')
    plt.show()

#p = pstats.Stats(pr)
#p.sort_stats('cumulative').print_stats(10)


#ur5 = u2c.URDFparser()
#path_to_urdf = absPath = os.path.dirname(
#    os.path.abspath(__file__)) + '/../../robot_urdf/generated_urdf/ur5_fix.urdf'
#ur5.from_file(path_to_urdf)
#root = "base_link"
#tip = "tool0"
#print(ur5.robot_desc.get_chain(root, tip))

#func = ur5.get_forward_kinematics(root, tip)
