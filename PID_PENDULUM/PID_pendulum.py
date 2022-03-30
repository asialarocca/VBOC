import numpy as np
from numpy import nan
import time, sys
from arc.utils.robot_loaders import loadPendulum
from arc.utils.robot_wrapper import RobotWrapper
from arc.utils.robot_simulator import RobotSimulator
import matplotlib.pyplot as plt
import config_file as conf
import warnings

import pinocchio as pin
np.set_printoptions(precision=3, linewidth=200, suppress=True)

warnings.filterwarnings("ignore") 

r = loadPendulum()
robot = RobotWrapper(r.model, r.collision_model, r.visual_model)

N = int(conf.T_SIMULATION/conf.dt)      # number of time steps
dt = conf.dt
PRINT_N = int(conf.PRINT_T/conf.dt)

tau_max = 0.1
tau_min = - 0.1
q_max = np.pi/2
q_min = 0.
v_max = 10.
v_min = - 10.

q0 = np.array([np.pi/2])
v0 = np.array([0.]) 
t = 0.0
tau    = np.empty(N)*nan    # joint torques
q      = np.empty(N)*nan    # joint angles
v      = np.empty(N)*nan    # joint velocities
dv     = np.empty(N)*nan    # joint accelerations
simu = RobotSimulator(conf, q0, v0, robot)

#THIS IS TO TEST GRAVITY COMPENSATION TORQUES:

#sg      = np.empty(N)*nan    # joint angles
#for k in range(N):
#	q[k] = k*np.pi/(2*N)
#	sg[k] = robot.gravity(np.array([q[k]]))
	
#print(min(sg))
#plt.plot(sg)
#plt.show()

for i in range(N):
	time_start = time.time()
	# read current state from simulator
	v[i] = simu.v
	q[i] = simu.q
	
	if v[i] > v_max or v[i] < v_min or q[i] > q_max or q[i] < q_min:
	    print("CONSTRAINTS VIOLATED")
	    break
	    
	if v[i] < 0.01:
		print("STOPPED")
		break
	
	#M = robot.mass(np.array([q[i]]))
	#h = robot.nle(np.array([q[i]]), np.array([v[i]]))
	#st = robot.gravity(np.array([q[i]]))
	#tau[i] = st
	
	tau[i] = - conf.kd*v[i]
	
	if tau[i] < tau_min:
	    tau[i] = tau_min
	elif tau[i] > tau_max:
	    tau[i] = tau_max
	
	# send joint torques to simulator
	simu.simulate(tau[i], dt, conf.ndt)
	t += conf.dt
    
	time_spent = time.time() - time_start
	if(conf.simulate_real_time and time_spent < conf.dt): 
	    time.sleep(conf.dt-time_spent)
		    
