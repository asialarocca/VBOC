import numpy as np
from numpy import nan
import time, sys
from example_robot_data.robots_loader import loadDoublePendulum
from arc.utils.robot_wrapper import RobotWrapper
from arc.utils.robot_simulator import RobotSimulator
import config_file_try as conf

r = loadDoublePendulum()
robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
simu = RobotSimulator(conf, robot)

N = int(conf.T_SIMULATION/conf.dt)      # number of time steps
tau    = np.empty((robot.na, N))*nan    # joint torques
tau_c  = np.empty((robot.na, N))*nan    # joint Coulomb torques
q      = np.empty((robot.nq, N))*nan    # joint angles
v      = np.empty((robot.nv, N))*nan    # joint velocities
dv     = np.empty((robot.nv, N))*nan    # joint accelerations

t = 0.0
dt = conf.dt
q[:,0], v[:,0] = simu.q, simu.v
PRINT_N = int(conf.PRINT_T/conf.dt)

for i in range(N):
    time_start = time.time()

    # read current state from simulator
    v[:,i] = simu.v
    q[:,i] = simu.q
    
    ## compute mass matrix M, bias terms h, gravity terms g
    M = robot.mass(q[:,i])
    h = robot.nle(q[:,i], v[:,i])
    sg = robot.gravity(q[:,i])
    
    # PID control law
    #a_pd[:,i] = conf.kd*([0.0,0.0] - v[:,i]) + conf.kp*([np.pi,0.0] - q[:,i])
    #tau[:,i] = M.dot(a_pd[:,i]) + h
    tau[:,i] = conf.kd*([0.0] - v[:,i]) + conf.kp*([np.pi] - q[:,i])
    
    # send joint torques to simulator
    simu.simulate(tau[i], dt, conf.ndt)
    tau_c[i] = simu.tau_c

    t += conf.dt
        
    time_spent = time.time() - time_start
    if(conf.simulate_real_time and time_spent < conf.dt): 
        time.sleep(conf.dt-time_spent)
