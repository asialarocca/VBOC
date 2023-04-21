import numpy as np
import random 
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from numpy.linalg import norm as norm
import time
from triplependulum_class_fixedveldir import OCPtriplependulumINIT, SYMtriplependulumINIT
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from my_nn import NeuralNetRegression
import math
from multiprocessing import Pool
from torch.utils.data import TensorDataset, DataLoader

def testing(v):
    valid_data = np.ndarray((0, 6))

    # Reset the time parameters:
    N = ocp.N # number of time steps
    ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
    ocp.ocp_solver.update_qp_solver_cond_N(N)
    
    dt_sym = 1e-2 # time step duration

    # Initialization of the OCP: The OCP is set to find an extreme trajectory. The initial joint positions
    # are set to random values, except for the reference joint whose position is set to an extreme value.
    # The initial joint velocities are left free. The final velocities are all set to 0. The OCP has to maximise 
    # the initial velocity norm in a predefined direction.

    # Selection of the reference joint:
    joint_sel = random.choice([0, 1, 2]) # 0 to select first joint as reference, 1 to select second joint

    # Selection of the start position of the reference joint:
    vel_sel = random.choice([-1, 1]) # -1 to maximise initial vel, + 1 to minimize it
    if vel_sel == -1:
        q_init_sel = q_min
        q_fin_sel = q_max
    else:
        q_init_sel = q_max
        q_fin_sel = q_min

    # Initial velocity optimization direction (the cost is in the form p[0] * dtheta1 + p[1] * dtheta2 + p[2] * dt):
    ran1 = vel_sel * random.random()
    ran2 = random.choice([-1, 1]) * random.random() 
    ran3 = random.choice([-1, 1]) * random.random() 
    norm_weights = norm(np.array([ran1, ran2, ran3]))         
    if joint_sel == 0:
        p = np.array([ran1/norm_weights, ran2/norm_weights, ran3/norm_weights, 0.])
    elif joint_sel == 1:
        p = np.array([ran2/norm_weights, ran1/norm_weights, ran3/norm_weights, 0.])
    else:
        p = np.array([ran2/norm_weights, ran3/norm_weights, ran1/norm_weights, 0.])

    # Initial position of the other joint:
    q_init1 = q_min + random.random() * (q_max-q_min)
    if q_init1 > q_max - eps:
        q_init1 = q_init1 - eps
    if q_init1 < q_min + eps:
        q_init1 = q_init1 + eps
        
    q_init2 = q_min + random.random() * (q_max-q_min)
    if q_init2 > q_max - eps:
        q_init2 = q_init2 - eps
    if q_init2 < q_min + eps:
        q_init2 = q_init2 + eps
        
    q_init3 = q_min + random.random() * (q_max-q_min)
    if q_init3 > q_max - eps:
        q_init3 = q_init3 - eps
    if q_init3 < q_min + eps:
        q_init3 = q_init3 + eps

    # Bounds on the initial state:
    q_init_lb = np.array([q_init1, q_init2, q_init3, v_min, v_min, v_min, dt_sym])
    q_init_ub = np.array([q_init1, q_init2, q_init3, v_max, v_max, v_max, dt_sym])
    if q_init_sel == q_min:
        q_init_lb[joint_sel] = q_min + eps
        q_init_ub[joint_sel] = q_min + eps
    else:
        q_init_lb[joint_sel] = q_max - eps
        q_init_ub[joint_sel] = q_max - eps

    # Guess:
    x_sol_guess = np.empty((N, 7))
    u_sol_guess = np.empty((N, 3))
    for i, tau in enumerate(np.linspace(0, 1, N)):
        x_guess = np.array([q_init1, q_init2, q_init3, 0., 0., 0., dt_sym])
        x_guess[joint_sel] = (1-tau)*q_init_sel + tau*q_fin_sel
        x_guess[joint_sel+3] = 2*(1-tau)*(q_fin_sel-q_init_sel) 
        x_sol_guess[i] = x_guess
        u_sol_guess[i] = np.array([0.,0.,0.])

    cost = 1e6
    all_ok = False

    # Iteratively solve the OCP with an increased number of time steps until the solution does not change.
    # If the solver fails, try with a slightly different initial condition:
    for _ in range(10):
        # Reset current iterate:
        ocp.ocp_solver.reset()

        # Set parameters, guesses and constraints:
        for i in range(N):
            ocp.ocp_solver.set(i, 'x', x_sol_guess[i])
            ocp.ocp_solver.set(i, 'u', u_sol_guess[i])
            ocp.ocp_solver.set(i, 'p', p)
            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, q_min, v_min, v_min, v_min, dt_sym])) 
            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, q_max, v_max, v_max, v_max, dt_sym])) 
            ocp.ocp_solver.constraints_set(i, 'lbu', np.array([-tau_max, -tau_max, -tau_max]))
            ocp.ocp_solver.constraints_set(i, 'ubu', np.array([tau_max, tau_max, tau_max]))
            ocp.ocp_solver.constraints_set(i, 'C', np.zeros((3,7)))
            ocp.ocp_solver.constraints_set(i, 'D', np.zeros((3,3)))
            ocp.ocp_solver.constraints_set(i, 'lg', np.zeros((3)))
            ocp.ocp_solver.constraints_set(i, 'ug', np.zeros((3)))

        C = np.zeros((3,7))
        d = np.array([p[:3].tolist()])
        dt = np.transpose(d)
        C[:,3:6] = np.identity(3)-np.matmul(dt,d) # np.identity(3)-np.matmul(np.matmul(dt,np.linalg.inv(np.matmul(d,dt))),d)
        ocp.ocp_solver.constraints_set(0, "C", C, api='new') 

        ocp.ocp_solver.constraints_set(0, "lbx", q_init_lb)
        ocp.ocp_solver.constraints_set(0, "ubx", q_init_ub)

        ocp.ocp_solver.constraints_set(N, "lbx", np.array([q_min, q_min, q_min, 0., 0., 0., dt_sym]))
        ocp.ocp_solver.constraints_set(N, "ubx", np.array([q_max, q_max, q_max, 0., 0., 0., dt_sym]))
        ocp.ocp_solver.set(N, 'x', x_sol_guess[-1])
        ocp.ocp_solver.set(N, 'p', p)

        # Solve the OCP:
        status = ocp.ocp_solver.solve()

        if status == 0: # the solver has found a solution
            # Compare the current cost with the previous one:
            cost_new = ocp.ocp_solver.get_cost()
            if cost_new > cost - tol: 
                all_ok = True # the time is sufficient to have achieved an optimal solution
                break
            cost = cost_new

            # Update the guess with the current solution:
            x_sol_guess = np.empty((N+1,7))
            u_sol_guess = np.empty((N+1,3))
            for i in range(N):
                x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
                u_sol_guess[i] = ocp.ocp_solver.get(i, "u")
            x_sol_guess[N] = ocp.ocp_solver.get(N, "x")
            u_sol_guess[N] = np.array([0.,0.,0.])

            # Increase the number of time steps:
            N = N + 1
            ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
            ocp.ocp_solver.update_qp_solver_cond_N(N)
        else:
            # Reset the number of steps used in the OCP:
            N = ocp.N
            ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
            ocp.ocp_solver.update_qp_solver_cond_N(N)

            # Initial velocity optimization direction (the cost is in the form p[0] * dtheta1 + p[1] * dtheta2 + p[2] * dt):
            ran1 = p[0] + random.random() * random.choice([-1, 1]) * 0.01
            ran2 = p[1] + random.random() * random.choice([-1, 1]) * 0.01
            ran3 = p[2] + random.random() * random.choice([-1, 1]) * 0.01
            norm_weights = norm(np.array([ran1, ran2, ran3]))
            p = np.array([ran1/norm_weights, ran2/norm_weights, ran3/norm_weights,0])

            # Initial position of the other joint:
            dev = random.random() * random.choice([-1, 1]) * 0.01
            for j in range(3):
            	if j != joint_sel:
                    val = q_init_lb[j] + dev
                    if val > q_max - eps:
                        val = val - eps
                    if val < q_min + eps:
                        val = val + eps
                    q_init_lb[j] = val
                    q_init_ub[j] = val

            # Guess:
            x_sol_guess = np.empty((N, 7))
            u_sol_guess = np.empty((N, 3))
            for i, tau in enumerate(np.linspace(0, 1, N)):
                x_guess = np.array([q_init_lb[0], q_init_lb[1], q_init_lb[2], 0., 0., 0., dt_sym])
                x_guess[joint_sel] = (1-tau)*q_init_sel + tau*q_fin_sel
                x_guess[joint_sel+3] = 2*(1-tau)*(q_fin_sel-q_init_sel) 
                x_sol_guess[i] = x_guess
                u_sol_guess[i] = np.array([0., 0., 0.])

            cost = 1e6

    if all_ok: 
        # Save the optimal trajectory:
        x_sol = np.empty((N+1,7))
        u_sol = np.empty((N,3))
        for i in range(N):
            x_sol[i] = ocp.ocp_solver.get(i, "x")
            u_sol[i] = ocp.ocp_solver.get(i, "u")
        x_sol[N] = ocp.ocp_solver.get(N, "x")

        # Generate the unviable sample in the cost direction:
        x_sym = np.full((N+1,6), None)

        x_out = np.copy(x_sol[0][:6])
        x_out[3] = x_out[3] - eps * p[0]
        x_out[4] = x_out[4] - eps * p[1]
        x_out[5] = x_out[5] - eps * p[2]

        # save the initial state:
        valid_data = np.append(valid_data, [[x_sol[0][0], x_sol[0][1], x_sol[0][2], x_sol[0][3], x_sol[0][4], x_sol[0][5]]], axis = 0)

        # Check if initial velocities lie on a limit:
        if x_out[4] > v_max or x_out[4] < v_min or x_out[3] > v_max or x_out[3] < v_min or x_out[5] > v_max or x_out[5] < v_min:
            is_x_at_limit = True # the state is on dX
        else:
            is_x_at_limit = False # the state is on dV
            x_sym[0] = x_out

        # Iterate through the trajectory to verify the location of the states with respect to V:
        for f in range(1, N):
            if is_x_at_limit:
                x_out = np.copy(x_sol[f][:6])
                norm_vel = norm(x_out[3:])    
                x_out[3] = x_out[3] + eps * x_out[3]/norm_vel
                x_out[4] = x_out[4] + eps * x_out[4]/norm_vel
                x_out[5] = x_out[5] + eps * x_out[5]/norm_vel

                # If the previous state was on a limit, the current state location cannot be identified using
                # the corresponding unviable state but it has to rely on the proximity to the state limits 
                # (more restrictive):
                if x_sol[f][0] > q_max - eps or x_sol[f][0] < q_min + eps or x_sol[f][1] > q_max - eps or x_sol[f][1] < q_min + eps or x_sol[f][2] > q_max - eps or x_sol[f][2] < q_min + eps or x_out[3] > v_max or x_out[3] < v_min or x_out[4] > v_max or x_out[4] < v_min or x_out[5] > v_max or x_out[5] < v_min:
                    is_x_at_limit = True # the state is on dX
                else:
                    is_x_at_limit = False # the state is either on the interior of V or on dV

                    # if the traj detouches from a position limit it usually enters V:
                    if x_sol[f-1][0] > q_max - eps or x_sol[f-1][0] < q_min + eps or x_sol[f-1][1] > q_max - eps or x_sol[f-1][1] < q_min + eps or x_sol[f-1][2] > q_max - eps or x_sol[f-1][2] < q_min + eps:
                        break

                    # Solve an OCP to verify whether the following part of the trajectory is on V or dV. To do so
                    # the initial joint positions are set to the current ones and the final state is fixed to the
                    # final state of the trajectory. The initial velocities are left free and maximized in the 
                    # direction of the current joint velocities.

                    N_test = N - f
                    ocp.ocp_solver.set_new_time_steps(np.full((N_test,), 1.))
                    ocp.ocp_solver.update_qp_solver_cond_N(N_test)

                    # Cost:
                    norm_weights = norm(np.array([x_sol[f][3], x_sol[f][4], x_sol[f][5]]))    
                    p = np.array([-x_sol[f][3]/norm_weights, -x_sol[f][4]/norm_weights, -x_sol[f][5]/norm_weights, 0.]) # the cost direction is based on the current velocity direction

                    # Bounds on the initial state:
                    q_init_lb = np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], v_min, v_min, v_min, dt_sym])
                    q_init_ub = np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], v_max, v_max, v_max, dt_sym])
                    #if x_sol[f][4] < 0.:
                    #    ubx_init[4] = x_sol[f][4]
                    #else:
                    #    lbx_init[4] = x_sol[f][4]
                    #if x_sol[f][3] < 0.:
                    #    ubx_init[3] = x_sol[f][3]
                    #else:
                    #    lbx_init[3] = x_sol[f][3]
                    #if x_sol[f][5] < 0.:
                    #    ubx_init[5] = x_sol[f][5]
                    #else:
                    #    lbx_init[5] = x_sol[f][5]

                    # Guess:
                    x_sol_guess = np.empty((N_test+1, 7))
                    u_sol_guess = np.empty((N_test+1, 3))
                    for i in range(N_test):
                        x_sol_guess[i] = x_sol[i+f]
                        u_sol_guess[i] = u_sol[i+f]
                    u_g = np.array([0.,0.,0.])
                    x_sol_guess[N_test] = x_sol[N]
                    u_sol_guess[N_test] = u_g

                    norm_old = norm(np.array([x_sol[f][3:6]]))
                    norm_bef = 0
                    all_ok = False

                    for _ in range(5):
                        # Reset current iterate:
                        ocp.ocp_solver.reset()

                        # Set parameters, guesses and constraints:
                        for i in range(N_test):
                            ocp.ocp_solver.set(i, 'x', x_sol_guess[i])
                            ocp.ocp_solver.set(i, 'u', u_sol_guess[i])
                            ocp.ocp_solver.set(i, 'p', p)
                            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, q_min, v_min, v_min, v_min, dt_sym])) 
                            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, q_max, v_max, v_max, v_max, dt_sym])) 
                            ocp.ocp_solver.constraints_set(i, 'lbu', np.array([-tau_max, -tau_max, -tau_max]))
                            ocp.ocp_solver.constraints_set(i, 'ubu', np.array([tau_max, tau_max, tau_max]))
                            ocp.ocp_solver.constraints_set(i, 'C', np.zeros((3,7)))
                            ocp.ocp_solver.constraints_set(i, 'D', np.zeros((3,3)))
                            ocp.ocp_solver.constraints_set(i, 'lg', np.zeros((3)))
                            ocp.ocp_solver.constraints_set(i, 'ug', np.zeros((3)))

                        C = np.zeros((3,7))
                        d = np.array([p[:3].tolist()])
                        dt = np.transpose(d)
                        C[:,3:6] = np.identity(3)-np.matmul(dt,d) # np.identity(3)-np.matmul(np.matmul(dt,np.linalg.inv(np.matmul(d,dt))),d)
                        ocp.ocp_solver.constraints_set(0, "C", C, api='new')  

                        ocp.ocp_solver.constraints_set(0, "lbx", q_init_lb)
                        ocp.ocp_solver.constraints_set(0, "ubx", q_init_ub)

                        ocp.ocp_solver.set(N_test, 'x', x_sol_guess[-1])
                        ocp.ocp_solver.set(N_test, 'p', p)
                        ocp.ocp_solver.constraints_set(N_test, 'lbx', np.array([q_min, q_min, q_min, 0., 0., 0., dt_sym])) 
                        ocp.ocp_solver.constraints_set(N_test, 'ubx', np.array([q_max, q_max, q_max, 0., 0., 0., dt_sym])) 

                        # Solve the OCP:
                        status = ocp.ocp_solver.solve()

                        if status == 0: # the solver has found a solution
                            # Compare the current cost with the previous one:
                            x0_new = ocp.ocp_solver.get(0, "x")
                            norm_new = norm(np.array([x0_new[3:6]]))
                            if norm_new < norm_bef + tol:
                                all_ok = True 

                                break
                            norm_bef = norm_new

                            # Update the guess with the current solution:
                            x_sol_guess = np.empty((N_test+1,7))
                            u_sol_guess = np.empty((N_test+1,3))
                            for i in range(N_test):
                                x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
                                u_sol_guess[i] = ocp.ocp_solver.get(i, "u")
                            x_sol_guess[N_test] = ocp.ocp_solver.get(N_test, "x")
                            u_sol_guess[N_test] = np.array([0.,0.,0.])

                            # Increase the number of time steps:
                            N_test = N_test + 1
                            ocp.ocp_solver.set_new_time_steps(np.full((N_test,), 1.))
                            ocp.ocp_solver.update_qp_solver_cond_N(N_test)
                        else:
                            break

                    if all_ok:
                        # Compare the old and new velocity norms:  
                        if norm_new > norm_old + tol: # the state is inside V
                            for i in range(N-f):
                                x_sol[i+f] = ocp.ocp_solver.get(i, "x")
                                u_sol[i+f] = ocp.ocp_solver.get(i, "u")

                            x_out = np.copy(x_sol[f][:6])
                            norm_vel = norm_new  
                            x_out[4] = x_out[4] + eps * x_out[4]/norm_vel
                            x_out[3] = x_out[3] + eps * x_out[3]/norm_vel
                            x_out[5] = x_out[5] + eps * x_out[5]/norm_vel

                            # Check if velocities lie on a limit:
                            if x_out[4] > v_max or x_out[4] < v_min or x_out[3] > v_max or x_out[3] < v_min or x_out[5] > v_max or x_out[5] < v_min:
                                is_x_at_limit = True # the state is on dX
                            else:
                                is_x_at_limit = False # the state is on dV
                                x_sym[f] = x_out

                        else:
                            is_x_at_limit = False # the state is on dV
                            # xv_state[f] = 0

                            # Generate the new corresponding unviable state in the cost direction:
                            x_out = np.copy(x_sol[f][:6])
                            x_out[3] = x_out[3] - eps * p[0]
                            x_out[4] = x_out[4] - eps * p[1]
                            x_out[5] = x_out[5] - eps * p[2]
                            if x_out[joint_sel+3] > v_max:
                                x_out[joint_sel+3] = v_max
                            if x_out[joint_sel+3] < v_min:
                                x_out[joint_sel+3] = v_min
                            x_sym[f] = x_out

                    else: # we cannot say whether the state is on dV or inside V
                        for r in range(f, N):
                            if abs(x_sol[r][4]) > v_max - eps or abs(x_sol[r][3]) > v_max - eps or abs(x_sol[r][5]) > v_max - eps:
                                # Save the viable states at velocity limits:
                                valid_data = np.append(valid_data, [[x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], x_sol[f][4], x_sol[f][5]]], axis = 0)

                        break
     
            else:
                # If the previous state was not on a limit, the current state location can be identified using
                # the corresponding unviable state which can be computed by simulating the system starting from 
                # the previous unviable state:
                u_sym = np.copy(u_sol[f-1])
                sim.acados_integrator.set("u", u_sym)
                sim.acados_integrator.set("x", x_sym[f-1])
                sim.acados_integrator.set("T", dt_sym)
                status = sim.acados_integrator.solve()
                x_out = sim.acados_integrator.get("x")

                x_sym[f] = x_out

                # When the state of the unviable simulated trajectory violates a limit, the corresponding viable state
                # of the optimal trajectory is on dX:
                if x_out[0] > q_max or x_out[0] < q_min or x_out[1] > q_max or x_out[1] < q_min or x_out[2] > q_max or x_out[2] < q_min or x_out[3] > v_max or x_out[3] < v_min or x_out[4] > v_max or x_out[4] < v_min or x_out[5] > v_max or x_out[5] < v_min:
                    is_x_at_limit = True # the state is on dX
                else:
                    is_x_at_limit = False # the state is on dV

            if x_sol[f][0] < q_max - eps and x_sol[f][0] > q_min + eps and x_sol[f][1] < q_max - eps and x_sol[f][1] > q_min + eps and x_sol[f][2] < q_max - eps and x_sol[f][2] > q_min + eps and abs(x_sol[f][3]) > tol and abs(x_sol[f][4]) > tol and abs(x_sol[f][5]) > tol:
                # Save the viable and unviable states:
                valid_data = np.append(valid_data, [[x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], x_sol[f][4], x_sol[f][5]]], axis = 0)

        return  valid_data.tolist()
    else:
        return None

start_time = time.time()

# Ocp initialization:
ocp = OCPtriplependulumINIT()
sim = SYMtriplependulumINIT()

# Position, velocity and torque bounds:
v_max = ocp.dthetamax
v_min = - ocp.dthetamax
q_max = ocp.thetamax
q_min = ocp.thetamin
tau_max = ocp.Cmax

# Pytorch device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Unviable data generation parameter:
eps = 1e-2
tol = 1e-3

cpu_num = 30
num_prob = 1000

# Data generation:
with Pool(cpu_num) as p:
    traj = p.map(testing, range(num_prob))

# traj, statpos, statneg = zip(*temp)
X_temp = [i for i in traj if i is not None]
X_save = np.array([i for f in X_temp for i in f])

model_dir = NeuralNetRegression(6, 500, 1).to(device)
criterion_dir = nn.MSELoss()
optimizer_dir = torch.optim.Adam(model_dir.parameters(), lr=1e-3)

mean_dir, std_dir = torch.mean(torch.tensor(X_save[:,:3].tolist())).to(device).item(), torch.std(torch.tensor(X_save[:,:3].tolist())).to(device).item()
torch.save(mean_dir, 'mean_3pendulum_10')
torch.save(std_dir, 'std_3pendulum_10')

X_train_dir = np.empty((X_save.shape[0],7))
for i in range(X_train_dir.shape[0]):
    X_train_dir[i][0] = (X_save[i][0] - mean_dir) / std_dir
    X_train_dir[i][1] = (X_save[i][1] - mean_dir) / std_dir
    X_train_dir[i][2] = (X_save[i][2] - mean_dir) / std_dir
    vel_norm = norm([X_save[i][3], X_save[i][4], X_save[i][5]])
    if vel_norm != 0:
        X_train_dir[i][3] = X_save[i][3] / vel_norm
        X_train_dir[i][4] = X_save[i][4] / vel_norm
        X_train_dir[i][5] = X_save[i][5] / vel_norm
    X_train_dir[i][6] = vel_norm 

it = 1
val = max(X_train_dir[:,6])

beta = 0.95
n_minibatch = 4096
B = int(X_save.shape[0]*100/n_minibatch) # number of iterations for 100 epoch
it_max = B * 10

# Train the model
while val > 1e-3 and it < it_max:
    ind = random.sample(range(len(X_train_dir)), n_minibatch)

    X_iter_tensor = torch.Tensor([X_train_dir[i][:6] for i in ind]).to(device)
    y_iter_tensor = torch.Tensor([[X_train_dir[i][6]] for i in ind]).to(device)

    # Forward pass
    outputs = model_dir(X_iter_tensor)
    loss = criterion_dir(outputs, y_iter_tensor)

    # Backward and optimize
    loss.backward()
    optimizer_dir.step()
    optimizer_dir.zero_grad()

    val = beta * val + (1 - beta) * loss.item()
    it += 1

    if it % B == 0: 
        print(val)

times = np.array([time.time() - start_time])

X_test = np.load('data3_test_10.npy')
X_test_dir = np.empty((X_test.shape[0],7))
for i in range(X_test_dir.shape[0]):
    X_test_dir[i][0] = (X_test[i][0] - mean_dir) / std_dir
    X_test_dir[i][1] = (X_test[i][1] - mean_dir) / std_dir
    X_test_dir[i][2] = (X_test[i][2] - mean_dir) / std_dir
    vel_norm = norm([X_test[i][3],X_test[i][4],X_test[i][5]])
    if vel_norm != 0:
        X_test_dir[i][3] = X_test[i][3] / vel_norm
        X_test_dir[i][4] = X_test[i][4] / vel_norm
        X_test_dir[i][5] = X_test[i][5] / vel_norm
    X_test_dir[i][6] = vel_norm 

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_test_dir[:,:6]).to(device)
    y_iter_tensor = torch.Tensor(X_test_dir[:,6:]).to(device)
    rmse = np.array([torch.sqrt(criterion_dir(model_dir(X_iter_tensor), y_iter_tensor)).item()])

while time.time() - start_time < 14400:
    # Data generation:
    with Pool(cpu_num) as p:
        traj = p.map(testing, range(num_prob))

    X_temp = [i for i in traj if i is not None]
    X_new = np.array([i for f in X_temp for i in f])
    X_save = np.concatenate((X_save, X_new))

    X_new_dir = np.empty((X_save.shape[0],7))
    for i in range(X_new_dir.shape[0]):
        X_new_dir[i][0] = (X_save[i][0] - mean_dir) / std_dir
        X_new_dir[i][1] = (X_save[i][1] - mean_dir) / std_dir
        X_new_dir[i][2] = (X_save[i][2] - mean_dir) / std_dir
        vel_norm = norm([X_save[i][3], X_save[i][4], X_save[i][5]])
        if vel_norm != 0:
            X_new_dir[i][3] = X_save[i][3] / vel_norm
            X_new_dir[i][4] = X_save[i][4] / vel_norm
            X_new_dir[i][5] = X_save[i][5] / vel_norm
        X_new_dir[i][6] = vel_norm 
    X_train_dir = np.concatenate((X_train_dir, X_new_dir))

    it = 1
    val = max(X_train_dir[:,6])

    # Train the model
    while val > 1e-3 and it < it_max:
        # ind = random.sample(range(len(X_train_dir)), n_minibatch)

        ind = random.sample(range(len(X_train_dir) - len(X_new_dir)), int(n_minibatch / 2))
        ind.extend(
            random.sample(
                range(len(X_train_dir) - len(X_new_dir), len(X_train_dir)),
                int(n_minibatch / 2),
            )
        )

        X_iter_tensor = torch.Tensor([X_train_dir[i][:6] for i in ind]).to(device)
        y_iter_tensor = torch.Tensor([[X_train_dir[i][6]] for i in ind]).to(device)

        # Forward pass
        outputs = model_dir(X_iter_tensor)
        loss = criterion_dir(outputs, y_iter_tensor)

        # Backward and optimize
        loss.backward()
        optimizer_dir.step()
        optimizer_dir.zero_grad()

        val = beta * val + (1 - beta) * loss.item()
        it += 1

        if it % B == 0: 
            print(val)

    times = np.append(times, [time.time() - start_time])

    with torch.no_grad():
        X_iter_tensor = torch.Tensor(X_test_dir[:,:6]).to(device)
        y_iter_tensor = torch.Tensor(X_test_dir[:,6:]).to(device)
        rmse = np.append(rmse, [torch.sqrt(criterion_dir(model_dir(X_iter_tensor), y_iter_tensor)).item()])

print(time.time() - start_time)
print('RMSE test data: ', torch.sqrt(criterion_dir(model_dir(X_iter_tensor), y_iter_tensor)).item()) 

np.save('times_vbocp3.npy', np.asarray(times))
np.save('rmse_vbocp3.npy', np.asarray(rmse))

plt.figure()
plt.plot(times, rmse)

np.save('data3_vbocp_10_14000.npy', np.asarray(X_save))
torch.save(model_dir.state_dict(), 'model_3pendulum_dir_10_14000')

# Plots:
h = 0.01
xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))
xrav = xx.ravel()
yrav = yy.ravel()

with torch.no_grad():
    # Plot the results:
    plt.figure()
    inp = np.float32(
            np.c_[(q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                    (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                    xrav,
                    np.zeros(yrav.shape[0]),
                    np.zeros(yrav.shape[0]),
                    yrav,
                    np.empty(yrav.shape[0]),
                    ]
        )
    for i in range(inp.shape[0]):
        vel_norm = norm([inp[i,3],inp[i,4],inp[i,5]])
        inp[i][0] = (inp[i][0] - mean_dir) / std_dir
        inp[i][1] = (inp[i][1] - mean_dir) / std_dir
        inp[i][2] = (inp[i][2] - mean_dir) / std_dir
        if vel_norm != 0:
            inp[i][4] = inp[i][4] / vel_norm
            inp[i][3] = inp[i][3] / vel_norm
            inp[i][5] = inp[i][5] / vel_norm
        inp[i][6] = vel_norm
    out = (model_dir(torch.from_numpy(inp[:,:6].astype(np.float32)).to(device))).cpu().numpy() 
    y_pred = np.empty(out.shape)
    for i in range(len(out)):
        if inp[i][6] > out[i]:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # for i in range(len(X_save)):
    #     if (
    #         norm(X_save[i][0] - (q_min + q_max) / 2) < 0.1 and
    #         norm(X_save[i][1] - (q_min + q_max) / 2) < 0.1
    #         and norm(X_save[i][3]) < 0.1
    #         and norm(X_save[i][4]) < 0.1
    #     ):
    #         xit.append(X_save[i][2])
    #         yit.append(X_save[i][5])
    # plt.plot(
    #     xit,
    #     yit,
    #     "ko",
    #     markersize=2
    # )
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.ylabel('$\dot{q}_3$')
    plt.xlabel('$q_3$')
    plt.grid()
    plt.title("Classifier section")

    # Plot the results:
    plt.figure()
    inp = np.float32(
            np.c_[(q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                    xrav,
                    (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                    np.zeros(yrav.shape[0]),
                    yrav,
                    np.zeros(yrav.shape[0]),
                    np.empty(yrav.shape[0]),
                    ]
        )
    for i in range(inp.shape[0]):
        vel_norm = norm([inp[i,3],inp[i,4],inp[i,5]])
        inp[i][0] = (inp[i][0] - mean_dir) / std_dir
        inp[i][1] = (inp[i][1] - mean_dir) / std_dir
        inp[i][2] = (inp[i][2] - mean_dir) / std_dir
        if vel_norm != 0:
            inp[i][4] = inp[i][4] / vel_norm
            inp[i][3] = inp[i][3] / vel_norm
            inp[i][5] = inp[i][5] / vel_norm
        inp[i][6] = vel_norm
    out = (model_dir(torch.from_numpy(inp[:,:6].astype(np.float32)).to(device))).cpu().numpy() 
    y_pred = np.empty(out.shape)
    for i in range(len(out)):
        if inp[i][6] > out[i]:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # for i in range(len(X_save)):
    #     if (
    #         norm(X_save[i][0] - (q_min + q_max) / 2) < 0.1 and
    #         norm(X_save[i][2] - (q_min + q_max) / 2) < 0.1
    #         and norm(X_save[i][3]) < 0.1
    #         and norm(X_save[i][5]) < 0.1
    #     ):
    #         xit.append(X_save[i][1])
    #         yit.append(X_save[i][4])
    # plt.plot(
    #     xit,
    #     yit,
    #     "ko",
    #     markersize=2
    # )
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.ylabel('$\dot{q}_2$')
    plt.xlabel('$q_2$')
    plt.grid()
    plt.title("Classifier section")

    # Plot the results:
    plt.figure()
    inp = np.float32(
            np.c_[xrav,
                    (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                    (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                    yrav,
                    np.zeros(yrav.shape[0]),
                    np.zeros(yrav.shape[0]),
                    np.empty(yrav.shape[0]),
                    ]
        )
    for i in range(inp.shape[0]):
        vel_norm = norm([inp[i,3],inp[i,4],inp[i,5]])
        inp[i][0] = (inp[i][0] - mean_dir) / std_dir
        inp[i][1] = (inp[i][1] - mean_dir) / std_dir
        inp[i][2] = (inp[i][2] - mean_dir) / std_dir
        if vel_norm != 0:
            inp[i][4] = inp[i][4] / vel_norm
            inp[i][3] = inp[i][3] / vel_norm
            inp[i][5] = inp[i][5] / vel_norm
        inp[i][6] = vel_norm
    out = (model_dir(torch.from_numpy(inp[:,:6].astype(np.float32)).to(device))).cpu().numpy() 
    y_pred = np.empty(out.shape)
    for i in range(len(out)):
        if inp[i][6] > out[i]:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # for i in range(len(X_save)):
    #     if (
    #         norm(X_save[i][1] - (q_min + q_max) / 2) < 0.1 and
    #         norm(X_save[i][2] - (q_min + q_max) / 2) < 0.1
    #         and norm(X_save[i][4]) < 0.1
    #         and norm(X_save[i][5]) < 0.1
    #     ):
    #         xit.append(X_save[i][0])
    #         yit.append(X_save[i][3])
    # plt.plot(
    #     xit,
    #     yit,
    #     "ko",
    #     markersize=2
    # )
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.ylabel('$\dot{q}_1$')
    plt.xlabel('$q_1$')
    plt.grid()
    plt.title("Classifier section")

X_test = np.load('data3_test_10.npy')
X_test_dir = np.empty((X_test.shape[0],7))
for i in range(X_test_dir.shape[0]):
    X_test_dir[i][0] = (X_test[i][0] - mean_dir) / std_dir
    X_test_dir[i][1] = (X_test[i][1] - mean_dir) / std_dir
    X_test_dir[i][2] = (X_test[i][2] - mean_dir) / std_dir
    vel_norm = norm([X_test[i][4],X_test[i][3],X_test[i][5]])
    if vel_norm != 0:
        X_test_dir[i][5] = X_test[i][5] / vel_norm
        X_test_dir[i][4] = X_test[i][4] / vel_norm
        X_test_dir[i][3] = X_test[i][3] / vel_norm
    X_test_dir[i][6] = vel_norm 

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_test_dir[:,:6]).to(device)
    y_iter_tensor = torch.Tensor(X_test_dir[:,6:]).to(device)
    outputs = model_dir(X_iter_tensor).cpu().numpy()
    print('RRMSE test data wrt VBOCP NN in %: ', math.sqrt(np.sum([((outputs[i] - X_test_dir[i,6])/X_test_dir[i,6])**2 for i in range(len(X_test_dir))])/len(X_test_dir))*100)
    print('RMSE test data wrt VBOCP NN: ', torch.sqrt(criterion_dir(model_dir(X_iter_tensor), y_iter_tensor)))

plt.show()
