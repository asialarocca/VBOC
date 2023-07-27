import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import random 
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
import time
from doublependulum_class_vboc import OCPdoublependulumINIT, SYMdoublependulumINIT
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from my_nn import NeuralNetDIR
import math
from multiprocessing import Pool

def data_generation(v):

    valid_data = np.ndarray((0, 4))

    # Reset the time parameters:
    N = ocp.N # number of time steps
    ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
    ocp.ocp_solver.update_qp_solver_cond_N(N)

    # Initialization of the OCP: The OCP is set to find an extreme trajectory. The initial joint positions
    # are set to random values, except for the reference joint whose position is set to an extreme value.
    # The final velocities are all set to 0. The OCP has to maximise 
    # the initial velocity norm in a predefined direction.

    # Selection of the reference joint:
    joint_sel = random.choice([0, 1]) # 0 to select first joint as reference, 1 to select second joint
    joint_oth = int(1 - joint_sel)

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
    norm_weights = norm(np.array([ran1, ran2]))         
    if joint_sel == 0:
        p = np.array([ran1/norm_weights, ran2/norm_weights, 0.])
    else:
        p = np.array([ran2/norm_weights, ran1/norm_weights, 0.])

    # Initial position of the other joint:
    q_init_oth = q_min + random.random() * (q_max-q_min)
    if q_init_oth > q_max - eps:
        q_init_oth = q_init_oth - eps
    if q_init_oth < q_min + eps:
        q_init_oth = q_init_oth + eps

    OCP_store_ic = [vel_sel+1+joint_sel, ran1, ran2, q_init_oth]

    # Bounds on the initial state:
    q_init_lb = np.array([q_min, q_min, v_min, v_min, dt_sym])
    q_init_ub = np.array([q_max, q_max, v_max, v_max, dt_sym])
    if q_init_sel == q_min:
        q_init_lb[joint_sel] = q_min + eps
        q_init_ub[joint_sel] = q_min + eps
    else:
        q_init_lb[joint_sel] = q_max - eps
        q_init_ub[joint_sel] = q_max - eps
    q_init_lb[joint_oth] = q_init_oth
    q_init_ub[joint_oth] = q_init_oth

    # Guess:
    x_sol_guess = np.empty((N, 5))
    u_sol_guess = np.empty((N, 2))
    for i, tau in enumerate(np.linspace(0, 1, N)):
        x_guess = np.array([q_init_oth, q_init_oth, 0., 0., dt_sym])
        x_guess[joint_sel] = (1-tau)*q_init_sel + tau*q_fin_sel
        x_guess[joint_sel+2] = 2*(1-tau)*(q_fin_sel-q_init_sel) 
        x_sol_guess[i] = x_guess
        u_sol_guess[i] = np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x_guess[0]),ocp.g*ocp.l2*ocp.m2*math.sin(x_guess[1])])

    cost = 1e6
    all_ok = False

    q_lb = np.array([q_min, q_min, v_min, v_min, dt_sym])
    q_ub = np.array([q_max, q_max, v_max, v_max, dt_sym])
    u_lb = np.array([-tau_max, -tau_max])
    u_ub = np.array([tau_max, tau_max])
    q_fin_lb = np.array([q_min, q_min, 0., 0., dt_sym])
    q_fin_ub = np.array([q_max, q_max, 0., 0., dt_sym])

    # Iteratively solve the OCP with an increased number of time steps until the solution does not change.
    # If the solver fails, try with a slightly different initial condition:
    for _ in range(10):

        # Solve the OCP:
        status = ocp.OCP_solve(x_sol_guess, u_sol_guess, p, q_lb, q_ub, u_lb, u_ub, q_init_lb, q_init_ub, q_fin_lb, q_fin_ub)

        if status == 0: # the solver has found a solution

            # Compare the current cost with the previous one:
            cost_new = ocp.ocp_solver.get_cost()

            if cost_new > cost - tol: # the time is sufficient to have achieved an optimal solution
                all_ok = True
                break

            cost = cost_new

            # Update the guess with the current solution:
            x_sol_guess = np.empty((N+1,5))
            u_sol_guess = np.empty((N+1,2))
            for i in range(N):
                x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
                u_sol_guess[i] = ocp.ocp_solver.get(i, "u")
            x_sol_guess[N] = ocp.ocp_solver.get(N, "x")
            u_sol_guess[N] = np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x_sol_guess[N][0]),ocp.g*ocp.l2*ocp.m2*math.sin(x_sol_guess[N][1])])

            # Increase the number of time steps:
            N = N + 1
            ocp.N = N
            ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
            ocp.ocp_solver.update_qp_solver_cond_N(N)

        else:

            # Reset the number of steps used in the OCP:
            N = ocp.N
            ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
            ocp.ocp_solver.update_qp_solver_cond_N(N)

            # Initial velocity optimization direction (the cost is in the form p[0] * dtheta1 + p[1] * dtheta2 + p[2] * dt):
            ran1 = ran1 + random.random() * random.choice([-1, 1]) * 0.01
            ran2 = ran2 + random.random() * random.choice([-1, 1]) * 0.01
            norm_weights = norm(np.array([ran1, ran2]))         
            if joint_sel == 0:
                p = np.array([ran1/norm_weights, ran2/norm_weights, 0.])
            else:
                p = np.array([ran2/norm_weights, ran1/norm_weights, 0.])

            # Initial position of the other joint:
            q_init_oth = q_init_oth + random.random() * random.choice([-1, 1]) * 0.01
            if q_init_oth > q_max - eps:
                q_init_oth = q_init_oth - eps
            if q_init_oth < q_min + eps:
                q_init_oth = q_init_oth + eps

            # Bounds on the initial state:
            q_init_lb[joint_oth] = q_init_oth
            q_init_ub[joint_oth] = q_init_oth

            OCP_store_ic = [vel_sel+1+joint_sel, ran1, ran2, q_init_oth]

            # Guess:
            x_sol_guess = np.empty((N, 5))
            u_sol_guess = np.empty((N, 2))
            for i, tau in enumerate(np.linspace(0, 1, N)):
                x_guess = np.array([q_init_oth, q_init_oth, 0., 0., dt_sym])
                x_guess[joint_sel] = (1-tau)*q_init_sel + tau*q_fin_sel
                x_guess[joint_sel+2] = 2*(1-tau)*(q_fin_sel-q_init_sel) 
                x_sol_guess[i] = x_guess
                u_sol_guess[i] = np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x_guess[0]),ocp.g*ocp.l2*ocp.m2*math.sin(x_guess[1])])

            cost = 1e6

    if all_ok: 

        # Save the optimal trajectory:
        x_sol = np.empty((N+1,5))
        u_sol = np.empty((N,2))
        for i in range(N):
            x_sol[i] = ocp.ocp_solver.get(i, "x")
            u_sol[i] = ocp.ocp_solver.get(i, "u")
        x_sol[N] = ocp.ocp_solver.get(N, "x")

        # # Minimum time OCP:
        # if min_time:
        #     # Reset current iterate:
        #     ocp.ocp_solver.reset()

        #     # Cost:
        #     p_mintime = np.array([0., 0., 1.]) # p[2] = 1 corresponds to the minimization of time

        #     # Set parameters, guesses and constraints:
        #     for i in range(N):
        #         ocp.ocp_solver.set(i, 'x', x_sol[i])
        #         ocp.ocp_solver.set(i, 'u', u_sol[i])
        #         ocp.ocp_solver.set(i, 'p', p_mintime)
        #         ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 0.])) 
        #         ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

        #     # Initial and final states are fixed (except for the time step duration):
        #     ocp.ocp_solver.constraints_set(N, "lbx", np.array([q_min, q_min, 0., 0., 0.]))
        #     ocp.ocp_solver.constraints_set(N, "ubx", np.array([q_max, q_max, 0., 0., dt_sym]))
        #     ocp.ocp_solver.set(N, 'x', x_sol[N])
        #     ocp.ocp_solver.set(N, 'p', p_mintime)

        #     ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x_sol[0][0], x_sol[0][1], x_sol[0][2], x_sol[0][3], 0.])) 
        #     ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x_sol[0][0], x_sol[0][1], x_sol[0][2], x_sol[0][3], dt_sym])) 
        #     ocp.ocp_solver.constraints_set(0, "C", np.array([[0., 0., 0., 0., 0.]])) 

        #     # Solve the OCP:
        #     status = ocp.ocp_solver.solve()

        #     if status == 0:
        #         # Save the new optimal trajectory:
        #         for i in range(N):
        #             x_sol[i] = ocp.ocp_solver.get(i, "x")
        #             u_sol[i] = ocp.ocp_solver.get(i, "u")
        #         x_sol[N] = ocp.ocp_solver.get(N, "x")

        #         # Save the optimized time step duration:
        #         dt_sym = ocp.ocp_solver.get(0, "x")[4]

        x_sym = np.full((N+1,4), None)

        # Generate the unviable sample in the cost direction:
        x_out = np.copy(x_sol[0][:4])
        x_out[2] = x_out[2] - eps * p[0]
        x_out[3] = x_out[3] - eps * p[1]

        # save the initial state:
        valid_data = np.append(valid_data, [[x_sol[0][0], x_sol[0][1], x_sol[0][2], x_sol[0][3]]], axis = 0)

        # Check if initial velocities lie on a limit:
        if x_out[2] > v_max or x_out[2] < v_min or x_out[3] > v_max or x_out[3] < v_min:
            is_x_at_limit = True # the state is on dX
        else:
            is_x_at_limit = False # the state is on dV
            x_sym[0] = x_out

        # Iterate through the trajectory to verify the location of the states with respect to V:
        for f in range(1, N):

            if is_x_at_limit:

                # Generate the unviable sample in the cost direction:
                x_out = np.copy(x_sol[f][:4])
                norm_vel = norm(x_out[2:])    
                x_out[2] = x_out[2] + eps * x_out[2]/norm_vel
                x_out[3] = x_out[3] + eps * x_out[3]/norm_vel

                # If the previous state was on a limit, the current state location cannot be identified using
                # the corresponding unviable state but it has to rely on the proximity to the state limits 
                # (more restrictive):
                if x_sol[f][0] > q_max - eps or x_sol[f][0] < q_min + eps or x_sol[f][1] > q_max - eps or x_sol[f][1] < q_min + eps or x_out[2] > v_max or x_out[2] < v_min or x_out[3] > v_max or x_out[3] < v_min:
                    is_x_at_limit = True # the state is on dX
                else:
                    is_x_at_limit = False # the state is either on the interior of V or on dV

                    # if the traj detouches from a position limit it usually enters V:
                    if x_sol[f-1][0] > q_max - eps or x_sol[f-1][0] < q_min + eps or x_sol[f-1][1] > q_max - eps or x_sol[f-1][1] < q_min + eps:
                        break

                    # Solve an OCP to verify whether the following part of the trajectory is on V or dV. To do so
                    # the initial joint positions are set to the current ones and the final state is fixed to the
                    # final state of the trajectory. The initial velocities are left free and maximized in the 
                    # direction of the current joint velocities.

                    # Reset the number of steps used in the OCP:
                    N_test = N - f
                    ocp.N = N_test
                    ocp.ocp_solver.set_new_time_steps(np.full((N_test,), 1.))
                    ocp.ocp_solver.update_qp_solver_cond_N(N_test)

                    # Cost:
                    norm_weights = norm(np.array([x_sol[f][2], x_sol[f][3]]))    
                    p = np.array([-x_sol[f][2]/norm_weights, -x_sol[f][3]/norm_weights, 0.]) # the cost direction is based on the current velocity direction

                    # Bounds on the initial state:
                    q_init_lb = np.array([x_sol[f][0], x_sol[f][1], v_min, v_min, dt_sym])
                    q_init_ub = np.array([x_sol[f][0], x_sol[f][1], v_max, v_max, dt_sym])

                    # Guess:
                    x_sol_guess = np.empty((N_test+1, 5))
                    u_sol_guess = np.empty((N_test+1, 2))
                    for i in range(N_test):
                        x_sol_guess[i] = x_sol[i+f]
                        u_sol_guess[i] = u_sol[i+f]
                    x_sol_guess[N_test] = x_sol[N]
                    u_sol_guess[N_test] = np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x_sol[N][0]),ocp.g*ocp.l2*ocp.m2*math.sin(x_sol[N][1])])

                    norm_old = norm(np.array([x_sol[f][2:4]])) # velocity norm of the original solution 

                    norm_prev = 0
                    all_ok = False

                    for _ in range(5):

                        # Solve the OCP:
                        status = ocp.OCP_solve(x_sol_guess, u_sol_guess, p, q_lb, q_ub, u_lb, u_ub, q_init_lb, q_init_ub, q_fin_lb, q_fin_ub)

                        if status == 0: # the solver has found a solution

                            # Compare the current cost with the previous one:
                            x0_new = ocp.ocp_solver.get(0, "x")
                            norm_new = norm(np.array([x0_new[2:4]]))

                            if norm_new < norm_prev + tol:
                                all_ok = True 
                                break

                            norm_prev = norm_new

                            # Update the guess with the current solution:
                            x_sol_guess = np.empty((N_test+1,5))
                            u_sol_guess = np.empty((N_test+1,2))
                            for i in range(N_test):
                                x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
                                u_sol_guess[i] = ocp.ocp_solver.get(i, "u")
                            x_sol_guess[N_test] = ocp.ocp_solver.get(N_test, "x")
                            u_sol_guess[N_test] = np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x_sol_guess[N_test][0]),ocp.g*ocp.l2*ocp.m2*math.sin(x_sol_guess[N_test][1])])

                            # Increase the number of time steps:
                            N_test = N_test + 1
                            ocp.N = N_test
                            ocp.ocp_solver.set_new_time_steps(np.full((N_test,), 1.))
                            ocp.ocp_solver.update_qp_solver_cond_N(N_test)

                        else:
                            break

                    if all_ok:

                        # Compare the old and new velocity norms:  
                        if norm_new > norm_old + tol: # the state was inside V

                            # Update the optimal solution:
                            for i in range(N-f):
                                x_sol[i+f] = ocp.ocp_solver.get(i, "x")
                                u_sol[i+f] = ocp.ocp_solver.get(i, "u")

                            # Generate the unviable sample in the cost direction:
                            x_out = np.copy(x_sol[f][:4])
                            x_out[2] = x_out[2] + eps * x_out[2]/norm_new
                            x_out[3] = x_out[3] + eps * x_out[3]/norm_new

                            # Check if velocities lie on a limit:
                            if x_out[2] > v_max or x_out[2] < v_min or x_out[3] > v_max or x_out[3] < v_min:
                                is_x_at_limit = True # the state is on dX
                            else:
                                is_x_at_limit = False # the state is on dV
                                x_sym[f] = x_out

                        else:

                            is_x_at_limit = False # the state is on dV

                            # Generate the new corresponding unviable state in the cost direction:
                            x_out = np.copy(x_sol[f][:4])
                            x_out[2] = x_out[2] - eps * p[0]
                            x_out[3] = x_out[3] - eps * p[1]
                            if x_out[joint_sel+2] > v_max:
                                x_out[joint_sel+2] = v_max
                            if x_out[joint_sel+2] < v_min:
                                x_out[joint_sel+2] = v_min
                            x_sym[f] = x_out

                    else: # we cannot say whether the state is on dV or inside V

                        for r in range(f, N):
                            if abs(x_sol[r][2]) > v_max - eps or abs(x_sol[r][3]) > v_max - eps:

                                # Save the viable states at velocity limits:
                                valid_data = np.append(valid_data, [[x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3]]], axis = 0)

                        break
     
            else:

                # If the previous state was not on a limit, the current state location can be identified using
                # the corresponding unviable state which can be computed by simulating the system starting from 
                # the previous unviable state.

                # Simulate next unviable state:
                u_sym = np.copy(u_sol[f-1])
                sim.acados_integrator.set("u", u_sym)
                sim.acados_integrator.set("x", x_sym[f-1])
                status = sim.acados_integrator.solve()
                x_out = sim.acados_integrator.get("x")
                x_sym[f] = x_out

                # When the state of the unviable simulated trajectory violates a limit, the corresponding viable state
                # of the optimal trajectory is on dX:
                if x_out[0] > q_max or x_out[0] < q_min or x_out[1] > q_max or x_out[1] < q_min or x_out[2] > v_max or x_out[2] < v_min or x_out[3] > v_max or x_out[3] < v_min:
                    is_x_at_limit = True # the state is on dX
                else:
                    is_x_at_limit = False # the state is on dV

            if x_sol[f][0] < q_max - eps and x_sol[f][0] > q_min + eps and x_sol[f][1] < q_max - eps and x_sol[f][1] > q_min + eps and abs(x_sol[f][2]) > tol and abs(x_sol[f][3]) > tol:
                
                # Save the viable and unviable states:
                valid_data = np.append(valid_data, [[x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3]]], axis = 0)

        return  valid_data.tolist(), OCP_store_ic, None

    else:
        return None, None, OCP_store_ic

start_time = time.time()

X_test = np.load('../data2_test.npy')

# Ocp initialization:
ocp = OCPdoublependulumINIT()
sim = SYMdoublependulumINIT()

# Position, velocity and torque bounds:
v_max = ocp.dthetamax
v_min = - ocp.dthetamax
q_max = ocp.thetamax
q_min = ocp.thetamin
tau_max = ocp.Cmax

stop_time = 3600 # total computational time
dt_sym = 1e-2 # time step duration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pytorch device

tol = ocp.ocp.solver_options.nlp_solver_tol_stat # OCP cost tolerance
eps = tol*10 # unviable data generation parameter

iteration = 0 
print('Start data generation, iteration:', iteration)

# Data generation:
cpu_num = 30
num_prob = 1000
with Pool(cpu_num) as p:
    temp = p.map(data_generation, range(num_prob))

traj, statpos, statneg = zip(*temp)
X_save = [i for i in traj if i is not None]
X_save = np.array([i for f in X_save for i in f])

print('Data generation completed')

# Pytorch params:
input_layers = 4
hidden_layers = 300
output_layers = 1
learning_rate = 1e-3

# Model and optimizer:
model_dir = NeuralNetDIR(input_layers, hidden_layers, output_layers).to(device)
criterion_dir = nn.MSELoss()
optimizer_dir = torch.optim.Adam(model_dir.parameters(), lr=learning_rate)

# Joint positions mean and variance:
mean_dir, std_dir = torch.mean(torch.tensor(X_save[:,:2].tolist())).to(device).item(), torch.std(torch.tensor(X_save[:,:2].tolist())).to(device).item()
torch.save(mean_dir, 'mean_2dof_vboc')
torch.save(std_dir, 'std_2dof_vboc')

# Rewrite data in the form [normalized positions, velocity direction, velocity norm]:
X_train_dir = np.empty((X_save.shape[0],5))
for i in range(X_train_dir.shape[0]):
    X_train_dir[i][0] = (X_save[i][0] - mean_dir) / std_dir
    X_train_dir[i][1] = (X_save[i][1] - mean_dir) / std_dir
    vel_norm = norm([X_save[i][2], X_save[i][3]])
    if vel_norm != 0:
        X_train_dir[i][2] = X_save[i][2] / vel_norm
        X_train_dir[i][3] = X_save[i][3] / vel_norm
    X_train_dir[i][4] = vel_norm 

beta = 0.95
n_minibatch = 4096
B = int(X_save.shape[0]*100/n_minibatch) # number of iterations for 100 epoch
it_max = B * 10

print('Start model training')

it = 1
val = max(X_train_dir[:,4])

# Train the model:
while val > 1e-3 and it < it_max:
    ind = random.sample(range(len(X_train_dir)), n_minibatch)

    X_iter_tensor = torch.Tensor([X_train_dir[i][:4] for i in ind]).to(device)
    y_iter_tensor = torch.Tensor([[X_train_dir[i][4]] for i in ind]).to(device)

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

print('Model training completed')

# Store time:
times = np.array([time.time() - start_time])

# Store resulting RMSE wrt testing data:
X_test_dir = np.empty((X_test.shape[0],5))
for i in range(X_test_dir.shape[0]):
    X_test_dir[i][0] = (X_test[i][0] - mean_dir) / std_dir
    X_test_dir[i][1] = (X_test[i][1] - mean_dir) / std_dir
    vel_norm = norm([X_test[i][2],X_test[i][3]])
    if vel_norm != 0:
        X_test_dir[i][2] = X_test[i][2] / vel_norm
        X_test_dir[i][3] = X_test[i][3] / vel_norm
    X_test_dir[i][4] = vel_norm 

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_test_dir[:,:4]).to(device)
    y_iter_tensor = torch.Tensor(X_test_dir[:,4:]).to(device)
    rmse = np.array([torch.sqrt(criterion_dir(model_dir(X_iter_tensor), y_iter_tensor)).item()])

while time.time() - start_time < stop_time:

    iteration += 1
    print('Start data generation, iteration:', iteration)

    # Data generation:
    with Pool(cpu_num) as p:
        temp = p.map(data_generation, range(num_prob))

    traj, statpos, statneg = zip(*temp)
    samples = [i for i in traj if i is not None]

    # Add new training data to the dataset:
    X_new = np.array([i for f in samples for i in f])
    X_save = np.concatenate((X_save, X_new))

    print('Data generation completed')

    # Rewrite data in the form [normalized positions, velocity direction, velocity norm]:
    X_new_dir = np.empty((X_new.shape[0],5))
    for i in range(X_new_dir.shape[0]):
        X_new_dir[i][0] = (X_new[i][0] - mean_dir) / std_dir
        X_new_dir[i][1] = (X_new[i][1] - mean_dir) / std_dir
        vel_norm = norm([X_new[i][2], X_new[i][3]])
        if vel_norm != 0:
            X_new_dir[i][2] = X_new[i][2] / vel_norm
            X_new_dir[i][3] = X_new[i][3] / vel_norm
        X_new_dir[i][4] = vel_norm 
    X_train_dir = np.concatenate((X_train_dir, X_new_dir))

    print('Start model training')

    it = 1
    val = max(X_train_dir[:,4])

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

        X_iter_tensor = torch.Tensor([X_train_dir[i][:4] for i in ind]).to(device)
        y_iter_tensor = torch.Tensor([[X_train_dir[i][4]] for i in ind]).to(device)

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

    print('Model training completed')

    # Store time:
    times = np.append(times, [time.time() - start_time])

    # Store resulting RMSE wrt testing data:
    with torch.no_grad():
        X_iter_tensor = torch.Tensor(X_test_dir[:,:4]).to(device)
        y_iter_tensor = torch.Tensor(X_test_dir[:,4:]).to(device)
        rmse = np.append(rmse, [torch.sqrt(criterion_dir(model_dir(X_iter_tensor), y_iter_tensor)).item()])

# Show total time and resulting RMSE:
print('Iterations completed with total time:', time.time() - start_time)
print('RMSE test data: ', torch.sqrt(criterion_dir(model_dir(X_iter_tensor), y_iter_tensor)).item()) 

# Store times and RMSE:
np.save('times_2dof_vboc.npy', np.asarray(times))
np.save('rmse_2dof_vboc.npy', np.asarray(rmse))

# Show the RMSE evolution over time:
plt.figure()
plt.plot(times, rmse)

# Save the training data and the model:
np.save('data_2dof_vboc.npy', np.asarray(X_save))
torch.save(model_dir.state_dict(), 'model_2dof_vboc.npy')

# Show the resulting set approximation:
with torch.no_grad():
    h = 0.01
    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max+h, h))
    xrav = xx.ravel()
    yrav = yy.ravel()

    # Plot the results:
    plt.figure()
    inp = np.c_[
                (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                xrav,
                np.zeros(yrav.shape[0]),
                yrav,
                np.empty(yrav.shape[0]),
            ]
    for i in range(inp.shape[0]):
        vel_norm = norm([inp[i][2],inp[i][3]])
        inp[i][0] = (inp[i][0] - mean_dir) / std_dir
        inp[i][1] = (inp[i][1] - mean_dir) / std_dir
        if vel_norm != 0:
            inp[i][2] = inp[i][2] / vel_norm
            inp[i][3] = inp[i][3] / vel_norm
        inp[i][4] = vel_norm
    out = (model_dir(torch.from_numpy(inp[:,:4].astype(np.float32)).to(device))).cpu().numpy() 
    y_pred = np.empty(out.shape)
    for i in range(len(out)):
        if inp[i][4] > out[i]:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # for i in range(X_test.shape[0]):
    #     if (
    #         norm(X_test[i][0] - (q_min + q_max) / 2) < 0.01
    #         and norm(X_test[i][2]) < 0.1
    #     ):
    #         xit.append(X_test[i][1])
    #         yit.append(X_test[i][3])
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

    plt.figure()
    inp = np.c_[
                xrav,
                (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                yrav,
                np.zeros(yrav.shape[0]),
                np.empty(yrav.shape[0]),
            ]
    for i in range(inp.shape[0]):
        vel_norm = norm([inp[i][2],inp[i][3]])
        inp[i][0] = (inp[i][0] - mean_dir) / std_dir
        inp[i][1] = (inp[i][1] - mean_dir) / std_dir
        if vel_norm != 0:
            inp[i][2] = inp[i][2] / vel_norm
            inp[i][3] = inp[i][3] / vel_norm
        inp[i][4] = vel_norm
    out = (model_dir(torch.from_numpy(inp[:,:4].astype(np.float32)).to(device))).cpu().numpy() 
    y_pred = np.empty(out.shape)
    for i in range(len(out)):
        if inp[i][4] > out[i]:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # for i in range(X_test.shape[0]):
    #     if (
    #         norm(X_test[i][1] - (q_min + q_max) / 2) < 0.01
    #         and norm(X_test[i][3]) < 0.1
    #     ):
    #         xit.append(X_test[i][0])
    #         yit.append(X_test[i][2])
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

    # for l in range(10):
    #     q1ran = q_min + random.random() * (q_max-q_min)
    #     q2ran = q_min + random.random() * (q_max-q_min)

    #     # Plot the results:
    #     plt.figure()
    #     inp = np.float32(
    #             np.c_[
    #                 q1ran * np.ones(xrav.shape[0]),
    #                 q2ran * np.ones(xrav.shape[0]),
    #                 xrav,
    #                 yrav,
    #                 np.empty(yrav.shape[0]),
    #             ]
    #         )
    #     for i in range(inp.shape[0]):
    #         vel_norm = norm([inp[i][2],inp[i][3]])
    #         inp[i][0] = (inp[i][0] - mean_dir) / std_dir
    #         inp[i][1] = (inp[i][1] - mean_dir) / std_dir
    #         if vel_norm != 0:
    #             inp[i][2] = inp[i][2] / vel_norm
    #             inp[i][3] = inp[i][3] / vel_norm
    #         inp[i][4] = vel_norm
    #     out = (model_dir(torch.Tensor(inp[:,:4]).to(device))).cpu().numpy() 
    #     y_pred = np.empty(out.shape)
    #     for i in range(len(out)):
    #         if inp[i][4] > out[i]:
    #             y_pred[i] = 0
    #         else:
    #             y_pred[i] = 1
    #     Z = y_pred.reshape(yy.shape)
    #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #     xit = []
    #     yit = []
    #     for i in range(X_save.shape[0]):
    #         if (
    #             norm(X_save[i][0] - q1ran) < 0.01
    #             and norm(X_save[i][1] - q2ran) < 0.01
    #         ):
    #             xit.append(X_save[i][2])
    #             yit.append(X_save[i][3])
    #     plt.plot(
    #         xit,
    #         yit,
    #         "ko",
    #         markersize=2
    #     )
    #     plt.xlim([v_min, v_max])
    #     plt.ylim([v_min, v_max])
    #     plt.grid()
    #     plt.title("q1="+str(q1ran)+" q2="+str(q2ran))

plt.show()
