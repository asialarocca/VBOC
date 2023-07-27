import numpy as np
import random 
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
import time
from ur5reduced_class_fixedveldir import OCPUR5INIT, SYMUR5INIT
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from my_nn import NeuralNetDIR
from multiprocessing import Pool
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
from plots_2dof import plots_2dof
from plots_3dof import plots_3dof
from plots_4dof import plots_4dof

def testing(v):

    valid_data = np.ndarray((0, ocp.ocp.dims.nx))

    # Reset the time parameters:
    N = N_start 
    ocp.N = N
    ocp.ocp_solver.set_new_time_steps(np.full((N,), dt_sym))
    ocp.ocp_solver.update_qp_solver_cond_N(N)

    # Selection of the reference joint:
    joint_sel = random.choice(range(ocp.ocp.dims.nu))

    # Selection of the start position of the reference joint:
    vel_sel = random.choice([-1, 1]) # -1 to maximise initial vel, + 1 to minimize it

    # Initial velocity optimization direction:
    p = np.empty((ocp.ocp.dims.nu,))

    for l in range(ocp.ocp.dims.nu):
        if l == joint_sel:
            p[l] = random.random() * vel_sel
        else:
            p[l] = random.random() * random.choice([-1, 1])

    norm_weights = norm(p)        
    p = p/norm_weights

    # Bounds on the initial state:
    q_init_lb = np.copy(ocp.xmin)
    q_init_ub = np.copy(ocp.xmax)

    q_init_oth = np.empty((ocp.ocp.dims.nu,))

    for l in range(ocp.ocp.dims.nu):

        if l == joint_sel:

            if vel_sel == -1:
                q_init_oth[l] = x_min[l] + eps
                q_fin_sel = x_max[l] - eps
            else:
                q_init_oth[l] = x_max[l] - eps
                q_fin_sel = x_min[l] + eps

            q_init_lb[l] = q_init_oth[l]
            q_init_ub[l] = q_init_oth[l]

        else:
            q_init_oth[l] = x_min[l] + random.random() * (x_max[l]-x_min[l])

            if q_init_oth[l] > x_max[l] - eps:
                q_init_oth[l] = q_init_oth[l] - eps
            if q_init_oth[l]  < x_min[l] + eps:
                q_init_oth[l] = q_init_oth[l] + eps

            q_init_lb[l] = q_init_oth[l]
            q_init_ub[l] = q_init_oth[l]

    # State and input bounds:
    q_lb = np.copy(ocp.xmin)
    q_ub = np.copy(ocp.xmax)
    u_lb = np.copy(ocp.Cmin)
    u_ub = np.copy(ocp.Cmax)

    # Bounds on the final state:
    q_fin_lb = np.copy(ocp.xmin)
    q_fin_ub = np.copy(ocp.xmax)

    for l in range(ocp.ocp.dims.nu):
        q_fin_lb[l+ocp.ocp.dims.nu] = 0.
        q_fin_ub[l+ocp.ocp.dims.nu] = 0.

    # Guess:
    x_sol_guess = np.empty((N, ocp.ocp.dims.nx))
    u_sol_guess = np.zeros((N, ocp.ocp.dims.nu))

    for i, tau in enumerate(np.linspace(0, 1, N)):

        x_guess = np.zeros((ocp.ocp.dims.nx,))

        for l in range(ocp.ocp.dims.nu):
            if l == joint_sel:
                x_guess[l] = (1-tau)*q_init_oth[l] + tau*q_fin_sel
                x_guess[l+ocp.ocp.dims.nu] = 2*(1-tau)*(q_fin_sel-q_init_oth[l]) 
            else:
                x_guess[l] = q_init_oth[l]

        x_sol_guess[i] = x_guess

    cost_old = 1e6
    all_ok = False

    while True:

        # Solve the OCP:
        status = ocp.OCP_solve(x_sol_guess, u_sol_guess, p, q_lb, q_ub, u_lb, u_ub, q_init_lb, q_init_ub, q_fin_lb, q_fin_ub)

        if status == 0: # the solver has found a solution

            # Compare the current cost with the previous one:
            cost_new = ocp.ocp_solver.get_cost()

            if cost_new > cost_old - tol: # the time is sufficient to have achieved an optimal solution
                all_ok = True
                break

            cost_old = cost_new

            # Update the guess with the current solution:
            x_sol_guess = np.empty((N+1,ocp.ocp.dims.nx))
            u_sol_guess = np.empty((N+1,ocp.ocp.dims.nu))
            for i in range(N):
                x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
                u_sol_guess[i] = ocp.ocp_solver.get(i, "u")
            x_sol_guess[N] = ocp.ocp_solver.get(N, "x")
            u_sol_guess[N] = np.zeros((ocp.ocp.dims.nu,)) # ocp.get_inverse_dynamics(x_sol_guess[N,:ocp.ocp.dims.nu],x_sol_guess[N,ocp.ocp.dims.nu:])

            # Increase the number of time steps:
            N = N + 1
            ocp.N = N
            ocp.ocp_solver.set_new_time_steps(np.full((N,), dt_sym))
            ocp.ocp_solver.update_qp_solver_cond_N(N)

        else:

            N = N_start 
            ocp.N = N
            ocp.ocp_solver.set_new_time_steps(np.full((N,), dt_sym))
            ocp.ocp_solver.update_qp_solver_cond_N(N)

            for i in range(ocp.ocp.dims.nu):

                if i != joint_sel:

                    p[i] = p[i] + random.random() * random.choice([-1, 1]) * 0.01
                    q_init = q_init_lb[i] + random.random() * random.choice([-1, 1]) * 0.01
                    q_init_lb[i] = q_init
                    q_init_ub[i] = q_init

            norm_weights = norm(p)         
            p = p/norm_weights

            # Guess:
            x_sol_guess = np.empty((N, ocp.ocp.dims.nx))
            u_sol_guess = np.zeros((N, ocp.ocp.dims.nu))

            for i, tau in enumerate(np.linspace(0, 1, N)):

                x_guess = np.zeros((ocp.ocp.dims.nx,))

                for l in range(ocp.ocp.dims.nu):
                    if l == joint_sel:
                        x_guess[l] = (1-tau)*q_init_oth[l] + tau*q_fin_sel
                        x_guess[l+ocp.ocp.dims.nu] = 2*(1-tau)*(q_fin_sel-q_init_oth[l]) 
                    else:
                        x_guess[l] = q_init_oth[l]

                x_sol_guess[i] = x_guess

            cost_old = 1e6

    if all_ok: 

        # Save the optimal trajectory:
        x_sol = np.empty((N+1, ocp.ocp.dims.nx))
        u_sol = np.empty((N, ocp.ocp.dims.nu))

        for i in range(N):
            x_sol[i] = ocp.ocp_solver.get(i, "x")
            u_sol[i] = ocp.ocp_solver.get(i, "u")

        x_sol[N] = ocp.ocp_solver.get(N, "x")

        # Generate the unviable sample in the cost direction:
        x_sym = np.full((N+1,ocp.ocp.dims.nx), None)

        x_out = np.copy(x_sol[0])
        for l in range(ocp.ocp.dims.nu):
            x_out[l+ocp.ocp.dims.nu] = x_out[l+ocp.ocp.dims.nu] - eps * p[l]

        # save the initial state:
        valid_data = np.append(valid_data, [x_sol[0]], axis = 0)

        # Check if initial velocities lie on a limit:
        if any(x_out[i] > x_max[i] or x_out[i] < x_min[i] for i in range(ocp.ocp.dims.nu,ocp.ocp.dims.nx)):
            is_x_at_limit = True # the state is on dX
        else:
            is_x_at_limit = False # the state is on dV
            x_sym[0] = x_out

        # Iterate through the trajectory to verify the location of the states with respect to V:
        for f in range(1, N):

            if is_x_at_limit:

                # If the previous state was on a limit, the current state location cannot be identified using
                # the corresponding unviable state but it has to rely on the proximity to the state limits 
                # (more restrictive):
                if any(x_sol[f][i] > x_max[i] - eps or x_sol[f][i] < x_min[i] + eps for i in range(ocp.ocp.dims.nu)) or any(x_sol[f][i] > x_max[i] - tol or x_sol[f][i] < x_min[i] + tol for i in range(ocp.ocp.dims.nu,ocp.ocp.dims.nx)):
                    is_x_at_limit = True # the state is on dX
                
                else:
                    is_x_at_limit = False # the state is either on the interior of V or on dV

                    # if the traj detouches from a position limit it usually enters V:
                    if any(x_sol[f-1][i] > x_max[i] - eps or x_sol[f-1][i] < x_min[i] + eps for i in range(ocp.ocp.dims.nu)):
                        break

                    # Solve an OCP to verify whether the following part of the trajectory is on V or dV. To do so
                    # the initial joint positions are set to the current ones and the final state is fixed to the
                    # final state of the trajectory. The initial velocities are left free and maximized in the 
                    # direction of the current joint velocities.

                    N_test = N - f
                    ocp.N = N_test
                    ocp.ocp_solver.set_new_time_steps(np.full((N_test,), dt_sym))
                    ocp.ocp_solver.update_qp_solver_cond_N(N_test)

                    # Cost: 
                    norm_weights = norm(x_sol[f][ocp.ocp.dims.nu:])    
                    p = np.zeros((ocp.ocp.dims.nu))
                    for l in range(ocp.ocp.dims.nu):
                        p[l] = -x_sol[f][l+ocp.ocp.dims.nu]/norm_weights # the cost direction is based on the current velocity direction

                    # Bounds on the initial state:
                    for l in range(ocp.ocp.dims.nu):
                        q_init_lb[l] = x_sol[f][l]
                        q_init_ub[l] = x_sol[f][l]

                    # Guess:
                    x_sol_guess = np.empty((N_test+1, ocp.ocp.dims.nx))
                    u_sol_guess = np.empty((N_test+1, ocp.ocp.dims.nu))
                    for i in range(N_test):
                        x_sol_guess[i] = x_sol[i+f]
                        u_sol_guess[i] = u_sol[i+f]
                    x_sol_guess[N_test] = x_sol[N]
                    u_sol_guess[N_test] = np.zeros((ocp.ocp.dims.nu))

                    norm_old = norm_weights # velocity norm of the original solution 
                    norm_bef = 0
                    all_ok = False

                    while True:
                        
                        # Solve the OCP:
                        status = ocp.OCP_solve(x_sol_guess, u_sol_guess, p, q_lb, q_ub, u_lb, u_ub, q_init_lb, q_init_ub, q_fin_lb, q_fin_ub)

                        if status == 0: # the solver has found a solution

                            # Compare the current cost with the previous one:
                            x0_new = ocp.ocp_solver.get(0, "x")
                            norm_new = norm(x0_new[ocp.ocp.dims.nu:])

                            if norm_new < norm_bef + tol:
                                all_ok = True 
                                break

                            norm_bef = norm_new

                            # Update the guess with the current solution:
                            x_sol_guess = np.empty((N_test+1,ocp.ocp.dims.nx))
                            u_sol_guess = np.empty((N_test+1,ocp.ocp.dims.nu))
                            for i in range(N_test):
                                x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
                                u_sol_guess[i] = ocp.ocp_solver.get(i, "u")
                            x_sol_guess[N_test] = ocp.ocp_solver.get(N_test, "x")
                            u_sol_guess[N_test] = np.zeros((ocp.ocp.dims.nu))

                            # Increase the number of time steps:
                            N_test = N_test + 1
                            ocp.N = N_test
                            ocp.ocp_solver.set_new_time_steps(np.full((N_test,), dt_sym))
                            ocp.ocp_solver.update_qp_solver_cond_N(N_test)

                        else:
                            break

                    if all_ok:

                        # Compare the old and new velocity norms:  
                        if norm_new > norm_old + tol: # the state is inside V

                            # Update the optimal solution:
                            for i in range(N-f):
                                x_sol[i+f] = ocp.ocp_solver.get(i, "x")
                                u_sol[i+f] = ocp.ocp_solver.get(i, "u")

                            x_out = np.copy(x_sol[f])

                            for l in range(ocp.ocp.dims.nu):
                                x_out[l+ocp.ocp.dims.nu] = x_out[l+ocp.ocp.dims.nu] + eps * p[l]

                            # Check if velocities lie on a limit:
                            if any(x_out[i] > x_max[i] or i < x_min[i] for i in range(ocp.ocp.dims.nu,ocp.ocp.dims.nx)):
                                is_x_at_limit = True # the state is on dX
                            else:
                                is_x_at_limit = False # the state is on dV
                                x_sym[f] = x_out
                                
                        else:
                            is_x_at_limit = False # the state is on dV

                            # Generate the new corresponding unviable state in the cost direction:
                            x_out = np.copy(x_sol[f])
                            
                            for l in range(ocp.ocp.dims.nu):
                                x_out[l+ocp.ocp.dims.nu] = x_out[l+ocp.ocp.dims.nu] - eps * p[l]

                            x_sym[f] = x_out

                    else: # we cannot say whether the state is on dV or inside V

                        for r in range(f, N):

                            if any(x_sol[r][i] > x_max[i] - eps or x_sol[r][i] < x_min[i] + eps for i in range(ocp.ocp.dims.nu,ocp.ocp.dims.nx)):
                                                                    
                                # Save the viable states at velocity limits:
                                valid_data = np.append(valid_data, [x_sol[r]], axis = 0)

                        break      
     
            else:

                # Simulate next unviable state:
                u_sym = np.copy(u_sol[f-1])
                sim.acados_integrator.set("u", u_sym)
                sim.acados_integrator.set("x", x_sym[f-1])
                # sim.acados_integrator.set("T", dt_sym)
                status = sim.acados_integrator.solve()
                x_out = sim.acados_integrator.get("x")

                # When the state of the unviable simulated trajectory violates a limit, the corresponding viable state
                # of the optimal trajectory is on dX:
                if any(x_out[i] > x_max[i] or x_out[i] < x_min[i] for i in range(ocp.ocp.dims.nx)):
                    is_x_at_limit = True # the state is on dX
                else:
                    is_x_at_limit = False # the state is on dV
                    x_sym[f] = x_out

            if all(x_sol[f][i] < x_max[i] - eps and x_sol[f][i] > x_min[i] + eps for i in range(ocp.ocp.dims.nu)) and all(abs(x_sol[f][i]) > tol for i in range(ocp.ocp.dims.nu,ocp.ocp.dims.nx)):
                
                # Save the viable and unviable states:
                valid_data = np.append(valid_data, [x_sol[f]], axis = 0)

        return  valid_data.tolist()
    
    else:
        return None
    
def testing_test(v):

    # Reset the time parameters:
    N = N_start 
    ocp.N = N
    ocp.ocp_solver.set_new_time_steps(np.full((N,), dt_sym))
    ocp.ocp_solver.update_qp_solver_cond_N(N)

    # Initial velocity optimization direction:
    p = np.empty((ocp.ocp.dims.nu,))

    for l in range(ocp.ocp.dims.nu):
        p[l] = random.random() * random.choice([-1, 1])

    norm_weights = norm(p)        
    p = p/norm_weights

    # Bounds on the initial state:
    q_init_lb = np.copy(ocp.xmin)
    q_init_ub = np.copy(ocp.xmax)

    q_init_oth = np.empty((ocp.ocp.dims.nu,))

    for l in range(ocp.ocp.dims.nu):

        q_init_oth[l] = x_min[l] + random.random() * (x_max[l]-x_min[l])

        q_init_lb[l] = q_init_oth[l]
        q_init_ub[l] = q_init_oth[l]

    # State and input bounds:
    q_lb = np.copy(ocp.xmin)
    q_ub = np.copy(ocp.xmax)
    u_lb = np.copy(ocp.Cmin)
    u_ub = np.copy(ocp.Cmax)

    # Bounds on the final state:
    q_fin_lb = np.copy(ocp.xmin)
    q_fin_ub = np.copy(ocp.xmax)

    for l in range(ocp.ocp.dims.nu):
        q_fin_lb[l+ocp.ocp.dims.nu] = 0.
        q_fin_ub[l+ocp.ocp.dims.nu] = 0.

    # Guess:
    x_sol_guess = np.empty((N, ocp.ocp.dims.nx))
    u_sol_guess = np.zeros((N, ocp.ocp.dims.nu))

    x_guess = np.zeros((ocp.ocp.dims.nx,))

    for l in range(ocp.ocp.dims.nu):
        x_guess[l] = q_init_oth[l]

    x_sol_guess[:] = x_guess

    cost_old = 1e6
    all_ok = False

    # Iteratively solve the OCP with an increased number of time steps until the solution does not change.
    # If the solver fails, try with a slightly different initial condition:
    # for _ in range(10):
    while True:

        # Solve the OCP:
        status = ocp.OCP_solve(x_sol_guess, u_sol_guess, p, q_lb, q_ub, u_lb, u_ub, q_init_lb, q_init_ub, q_fin_lb, q_fin_ub)

        if status == 0: # the solver has found a solution

            # Compare the current cost with the previous one:
            cost_new = ocp.ocp_solver.get_cost()

            if cost_new > cost_old - tol: # the time is sufficient to have achieved an optimal solution
                all_ok = True
                break

            cost_old = cost_new

            # Update the guess with the current solution:
            x_sol_guess = np.empty((N+1,ocp.ocp.dims.nx))
            u_sol_guess = np.empty((N+1,ocp.ocp.dims.nu))
            for i in range(N):
                x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
                u_sol_guess[i] = ocp.ocp_solver.get(i, "u")
            x_sol_guess[N] = ocp.ocp_solver.get(N, "x")
            u_sol_guess[N] = np.zeros((ocp.ocp.dims.nu,)) # ocp.get_inverse_dynamics(x_sol_guess[N,:ocp.ocp.dims.nu],x_sol_guess[N,ocp.ocp.dims.nu:])

            # Increase the number of time steps:
            N = N + 1
            ocp.N = N
            ocp.ocp_solver.set_new_time_steps(np.full((N,), dt_sym))
            ocp.ocp_solver.update_qp_solver_cond_N(N)

        else:
            break

    if all_ok: 
        return [ocp.ocp_solver.get(0, "x")]
    else:
        return None

start_time = time.time()

# Ocp initialization:
ocp = OCPUR5INIT()
sim = SYMUR5INIT()

# State bounds:
x_max = ocp.xmax
x_min = ocp.xmin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pytorch device

dt_sym = 1e-2 # time step duration
N_start = 100 # initial number of timesteps
tol = ocp.ocp.solver_options.nlp_solver_tol_stat # OCP cost tolerance
eps = tol*10 # unviable data generation parameter

# print('Start test data generation')

# Testing data generation:
cpu_num = 30
num_prob = 10000
with Pool(cpu_num) as p:
    traj = p.map(testing_test, range(num_prob))

X_temp = [i for i in traj if i is not None]
X_test = np.array([i for f in X_temp for i in f])

# Save training data:
np.save('data_' + str(ocp.ocp.dims.nu) + 'dof_vboc_test', np.asarray(X_test))

X_test = np.load('data_' + str(ocp.ocp.dims.nu) + 'dof_vboc_test.npy')
X_old = np.load('data_' + str(ocp.ocp.dims.nu) + 'dof_vboc_train.npy')

print('Start data generation')

time_start = time.time()

# Data generation:
cpu_num = 30
num_prob = 1000000
with Pool(cpu_num) as p:
    traj = p.map(testing_test, range(num_prob))

print(time.time() - time_start)

# traj, statpos, statneg = zip(*temp)
X_temp = [i for i in traj if i is not None]
print('Data generation completed')

# Print data generations statistics:
solved=len(X_temp)
print('Solved/tot', len(X_temp)/num_prob)
X_save = np.array([i for f in X_temp for i in f])
print('Saved/tot', len(X_save)/(solved*100))

X_tot = np.concatenate((X_old,X_save))

# Save training data:
np.save('data_' + str(ocp.ocp.dims.nu) + 'dof_vboc_train', np.asarray(X_tot))

# X_save = np.load('data_' + str(ocp.ocp.dims.nu) + 'dof_vboc_train.npy')

# Pytorch params:
input_layers = ocp.ocp.dims.nx
hidden_layers = 1000 #(input_layers) * 100
output_layers = 1
learning_rate = 1e-3

# Model and optimizer:
model_dir = NeuralNetDIR(input_layers, hidden_layers, output_layers).to(device)
criterion_dir = nn.MSELoss()
optimizer_dir = torch.optim.Adam(model_dir.parameters(), lr=learning_rate)

model_dir.load_state_dict(torch.load('model_' + str(ocp.ocp.dims.nu) + 'dof_vboc'))

# # Joint positions mean and variance:
# mean_dir, std_dir = torch.mean(torch.tensor(X_save[:,:ocp.ocp.dims.nu].tolist())).to(device).item(), torch.std(torch.tensor(X_save[:,:ocp.ocp.dims.nu].tolist())).to(device).item()
# torch.save(mean_dir, 'mean_' + str(ocp.ocp.dims.nu) + 'dof_vboc')
# torch.save(std_dir, 'std_' + str(ocp.ocp.dims.nu) + 'dof_vboc')

mean_dir = torch.load('mean_' + str(ocp.ocp.dims.nu) + 'dof_vboc')
std_dir = torch.load('std_' + str(ocp.ocp.dims.nu) + 'dof_vboc')

# Rewrite data in the form [normalized positions, velocity direction, velocity norm]:
X_train_dir = np.empty((X_save.shape[0],ocp.ocp.dims.nx+1))

for i in range(X_train_dir.shape[0]):
    vel_norm = norm(X_save[i][ocp.ocp.dims.nu:])
    X_train_dir[i][ocp.ocp.dims.nx] = vel_norm 
    for l in range(ocp.ocp.dims.nu):
        X_train_dir[i][l] = (X_save[i][l] - mean_dir) / std_dir
        X_train_dir[i][l+ocp.ocp.dims.nu] = X_save[i][l+ocp.ocp.dims.nu] / vel_norm

beta = 0.95
n_minibatch = pow(2,15)
B = int(X_save.shape[0]*100/n_minibatch) # number of iterations for 100 epoch
it_max = B * 20

print('Start model training')

it = 1
val = max(X_train_dir[:,ocp.ocp.dims.nx])

# Train the model
while val > 1e-3 and it < it_max:
    ind = random.sample(range(len(X_train_dir)), n_minibatch)

    X_iter_tensor = torch.Tensor([X_train_dir[i][:ocp.ocp.dims.nx] for i in ind]).to(device)
    y_iter_tensor = torch.Tensor([[X_train_dir[i][ocp.ocp.dims.nx]] for i in ind]).to(device)

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

# Show the resulting RMSE on the training data:
outputs = np.empty((len(X_train_dir),1))
n_minibatch_model = pow(2,16)
with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_train_dir[:,:ocp.ocp.dims.nx]).to(device)
    y_iter_tensor = torch.Tensor(X_train_dir[:,ocp.ocp.dims.nx:]).to(device)
    my_dataloader = DataLoader(X_iter_tensor,batch_size=n_minibatch_model,shuffle=False)
    for (idx, batch) in enumerate(my_dataloader):
        if n_minibatch_model*(idx+1) > len(X_train_dir):
            outputs[n_minibatch_model*idx:len(X_train_dir)] = model_dir(batch).cpu().numpy()
        else:
            outputs[n_minibatch_model*idx:n_minibatch_model*(idx+1)] = model_dir(batch).cpu().numpy()
    outputs_tensor = torch.Tensor(outputs).to(device)
    print('RMSE train data: ', torch.sqrt(criterion_dir(outputs_tensor, y_iter_tensor))) 

# Compute resulting RMSE wrt testing data:
X_test_dir = np.empty((X_test.shape[0],ocp.ocp.dims.nx+1))
for i in range(X_test_dir.shape[0]):
    vel_norm = norm(X_test[i][ocp.ocp.dims.nu:ocp.ocp.dims.nx])
    X_test_dir[i][ocp.ocp.dims.nx] = vel_norm
    for l in range(ocp.ocp.dims.nu):
        X_test_dir[i][l] = (X_test[i][l] - mean_dir) / std_dir
        X_test_dir[i][l+ocp.ocp.dims.nu] = X_test[i][l+ocp.ocp.dims.nu] / vel_norm

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_test_dir[:,:ocp.ocp.dims.nx]).to(device)
    y_iter_tensor = torch.Tensor(X_test_dir[:,ocp.ocp.dims.nx:]).to(device)
    outputs = model_dir(X_iter_tensor)
    print('RMSE test data: ', torch.sqrt(criterion_dir(outputs, y_iter_tensor)))

# Save the model:
torch.save(model_dir.state_dict(), 'model_' + str(ocp.ocp.dims.nu) + 'dof_vboc')

print("Execution time: %s seconds" % (time.time() - time_start + 143651))

# Show training data and resulting set approximation:
if ocp.ocp.dims.nu == 3:
    plots_3dof(X_save, -3.14, 3.14, -3.15, 3.15, model_dir, mean_dir, std_dir, device)
elif ocp.ocp.dims.nu == 2:
    plots_2dof(X_save, -3.14, 3.14, -3.15, 3.15, model_dir, mean_dir, std_dir, device)
elif ocp.ocp.dims.nu == 4:
    plots_4dof(X_save, -3.14, 3.14, -3.15, 3.15, model_dir, mean_dir, std_dir, device)

plt.show()