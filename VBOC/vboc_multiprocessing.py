import numpy as np
import random 
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
import time
from triplependulum_class_fixedveldir import OCPtriplependulumINIT, SYMtriplependulumINIT
from doublependulum_class_fixedveldir import OCPdoublependulumINIT, SYMdoublependulumINIT
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from my_nn import NeuralNetRegression
from multiprocessing import Pool
from torch.utils.data import DataLoader
from plots_2dof import plots_2dof
from plots_3dof import plots_3dof
import torch.nn.utils.prune as prune

def testing(v):

    valid_data = np.ndarray((0, ocp.ocp.dims.nx - 1))

    # Reset the time parameters:
    N = N_start 
    ocp.N = N
    ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
    ocp.ocp_solver.update_qp_solver_cond_N(N)

    # Initialization of the OCP: The OCP is set to find an extreme trajectory. The initial joint positions
    # are set to random values, except for the reference joint whose position is set to an extreme value.
    # The initial joint velocities are left free. The final velocities are all set to 0. The OCP has to maximise 
    # the initial velocity norm in a predefined direction.

    # Selection of the reference joint:
    joint_sel = random.choice(range(system_sel))

    # Selection of the start position of the reference joint:
    vel_sel = random.choice([-1, 1]) # -1 to maximise initial vel, + 1 to minimize it

    # Initial velocity optimization direction:
    p = np.zeros((system_sel+1))

    for l in range(system_sel):
        if l == joint_sel:
            p[l] = random.random() * vel_sel
        else:
            p[l] = random.random() * random.choice([-1, 1])

    norm_weights = norm(p)        
    p = p/norm_weights

    # Bounds on the initial state:
    q_init_lb = np.full((ocp.ocp.dims.nx), v_min)
    q_init_ub = np.full((ocp.ocp.dims.nx), v_max)
    q_init_lb[-1] = dt_sym
    q_init_ub[-1] = dt_sym

    for l in range(system_sel):

        if l == joint_sel:

            if vel_sel == -1:
                q_init_sel = q_min + eps
                q_fin_sel = q_max - eps
            else:
                q_init_sel = q_max - eps
                q_fin_sel = q_min + eps

            q_init_lb[l] = q_init_sel
            q_init_ub[l] = q_init_sel

        else:
            q_init_oth = q_min + random.random() * (q_max-q_min)

            if q_init_oth > q_max - eps:
                q_init_oth = q_init_oth - eps
            if q_init_oth < q_min + eps:
                q_init_oth = q_init_oth + eps

            q_init_lb[l] = q_init_oth
            q_init_ub[l] = q_init_oth

    # State and input bounds:
    q_lb = np.copy(q_init_lb)
    q_ub = np.copy(q_init_ub)

    for l in range(system_sel):
        q_lb[l] = q_min
        q_ub[l] = q_max

    u_lb = np.full((system_sel), -tau_max)
    u_ub = np.full((system_sel), tau_max)

    # Bounds on the final state:
    q_fin_lb = np.copy(q_lb)
    q_fin_ub = np.copy(q_ub)

    for l in range(system_sel):
        q_fin_lb[l+system_sel] = 0.
        q_fin_ub[l+system_sel] = 0.

    # Guess:
    x_sol_guess = np.empty((N, ocp.ocp.dims.nx))
    u_sol_guess = np.empty((N, ocp.ocp.dims.nu))

    for i, tau in enumerate(np.linspace(0, 1, N)):

        x_guess = np.copy(q_init_ub)

        for l in range(system_sel):
            if l == joint_sel:
                x_guess[l] = (1-tau)*q_init_sel + tau*q_fin_sel
                x_guess[l+system_sel] = 2*(1-tau)*(q_fin_sel-q_init_sel) 
            else:
                x_guess[l] = q_init_oth
                x_guess[l+system_sel] = 0

        x_sol_guess[i] = x_guess
        u_sol_guess[i] = np.zeros((system_sel))

    cost_old = 1e6
    all_ok = False

    # Iteratively solve the OCP with an increased number of time steps until the solution does not change.
    # If the solver fails, try with a slightly different initial condition:
    for _ in range(10):

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
            u_sol_guess[N] = np.zeros((ocp.ocp.dims.nu))

            # Increase the number of time steps:
            N = N + 1
            ocp.N = N
            ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
            ocp.ocp_solver.update_qp_solver_cond_N(N)

        else:
            break

        # else:

        #     # Reset the number of steps used in the OCP:
        #     N = N_start 
        #     ocp.N = N
        #     ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
        #     ocp.ocp_solver.update_qp_solver_cond_N(N)

        #     # Initial velocity optimization direction (the cost is in the form p[0] * dtheta1 + p[1] * dtheta2 + p[2] * dt):
        #     for l in range(system_sel):
        #         p[l] = p[l] + random.random() * random.choice([-1, 1]) * 0.01

        #     norm_weights = norm(p)
        #     p = p/norm_weights

        #     # Initial position of the other joint:
        #     for j in range(system_sel):
        #     	if j != joint_sel:

        #             val = q_init_lb[j] + random.random() * random.choice([-1, 1]) * 0.01

        #             if val > q_max - eps:
        #                 val = val - eps
        #             if val < q_min + eps:
        #                 val = val + eps

        #             q_init_lb[j] = val
        #             q_init_ub[j] = val

        #     # Guess:
        #     x_sol_guess = np.empty((N, ocp.ocp.dims.nx))
        #     u_sol_guess = np.empty((N, ocp.ocp.dims.nu))

        #     for i, tau in enumerate(np.linspace(0, 1, N)):

        #         x_guess = np.copy(q_init_ub)

        #         for l in range(system_sel):
        #             if l == joint_sel:
        #                 x_guess[l+system_sel] = 2*(1-tau)*(q_fin_sel-q_init_sel) 
        #             else:
        #                 x_guess[l+system_sel] = 0

        #         x_guess[joint_sel] = (1-tau)*q_init_sel + tau*q_fin_sel

        #         x_sol_guess[i] = x_guess
        #         u_sol_guess[i] = np.zeros((system_sel))

        #     cost_old = 1e6

    if all_ok: 

        # Save the optimal trajectory:
        x_sol = np.empty((N+1, ocp.ocp.dims.nx))
        u_sol = np.empty((N, ocp.ocp.dims.nu))

        for i in range(N):
            x_sol[i] = ocp.ocp_solver.get(i, "x")
            u_sol[i] = ocp.ocp_solver.get(i, "u")

        x_sol[N] = ocp.ocp_solver.get(N, "x")

        # Generate the unviable sample in the cost direction:
        x_sym = np.full((N+1,ocp.ocp.dims.nx - 1), None)

        x_out = np.copy(x_sol[0][:ocp.ocp.dims.nx - 1])
        for l in range(system_sel):
            x_out[l+system_sel] = x_out[l+system_sel] - eps * p[l]

        # save the initial state:
        valid_data = np.append(valid_data, [x_sol[0][:ocp.ocp.dims.nx - 1]], axis = 0)

        # Check if initial velocities lie on a limit:
        if any(i > v_max or i < v_min for i in x_out[system_sel:ocp.ocp.dims.nx - 1]):
            is_x_at_limit = True # the state is on dX
        else:
            is_x_at_limit = False # the state is on dV
            x_sym[0] = x_out

        # Iterate through the trajectory to verify the location of the states with respect to V:
        for f in range(1, N):

            if is_x_at_limit:

                x_out = np.copy(x_sol[f][:ocp.ocp.dims.nx - 1])
                norm_vel = norm(x_out[system_sel:])    

                for l in range(system_sel):
                    x_out[l+system_sel] = x_out[l+system_sel] + eps * x_out[l+system_sel]/norm_vel

                # If the previous state was on a limit, the current state location cannot be identified using
                # the corresponding unviable state but it has to rely on the proximity to the state limits 
                # (more restrictive):
                if any(i > q_max - eps or i < q_min + eps for i in x_sol[f][:system_sel]) or any(i > v_max or i < v_min for i in x_out[system_sel:ocp.ocp.dims.nx - 1]):
                    is_x_at_limit = True # the state is on dX
                
                else:
                    is_x_at_limit = False # the state is either on the interior of V or on dV

                    # if the traj detouches from a position limit it usually enters V:
                    if any(i > q_max - eps or i < q_min + eps for i in x_sol[f-1][:system_sel]):
                        break

                    # Solve an OCP to verify whether the following part of the trajectory is on V or dV. To do so
                    # the initial joint positions are set to the current ones and the final state is fixed to the
                    # final state of the trajectory. The initial velocities are left free and maximized in the 
                    # direction of the current joint velocities.

                    N_test = N - f
                    ocp.N = N_test
                    ocp.ocp_solver.set_new_time_steps(np.full((N_test,), 1.))
                    ocp.ocp_solver.update_qp_solver_cond_N(N_test)

                    # Cost: 
                    norm_weights = norm(x_sol[f][system_sel:ocp.ocp.dims.nx - 1])    
                    p = np.zeros((system_sel+1))
                    for l in range(system_sel):
                        p[l] = -x_sol[f][l+system_sel]/norm_weights # the cost direction is based on the current velocity direction

                    # Bounds on the initial state:
                    for l in range(system_sel):
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

                    norm_old = norm(x_sol[f][system_sel:ocp.ocp.dims.nx - 1]) # velocity norm of the original solution 
                    norm_bef = 0
                    all_ok = False

                    for _ in range(5):
                        
                        # Solve the OCP:
                        status = ocp.OCP_solve(x_sol_guess, u_sol_guess, p, q_lb, q_ub, u_lb, u_ub, q_init_lb, q_init_ub, q_fin_lb, q_fin_ub)

                        if status == 0: # the solver has found a solution

                            # Compare the current cost with the previous one:
                            x0_new = ocp.ocp_solver.get(0, "x")
                            norm_new = norm(x0_new[system_sel:ocp.ocp.dims.nx - 1])

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
                            ocp.ocp_solver.set_new_time_steps(np.full((N_test,), 1.))
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

                            x_out = np.copy(x_sol[f][:ocp.ocp.dims.nx - 1])

                            for l in range(system_sel):
                                x_out[l+system_sel] = x_out[l+system_sel] + eps * x_out[l+system_sel]/norm_new

                            # Check if velocities lie on a limit:
                            if any(i > v_max or i < v_min for i in x_out[system_sel:ocp.ocp.dims.nx - 1]):
                                is_x_at_limit = True # the state is on dX
                            else:
                                is_x_at_limit = False # the state is on dV
                                x_sym[f] = x_out
                                
                        else:
                            is_x_at_limit = False # the state is on dV

                            # Generate the new corresponding unviable state in the cost direction:
                            x_out = np.copy(x_sol[f][:ocp.ocp.dims.nx - 1])
                            
                            for l in range(system_sel):
                                x_out[l+system_sel] = x_out[l+system_sel] - eps * p[l]

                            if x_out[joint_sel+system_sel] > v_max:
                                x_out[joint_sel+system_sel] = v_max
                            if x_out[joint_sel+system_sel] < v_min:
                                x_out[joint_sel+system_sel] = v_min

                            x_sym[f] = x_out

                    else: # we cannot say whether the state is on dV or inside V

                        for r in range(f, N):
                            if any(abs(i) > v_max - eps for i in x_sol[r][system_sel:ocp.ocp.dims.nx - 1]):
                                                                    
                                # Save the viable states at velocity limits:
                                valid_data = np.append(valid_data, [x_sol[f][:ocp.ocp.dims.nx - 1]], axis = 0)

                        break      
     
            else:
                # If the previous state was not on a limit, the current state location can be identified using
                # the corresponding unviable state which can be computed by simulating the system starting from 
                # the previous unviable state.

                # Simulate next unviable state:
                u_sym = np.copy(u_sol[f-1])
                sim.acados_integrator.set("u", u_sym)
                sim.acados_integrator.set("x", x_sym[f-1])
                # sim.acados_integrator.set("T", dt_sym)
                status = sim.acados_integrator.solve()
                x_out = sim.acados_integrator.get("x")
                x_sym[f] = x_out

                # When the state of the unviable simulated trajectory violates a limit, the corresponding viable state
                # of the optimal trajectory is on dX:
                if any(i > q_max or i < q_min for i in x_out[:system_sel]) or any(i > v_max or i < v_min for i in x_out[system_sel:ocp.ocp.dims.nx - 1]):
                    is_x_at_limit = False # the state is on dV
                else:
                    is_x_at_limit = True # the state is on dX

            if all(i < q_max - eps and i > q_min + eps for i in x_sol[f][:system_sel]) and all(abs(i) > tol for i in x_sol[f][system_sel:ocp.ocp.dims.nx - 1]):
                
                # Save the viable and unviable states:
                valid_data = np.append(valid_data, [x_sol[f][:ocp.ocp.dims.nx - 1]], axis = 0)

        return  valid_data.tolist()
    
    else:
        return None

start_time = time.time()

# Select system:
system_sel = 2 # 2 for 2dof, 3 for 3dof

# Prune the model:
prune_model = True
prune_amount = 0.5 # percentage of connections to delete

# Ocp initialization:
if system_sel == 3:
    ocp = OCPtriplependulumINIT()
    sim = SYMtriplependulumINIT()
elif system_sel == 2:
    ocp = OCPdoublependulumINIT()
    sim = SYMdoublependulumINIT()
else:
    raise Exception("Sorry, the selected system is not recognised") 

# Position, velocity and torque bounds:
v_max = ocp.dthetamax
v_min = - ocp.dthetamax
q_max = ocp.thetamax
q_min = ocp.thetamin
tau_max = ocp.Cmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pytorch device

dt_sym = 1e-2 # time step duration
N_start = 100 # initial number of timesteps
tol = ocp.ocp.solver_options.nlp_solver_tol_stat # OCP cost tolerance
eps = tol*10 # unviable data generation parameter

print('Start data generation')

# # Data generation:
# cpu_num = 30
# num_prob = 10000
# with Pool(cpu_num) as p:
#     traj = p.map(testing, range(num_prob))

# # traj, statpos, statneg = zip(*temp)
# X_temp = [i for i in traj if i is not None]
# print('Data generation completed')

# # Print data generations statistics:
# solved=len(X_temp)
# print('Solved/tot', len(X_temp)/num_prob)
# X_save = np.array([i for f in X_temp for i in f])
# print('Saved/tot', len(X_save)/(solved*100))

# # Save training data:
# np.save('data_' + str(system_sel) + 'dof_vboc_' + str(int(v_max)), np.asarray(X_save))

# Load training data:
X_save = np.load('data_' + str(system_sel) + 'dof_vboc_' + str(int(v_max)) + '.npy')

# Pytorch params:
input_layers = ocp.ocp.dims.nx - 1
hidden_layers = (input_layers - 1) * 100
output_layers = 1
learning_rate = 1e-3

# Model and optimizer:
model_dir = NeuralNetRegression(input_layers, hidden_layers, output_layers).to(device)
criterion_dir = nn.MSELoss()
optimizer_dir = torch.optim.Adam(model_dir.parameters(), lr=learning_rate)

# Joint positions mean and variance:
mean_dir, std_dir = torch.mean(torch.tensor(X_save[:,:system_sel].tolist())).to(device).item(), torch.std(torch.tensor(X_save[:,:system_sel].tolist())).to(device).item()
torch.save(mean_dir, 'mean_' + str(system_sel) + 'dof_vboc_' + str(int(v_max)) + '_' + str(hidden_layers))
torch.save(std_dir, 'std_' + str(system_sel) + 'dof_vboc_' + str(int(v_max)) + '_' + str(hidden_layers))

# Rewrite data in the form [normalized positions, velocity direction, velocity norm]:
X_train_dir = np.empty((X_save.shape[0],ocp.ocp.dims.nx))

for i in range(X_train_dir.shape[0]):

    vel_norm = norm(X_save[i][system_sel:ocp.ocp.dims.nx - 1])
    X_train_dir[i][ocp.ocp.dims.nx - 1] = vel_norm #* 0.9

    for l in range(system_sel):
        X_train_dir[i][l] = (X_save[i][l] - mean_dir) / std_dir
        X_train_dir[i][l+system_sel] = X_save[i][l+system_sel] / vel_norm

# model_dir.load_state_dict(torch.load('model_' + system_sel + 'dof_vboc_' + v_max))

beta = 0.95
n_minibatch = 4096
B = int(X_save.shape[0]*100/n_minibatch) # number of iterations for 100 epoch
it_max = B * 10

training_evol = []

print('Start model training')

it = 1
val = max(X_train_dir[:,ocp.ocp.dims.nx - 1])

# Train the model
while val > 1e-3 and it < it_max:
    ind = random.sample(range(len(X_train_dir)), n_minibatch)

    X_iter_tensor = torch.Tensor([X_train_dir[i][:ocp.ocp.dims.nx - 1] for i in ind]).to(device)
    y_iter_tensor = torch.Tensor([[X_train_dir[i][ocp.ocp.dims.nx - 1]] for i in ind]).to(device)

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
        training_evol.append(val)

print('Model training completed')

# Show the resulting RMSE on the training data:
outputs = np.empty((len(X_train_dir),1))
n_minibatch_model = pow(2,15)
with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_train_dir[:,:ocp.ocp.dims.nx - 1]).to(device)
    y_iter_tensor = torch.Tensor(X_train_dir[:,ocp.ocp.dims.nx - 1:]).to(device)
    my_dataloader = DataLoader(X_iter_tensor,batch_size=n_minibatch_model,shuffle=False)
    for (idx, batch) in enumerate(my_dataloader):
        if n_minibatch_model*(idx+1) > len(X_train_dir):
            outputs[n_minibatch_model*idx:len(X_train_dir)] = model_dir(batch).cpu().numpy()
        else:
            outputs[n_minibatch_model*idx:n_minibatch_model*(idx+1)] = model_dir(batch).cpu().numpy()
    outputs_tensor = torch.Tensor(outputs).to(device)
    print('RMSE train data: ', torch.sqrt(criterion_dir(outputs_tensor, y_iter_tensor))) 

# Compute resulting RMSE wrt testing data:
X_test = np.load('../data' + str(system_sel) + '_test_10.npy')

X_test_dir = np.empty((X_test.shape[0],ocp.ocp.dims.nx))
for i in range(X_test_dir.shape[0]):
    vel_norm = norm(X_test[i][system_sel:ocp.ocp.dims.nx - 1])
    X_test_dir[i][ocp.ocp.dims.nx - 1] = vel_norm
    for l in range(system_sel):
        X_test_dir[i][l] = (X_test[i][l] - mean_dir) / std_dir
        X_test_dir[i][l+system_sel] = X_test[i][l+system_sel] / vel_norm

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_test_dir[:,:ocp.ocp.dims.nx - 1]).to(device)
    y_iter_tensor = torch.Tensor(X_test_dir[:,ocp.ocp.dims.nx - 1:]).to(device)
    outputs = model_dir(X_iter_tensor)
    print('RMSE test data: ', torch.sqrt(criterion_dir(outputs, y_iter_tensor)))

# # Save the model:
# torch.save(model_dir.state_dict(), 'model_' + str(system_sel) + 'dof_vboc_' + str(int(v_max)) + '_' + str(hidden_layers))

if prune_model:

    parameters_to_prune = (
        (model_dir.linear_relu_stack[0], 'weight'),
        (model_dir.linear_relu_stack[2], 'weight'),
        (model_dir.linear_relu_stack[4], 'weight'),
        (model_dir.linear_relu_stack[0], 'bias'),
        (model_dir.linear_relu_stack[2], 'bias'),
        (model_dir.linear_relu_stack[4], 'bias'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_amount,
    )

    # prune.l1_unstructured(model_dir.linear_relu_stack[0], name='weight', amount=prune_amount)
    # prune.l1_unstructured(model_dir.linear_relu_stack[2], name='weight', amount=prune_amount)
    # prune.l1_unstructured(model_dir.linear_relu_stack[4], name='weight', amount=prune_amount)
    # prune.l1_unstructured(model_dir.linear_relu_stack[0], name='bias', amount=prune_amount)
    # prune.l1_unstructured(model_dir.linear_relu_stack[2], name='bias', amount=prune_amount)
    # prune.l1_unstructured(model_dir.linear_relu_stack[4], name='bias', amount=prune_amount)

    # model_dir.load_state_dict({k: v for k, v in initial_model_params.items() if 'weight' in k or 'bias' in k}, strict=False)

    print('Restart model training after pruning')

    it = 1
    val = max(X_train_dir[:,ocp.ocp.dims.nx - 1])

    # Train the model
    while val > 1e-3 and it < it_max:
        ind = random.sample(range(len(X_train_dir)), n_minibatch)

        X_iter_tensor = torch.Tensor([X_train_dir[i][:ocp.ocp.dims.nx - 1] for i in ind]).to(device)
        y_iter_tensor = torch.Tensor([[X_train_dir[i][ocp.ocp.dims.nx - 1]] for i in ind]).to(device)

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
            training_evol.append(val)

    print('Model training completed')

    prune.remove(model_dir.linear_relu_stack[0], 'weight')
    prune.remove(model_dir.linear_relu_stack[2], 'weight')
    prune.remove(model_dir.linear_relu_stack[4], 'weight')
    prune.remove(model_dir.linear_relu_stack[0], 'bias')
    prune.remove(model_dir.linear_relu_stack[2], 'bias')
    prune.remove(model_dir.linear_relu_stack[4], 'bias')

    # print('----------------------')
    # print(list(model_dir.named_parameters()))
    # print(list(model_dir.named_buffers()))

    with torch.no_grad():
        X_iter_tensor = torch.Tensor(X_test_dir[:,:ocp.ocp.dims.nx - 1]).to(device)
        y_iter_tensor = torch.Tensor(X_test_dir[:,ocp.ocp.dims.nx - 1:]).to(device)
        outputs = model_dir(X_iter_tensor)
        print('RMSE test data after pruning: ', torch.sqrt(criterion_dir(outputs, y_iter_tensor)))

# Compute safety margin:
outputs = model_dir(X_iter_tensor).cpu().numpy()
safety_margin = np.amax(np.array([(outputs[i] - X_test_dir[i][-1])/X_test_dir[i][-1] for i in range(X_test_dir.shape[0]) if outputs[i] - X_test_dir[i][-1] > 0]))
print(safety_margin)

# Save the pruned model:
torch.save(model_dir.state_dict(), 'model_' + str(system_sel) + 'dof_vboc_' + str(int(v_max)) + '_' + str(hidden_layers) + '_' + str(prune_amount)  + '_' + str(safety_margin))

print("Execution time: %s seconds" % (time.time() - start_time))

# Show the training evolution:
plt.figure()
plt.plot(training_evol)

# Show training data and resulting set approximation:
if system_sel == 3:
    plots_3dof(X_save, q_min, q_max, v_min, v_max, model_dir, mean_dir, std_dir, device)
elif system_sel == 2:
    plots_2dof(X_save, q_min, q_max, v_min, v_max, model_dir, mean_dir, std_dir, device)

plt.show()
