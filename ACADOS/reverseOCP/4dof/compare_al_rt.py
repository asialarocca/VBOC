import numpy as np
import random 
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from numpy.linalg import norm as norm
import time
from doublependulum_class_fixedveldir import OCPdoublependulumRINIT, SYMdoublependulumINIT
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from my_nn import NeuralNet, NeuralNetRegression
import math
from multiprocessing import Pool

def testing(v):
    valid_data = np.ndarray((0, 4))

    # Reset the number of steps used in the OCP:
    N = ocp.N
    ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
    ocp.ocp_solver.update_qp_solver_cond_N(N)

    # Time step duration:
    dt_sym = 1e-2

    # Initialization of the OCP: The OCP is set to find an extreme trajectory. The initial joint positions
    # are set to random values, except for the reference joint whose position is set to an extreme value.
    # The initial joint velocities are left free. The final velocities are all set to 0 and the final 
    # position of the reference joint is set to the other extreme. The final positions of the other joints
    # are left free. The OCP has to maximise the initial velocity module in a predefined direction.

    # Selection of the reference joint:
    joint_sel = random.choice([0, 1]) # 0 to select first joint as reference, 1 to select second joint
    joint_oth = int(1 - joint_sel)

    # Selection of the start and end position of the reference joint:
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

    tmp = [vel_sel+1+joint_sel,ran1,ran2,q_init_oth]

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

    # Bounds on the final state:
    q_fin_lb = np.array([q_min, q_min, 0., 0., dt_sym])
    q_fin_ub = np.array([q_max, q_max, 0., 0., dt_sym])
    # q_fin_lb[joint_sel] = q_fin_sel
    # q_fin_ub[joint_sel] = q_fin_sel

    # Guess:
    x_sol_guess = np.empty((N, 5))
    u_sol_guess = np.empty((N, 2))
    for i, tau in enumerate(np.linspace(0, 1, N)):
        x_guess = np.array([q_init_oth, q_init_oth, 0., 0., dt_sym])
        x_guess[joint_sel] = (1-tau)*q_init_sel + tau*q_fin_sel
        x_guess[joint_sel+2] = 2*(1-tau)*(q_fin_sel-q_init_sel) 
        x_sol_guess[i] = x_guess
        u_sol_guess[i] = np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x_guess[0]),ocp.g*ocp.l2*ocp.m2*math.sin(x_guess[1])])

    cost = 1.

    all_ok = False

    for _ in range(2):
        # Iteratively solve the OCP with an increased number of time steps until the the solution does not change:
        while True:
            # Reset current iterate:
            ocp.ocp_solver.reset()

            # Set parameters, guesses and constraints:
            for i in range(N):
                ocp.ocp_solver.set(i, 'x', x_sol_guess[i])
                ocp.ocp_solver.set(i, 'u', u_sol_guess[i])
                ocp.ocp_solver.set(i, 'p', p)
                ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 
                ocp.ocp_solver.constraints_set(i, 'lbu', np.array([-tau_max, -tau_max]))
                ocp.ocp_solver.constraints_set(i, 'ubu', np.array([tau_max, tau_max]))
                ocp.ocp_solver.constraints_set(i, 'C', np.array([[0., 0., 0., 0., 0.]]))
                ocp.ocp_solver.constraints_set(i, 'D', np.array([[0., 0.]]))
                ocp.ocp_solver.constraints_set(i, 'lg', np.array([0.]))
                ocp.ocp_solver.constraints_set(i, 'ug', np.array([0.]))

            ocp.ocp_solver.constraints_set(0, "lbx", q_init_lb)
            ocp.ocp_solver.constraints_set(0, "ubx", q_init_ub)
            ocp.ocp_solver.constraints_set(0, "C", np.array([[0., 0., p[1], -p[0], 0.]])) 

            ocp.ocp_solver.constraints_set(N, "lbx", q_fin_lb)
            ocp.ocp_solver.constraints_set(N, "ubx", q_fin_ub)
            ocp.ocp_solver.set(N, 'x', x_sol_guess[-1])
            ocp.ocp_solver.set(N, 'p', p)

            # Solve the OCP:
            status = ocp.ocp_solver.solve()

            if status == 0: # the solver has found a solution
                # Compare the current cost with the previous one:
                cost_new = ocp.ocp_solver.get_cost()
                if cost_new > float(f'{cost:.6f}') - 1e-6:
                    all_ok = True # the time is sufficient to have achieved an optimal solution
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
                ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
                ocp.ocp_solver.update_qp_solver_cond_N(N)
            else:
                all_ok = False # the solution is either not available or not guaranteed to be optimal
                break
        
        if all_ok: 
            break
        else:
            # Reset the number of steps used in the OCP:
            N = ocp.N
            ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
            ocp.ocp_solver.update_qp_solver_cond_N(N)

            # Time step duration:
            dt_sym = 1e-2

            # Initial velocity optimization direction (the cost is in the form p[0] * dtheta1 + p[1] * dtheta2 + p[2] * dt):
            ran1 = ran1
            ran2 = ran2 + random.random() * random.choice([-1, 1]) * 0.1
            norm_weights = norm(np.array([ran1, ran2]))         
            if joint_sel == 0:
                p = np.array([ran1/norm_weights, ran2/norm_weights, 0.])
            else:
                p = np.array([ran2/norm_weights, ran1/norm_weights, 0.])

            # Initial position of the other joint:
            q_init_oth = q_init_oth + random.random() * random.choice([-1, 1]) * 0.1
            if q_init_oth > q_max - eps:
                q_init_oth = q_init_oth - eps
            if q_init_oth < q_min + eps:
                q_init_oth = q_init_oth + eps

            tmp = [vel_sel+1+joint_sel,ran1,ran2,q_init_oth]

            # Guess:
            x_sol_guess = np.empty((N, 5))
            u_sol_guess = np.empty((N, 2))
            for i, tau in enumerate(np.linspace(0, 1, N)):
                x_guess = np.array([q_init_oth, q_init_oth, 0., 0., dt_sym])
                x_guess[joint_sel] = (1-tau)*q_init_sel + tau*q_fin_sel
                x_guess[joint_sel+2] = 2*(1-tau)*(q_fin_sel-q_init_sel) 
                x_sol_guess[i] = x_guess
                u_sol_guess[i] = np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x_guess[0]),ocp.g*ocp.l2*ocp.m2*math.sin(x_guess[1])])

            cost = 1.

    if all_ok: 
        # Save the optimal trajectory:
        x_sol = np.empty((N+1,5))
        u_sol = np.empty((N,2))
        for i in range(N):
            x_sol[i] = ocp.ocp_solver.get(i, "x")
            u_sol[i] = ocp.ocp_solver.get(i, "u")
        x_sol[N] = ocp.ocp_solver.get(N, "x")

        # Minimum time OCP:
        if min_time:
            # Reset current iterate:
            ocp.ocp_solver.reset()

            # Cost:
            p_mintime = np.array([0., 0., 1.]) # p[2] = 1 corresponds to the minimization of time

            # Set parameters, guesses and constraints:
            for i in range(N):
                ocp.ocp_solver.set(i, 'x', x_sol[i])
                ocp.ocp_solver.set(i, 'u', u_sol[i])
                ocp.ocp_solver.set(i, 'p', p_mintime)
                ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 0.])) 
                ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

            # Initial and final states are fixed (except for the time step duration):
            ocp.ocp_solver.constraints_set(N, "lbx", np.array([x_sol[N][0], x_sol[N][1], 0., 0., 0.]))
            ocp.ocp_solver.constraints_set(N, "ubx", np.array([x_sol[N][0], x_sol[N][1], 0., 0., dt_sym]))
            ocp.ocp_solver.set(N, 'x', x_sol[N])
            ocp.ocp_solver.set(N, 'p', p_mintime)

            ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x_sol[0][0], x_sol[0][1], x_sol[0][2], x_sol[0][3], 0.])) 
            ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x_sol[0][0], x_sol[0][1], x_sol[0][2], x_sol[0][3], dt_sym])) 
            ocp.ocp_solver.constraints_set(0, "C", np.array([[0., 0., 0., 0., 0.]])) 

            # Solve the OCP:
            status = ocp.ocp_solver.solve()

            if status == 0:
                # Save the new optimal trajectory:
                for i in range(N):
                    x_sol[i] = ocp.ocp_solver.get(i, "x")
                    u_sol[i] = ocp.ocp_solver.get(i, "u")
                x_sol[N] = ocp.ocp_solver.get(N, "x")

                # Save the optimized time step duration:
                dt_sym = ocp.ocp_solver.get(0, "x")[4]

        # Save the last state of the optimal trajectory and its corresponding unviable one:
        x_fin_out = np.copy(x_sol[N])
        x_fin_out[joint_sel] = x_fin_out[joint_sel] - vel_sel * eps

        # if abs(x_sol[N][2]) > 1e-3 and abs(x_sol[N][3]) > 1e-3:
        #     valid_data = np.append(valid_data, [[x_sol[N][0], x_sol[N][1], x_sol[N][2], x_sol[N][3]]], axis = 0)

        # Generate the unviable sample in the cost direction:
        x_out = np.copy(x_sol[0][:4])
        x_out[2] = x_out[2] - eps * p[0]
        x_out[3] = x_out[3] - eps * p[1]
        x_sym = np.empty((N+1,4))
        x_sym[0] = x_out

        valid_data = np.append(valid_data, [[x_sol[0][0], x_sol[0][1], x_sol[0][2], x_sol[0][3]]], axis = 0)

        # xv_state = np.full((N+1,1),2)
        # xv_state[N] = 1

        # Check if initial velocities lie on a limit:
        if x_out[2] > v_max or x_out[2] < v_min or x_out[3] > v_max or x_out[3] < v_min:
            is_x_at_limit = True # the state is on dX
            # xv_state[0] = 1
        else:
            is_x_at_limit = False # the state is on dV
            # xv_state[0] = 0

        # Iterate through the trajectory to verify the location of the states with respect to V:
        for f in range(1, N+1):

            if is_x_at_limit:
                # If the previous state was on a limit, the current state location cannot be identified using
                # the corresponding unviable state but it has to rely on the proximity to the state limits 
                # (more restrictive):
                if x_sol[f][0] > q_max - eps or x_sol[f][0] < q_min + eps or x_sol[f][1] > q_max - eps or x_sol[f][1] < q_min + eps or x_sol[f][2] > v_max - eps or x_sol[f][2] < v_min + eps or x_sol[f][3] > v_max - eps or x_sol[f][3] < v_min + eps:
                    is_x_at_limit = True # the state is on dX
                    # xv_state[f] = 1

                    # Generate the corresponding unviable state:
                    x_out = np.copy(x_sol[f][:4])
                    if x_sol[f][0] > q_max - eps:
                        x_out[0] = q_max + eps
                    if x_sol[f][0] < q_min + eps:
                        x_out[0] = q_min - eps
                    if x_sol[f][1] > q_max - eps:
                        x_out[1] = q_max + eps
                    if x_sol[f][1] < q_min + eps:
                        x_out[1] = q_min - eps
                    if x_sol[f][2] > v_max - eps:
                        x_out[2] = v_max + eps
                    if x_sol[f][2] < v_min + eps:
                        x_out[2] = v_min - eps
                    if x_sol[f][3] > v_max - eps:
                        x_out[3] = v_max + eps
                    if x_sol[f][3] < v_min + eps:
                        x_out[3] = v_min - eps

                    # Save the unviable state:
                    x_sym[f] = x_out
                else:
                    is_x_at_limit = False # the state is either on the interior of V or on dV

                    # Solve an OCP to verify whether the following part of the trajectory is on V or dV. To do so
                    # the initial joint positions are set to the current ones and the final state is fixed to the
                    # final state of the trajectory. The initial velocities are left free and maximized in the 
                    # direction of the current joint velocities.

                    # Reset current iterate:
                    ocp.ocp_solver.reset()

                    # Cost:
                    norm_weights = norm(np.array([x_sol[f][2], x_sol[f][3]]))    
                    p = np.array([-(x_sol[f][2])/norm_weights, -(x_sol[f][3])/norm_weights, 0.]) # the cost direction is based on the current velocity direction

                    # Bounds on the initial state:
                    lbx_init = np.array([x_sol[f][0], x_sol[f][1], v_min, v_min, dt_sym])
                    ubx_init = np.array([x_sol[f][0], x_sol[f][1], v_max, v_max, dt_sym])
                    if x_sol[f][2] < 0.:
                        ubx_init[2] = x_sol[f][2]
                    else:
                        lbx_init[2] = x_sol[f][2]
                    if x_sol[f][3] < 0.:
                        ubx_init[3] = x_sol[f][3]
                    else:
                        lbx_init[3] = x_sol[f][3]

                    # Set parameters, guesses and constraints:
                    for i in range(N-f):
                        ocp.ocp_solver.set(i, 'x', x_sol[i+f])
                        ocp.ocp_solver.set(i, 'u', u_sol[i+f])
                        ocp.ocp_solver.set(i, 'p', p)
                        ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                        ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                    ocp.ocp_solver.constraints_set(0, 'lbx', lbx_init) 
                    ocp.ocp_solver.constraints_set(0, 'ubx', ubx_init) 
                    ocp.ocp_solver.constraints_set(0, "C", np.array([[0., 0., p[1], -p[0], 0.]])) 

                    u_g = np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x_sol[N][0]),ocp.g*ocp.l2*ocp.m2*math.sin(x_sol[N][1])])
                    
                    # The OCP is set with N time steps instead of N-f (corresponding with the remaining states of the
                    # optimal trajectory) because it is also necessary to check if the maximum velocity norm assume the same value 
                    # also with more time:
                    for i in range(N - f, N):
                        ocp.ocp_solver.set(i, 'x', x_sol[N])
                        ocp.ocp_solver.set(i, 'u', u_g)
                        ocp.ocp_solver.set(i, 'p', p)
                        ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                        ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                    ocp.ocp_solver.set(N, 'x', x_sol[N])
                    ocp.ocp_solver.set(N, 'p', p)
                    ocp.ocp_solver.constraints_set(N, 'lbx', x_sol[N]) 
                    ocp.ocp_solver.constraints_set(N, 'ubx', x_sol[N]) 

                    # Solve the OCP:
                    status = ocp.ocp_solver.solve()

                    if status == 0:
                        # Compare the old and new velocity norms:
                        x0_new = ocp.ocp_solver.get(0, "x")
                        norm_old = norm(np.array([x_sol[f][2:4]]))    
                        norm_new = norm(np.array([x0_new[2:4]]))    

                        if norm_new > norm_old + 1e-6: # the state is inside V

                            break
                        else:
                            is_x_at_limit = False # the state is on dV
                            # xv_state[f] = 0

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
                        break
                    
            else:
                # If the previous state was not on a limit, the current state location can be identified using
                # the corresponding unviable state which can be computed by simulating the system starting from 
                # the previous unviable state:
                u_sym = np.copy(u_sol[f-1])
                sim.acados_integrator.set("u", u_sym)
                sim.acados_integrator.set("x", x_out)
                sim.acados_integrator.set("T", dt_sym)
                status = sim.acados_integrator.solve()
                x_out = sim.acados_integrator.get("x")

                x_sym[f] = x_out

                # When the state of the unviable simulated trajectory violates a limit, the corresponding viable state
                # of the optimal trajectory is on dX:
                if x_out[0] > q_max or x_out[0] < q_min or x_out[1] > q_max or x_out[1] < q_min or x_out[2] > v_max or x_out[2] < v_min or x_out[3] > v_max or x_out[3] < v_min:
                    is_x_at_limit = True # the state is on dX
                    # xv_state[f] = 1
                else:
                    is_x_at_limit = False # the state is on dV
                    # xv_state[f] = 0

            if is_x_at_limit == False and abs(x_sol[f][2]) > 1e-3 and abs(x_sol[f][3]) > 1e-3:
                # Save the viable and unviable states:
                valid_data = np.append(valid_data, [[x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3]]], axis = 0)

        return  valid_data.tolist(), tmp, None

    else:
        return None, None, tmp

# Ocp initialization:
ocp = OCPdoublependulumRINIT()
sim = SYMdoublependulumINIT()

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

min_time = 1 # set to 1 to also solve a minimum time problem to improve the solution

cpu_num = 30

num_prob = 1000

# # Data generation:
# with Pool(cpu_num) as p:
#     temp = p.map(testing, range(num_prob))

# traj, statpos, statneg = zip(*temp)
# X_save = [i for i in traj if i is not None]

# print('Solved/tot', len(X_save)/num_prob)

# X_save = np.array([i for f in X_save for i in f])
# np.save('data_reverse_test.npy', np.asarray(X_save))

X_test = np.load('data_reverse_test.npy')

# Reverse OCP model and data:
model_dir = NeuralNetRegression(4, 512, 1).to(device)
criterion_dir = nn.MSELoss()
model_dir.load_state_dict(torch.load('model_2pendulum_dir_20_newlimits_new'))
X_reverse = np.load('data_reverse_10000_20_newlimits_highertol_new.npy')
mean_dir, std_dir = torch.mean(torch.tensor(X_reverse[:,:2].tolist())).to(device).item(), torch.std(torch.tensor(X_reverse[:,:2].tolist())).to(device).item()
std_out_dir = torch.std(torch.tensor(X_reverse[:,2:].tolist())).to(device).item()

# Active Learning model and data:
model_al = NeuralNet(4, 400, 2).to(device)
model_al.load_state_dict(torch.load('data_vel_20/model_2pendulum_20_newlimits'))
mean_al, std_al = torch.tensor(1.5708), torch.tensor(9.1246) # max vel = 20 e different pos limits
data_al = np.load('data_vel_20/data_al_20_newlimits.npy')

# Test data rewritten with velocity vector and norm:
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
    outp = np.argmax(model_al((torch.Tensor(X_save).to(device) - mean_al) / std_al).cpu().numpy(), axis=1)

data_neg = np.array([X_save[i] for i in range(X_save.shape[0]) if outp[i] == 0])
data_pos = np.array([X_save[i] for i in range(X_save.shape[0]) if outp[i] == 1])

error_norm_neg = np.empty((len(data_neg),))

for i in range(len(data_neg)):
    vel_norm = norm([data_neg[i][2],data_neg[i][3]])

    v0 = data_neg[i][2]
    v1 = data_neg[i][3]

    out = 0

    while out == 0 and norm([v0,v1]) > 1e-1:
        v0 = v0 - 1e-1 * data_neg[i][2]/vel_norm
        v1 = v1 - 1e-1 * data_neg[i][3]/vel_norm
        out = np.argmax(model_al((torch.Tensor([[data_neg[i][0], data_neg[i][1], v0, v1]]).to(device) - mean_al) / std_al).cpu().detach().numpy(), axis=1)

    error_norm_neg[i] = vel_norm - norm([v0,v1]) # 100*(vel_norm - norm([v0,v1]))/vel_norm

error_norm_pos = np.empty((len(data_pos),))

for i in range(len(data_pos)):
    vel_norm = norm([data_pos[i][2],data_pos[i][3]])

    v0 = data_pos[i][2]
    v1 = data_pos[i][3]

    out = 1

    while out == 1 and norm([v0,v1]) > 1e-1:
        v0 = v0 + 1e-1 * data_pos[i][2]/vel_norm
        v1 = v1 + 1e-1 * data_pos[i][3]/vel_norm
        out = np.argmax(model_al((torch.Tensor([[data_pos[i][0], data_pos[i][1], v0, v1]]).to(device) - mean_al) / std_al).cpu().detach().numpy(), axis=1)

    error_norm_pos[i] = vel_norm - norm([v0,v1]) # 100*(vel_norm - norm([v0,v1]))/vel_norm

error_norm = np.concatenate((error_norm_neg,error_norm_pos))

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_dir[:,:4]).to(device)
    y_iter_tensor = torch.Tensor(X_dir[:,4:]).to(device)
    outputs = model_dir(X_iter_tensor).cpu().numpy() * std_out_dir
    relative_sq_errs = [((outputs[i] - X_save_dir[i,4])/X_save_dir[i,4])**2 for i in range(len(X_save_dir))]
    rrmse = math.sqrt(np.sum(relative_sq_errs)/len(X_save_dir))
    print('RMSE relative test data in %: ', rrmse*100)
    print('RMSE test data: ', torch.sqrt(criterion_dir(model_dir(X_iter_tensor) * std_out_dir, y_iter_tensor)))

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_dir[:,:4]).to(device)
    y_iter_tensor = torch.Tensor(X_dir[:,4:]).to(device)
    outputs = model_dir(X_iter_tensor).cpu().numpy() * std_out_dir
    relative_sq_errs = [(error_norm[i])**2 for i in range(len(X_save))]
    rrmse = math.sqrt(np.sum(relative_sq_errs)/len(X_save))
    print('RMSE relative AL data in %: ', rrmse)
    # print('RMSE AL data: ', torch.sqrt(criterion_dir(model_dir(X_iter_tensor) * std_out_dir, y_iter_tensor)))

plt.figure()
x=[100*(X_save_dir[i,4] - outputs[i].tolist()[0])/X_save_dir[i,4] for i in range(len(X_save_dir))]
bins = np.linspace(-50, 50, 100)
plt.hist(x, bins, alpha=0.5, label='Reverse-time')
plt.hist(error_norm, bins, alpha=0.5, label='Active Learning')
plt.legend(loc='upper right')
plt.show()



X_dir_al = np.empty((data_al.shape[0],6))

for i in range(X_dir_al.shape[0]):
    X_dir_al[i][0] = (data_al[i][0] - mean_dir) / std_dir
    X_dir_al[i][1] = (data_al[i][1] - mean_dir) / std_dir
    vel_norm = norm([data_al[i][2],data_al[i][3]])
    if vel_norm != 0:
        X_dir_al[i][2] = data_al[i][2] / vel_norm
        X_dir_al[i][3] = data_al[i][3] / vel_norm
    X_dir_al[i][4] = vel_norm 
    X_dir_al[i][5] = data_al[i][5]

correct = 0

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
    outputs = model_dir(X_iter_tensor).cpu().numpy() * std_out_dir
    for i in range(X_dir_al.shape[0]):
        if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
            correct += 1
        if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
            correct += 1

    print('Accuracy AL data: ', correct/X_dir_al.shape[0])

data_internal = np.array([data_al[i] for i in range(data_al.shape[0]) if model_al((torch.Tensor(data_al[i,:4]).to(device) - mean_al) / std_al)[1]>0])

X_dir_al = np.empty((data_internal.shape[0],6))

for i in range(X_dir_al.shape[0]):
    X_dir_al[i][0] = (data_internal[i][0] - mean_dir) / std_dir
    X_dir_al[i][1] = (data_internal[i][1] - mean_dir) / std_dir
    vel_norm = norm([data_internal[i][2],data_internal[i][3]])
    if vel_norm != 0:
        X_dir_al[i][2] = data_internal[i][2] / vel_norm
        X_dir_al[i][3] = data_internal[i][3] / vel_norm
    X_dir_al[i][4] = vel_norm 
    X_dir_al[i][5] = data_internal[i][5]

correct = 0

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
    outputs = model_dir(X_iter_tensor).cpu().numpy() * std_out_dir
    for i in range(X_dir_al.shape[0]):
        if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
            correct += 1
        if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
            correct += 1

    print('Accuracy AL internal data: ', correct/X_dir_al.shape[0])

data_boundary = np.array([data_al[i] for i in range(data_al.shape[0]) if abs(model_al((torch.Tensor(data_al[i,:4]).to(device) - mean_al) / std_al)[0]) < 10])

X_dir_al = np.empty((data_boundary.shape[0],6))

for i in range(X_dir_al.shape[0]):
    X_dir_al[i][0] = (data_boundary[i][0] - mean_dir) / std_dir
    X_dir_al[i][1] = (data_boundary[i][1] - mean_dir) / std_dir
    vel_norm = norm([data_boundary[i][2],data_boundary[i][3]])
    if vel_norm != 0:
        X_dir_al[i][2] = data_boundary[i][2] / vel_norm
        X_dir_al[i][3] = data_boundary[i][3] / vel_norm
    X_dir_al[i][4] = vel_norm 
    X_dir_al[i][5] = data_boundary[i][5]

correct = 0

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
    outputs = model_dir(X_iter_tensor).cpu().numpy() * std_out_dir
    for i in range(X_dir_al.shape[0]):
        if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
            correct += 1
        if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
            correct += 1

    print('Accuracy AL data on boundary: ', correct/X_dir_al.shape[0])

data_notboundary = np.array([data_al[i] for i in range(data_al.shape[0]) if abs(model_al((torch.Tensor(data_al[i,:4]).to(device) - mean_al) / std_al)[0]) > 10])

X_dir_al = np.empty((data_notboundary.shape[0],6))

for i in range(X_dir_al.shape[0]):
    X_dir_al[i][0] = (data_notboundary[i][0] - mean_dir) / std_dir
    X_dir_al[i][1] = (data_notboundary[i][1] - mean_dir) / std_dir
    vel_norm = norm([data_notboundary[i][2],data_notboundary[i][3]])
    if vel_norm != 0:
        X_dir_al[i][2] = data_notboundary[i][2] / vel_norm
        X_dir_al[i][3] = data_notboundary[i][3] / vel_norm
    X_dir_al[i][4] = vel_norm 
    X_dir_al[i][5] = data_notboundary[i][5]

correct = 0

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
    outputs = model_dir(X_iter_tensor).cpu().numpy() * std_out_dir
    for i in range(X_dir_al.shape[0]):
        if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
            correct += 1
        if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
            correct += 1

    print('Accuracy AL data not on boundary: ', correct/X_dir_al.shape[0])
