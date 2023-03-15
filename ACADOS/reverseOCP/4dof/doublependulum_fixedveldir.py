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
from sklearn.decomposition import PCA


if __name__ == "__main__":

    start_time = time.time()

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

    # Active Learning model and data (used as the "ground truth" to check correctness of the approach):
    model_al = NeuralNet(4, 400, 2).to(device)
    model_al.load_state_dict(torch.load('data_vel_30/model_2pendulum_30_newlimits'))
    # mean, std = torch.tensor(1.9635), torch.tensor(3.0036) # max vel = 5
    # mean_al, std_al = torch.tensor(1.9635), torch.tensor(7.0253) # max vel = 15
    mean_al, std_al = torch.tensor(1.1781), torch.tensor(13.5420) # max vel = 30 e different pos limits
    # mean_al, std_al = torch.tensor(1.9635).to(device), torch.tensor(9.2003).to(device) # max vel = 20
    data_al = np.load('data_vel_30/data_al_30_newlimits.npy')

    # Unviable data generation parameter:
    eps = 1e-4
    
    # Initialization of the array that will contain generated data:
    X_save = np.empty((0,4))

    count_solved = 0
    count_tot = 0
    count_totpoints = 0
    count_added = 0
    error_found = 0
    count_first_max = 0
    count_first_min = 0
    count_second_min = 0
    count_second_max = 0

    check_data = 0 # set to 1 to check the correctness of generated data (to be used only while debugging)
    min_time = 1 # set to 1 to also solve a minimum time problem to improve the solution

    # Data generation:
    while True:
        # Stopping condition:
        if X_save.shape[0] >= 10000:
            break

        # Additional stopping condition when an error is found while checking data:
        if check_data and error_found:
            print('ERROR FOUND')

            # Save the trajectory that resulted in an error:
            np.save('x_sol.npy', x_sol)
            np.save('u_sol.npy', u_sol)
            np.save('x_sym.npy', x_sym)
            break

        print('--------------------------------------------')

        count_tot += 1

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
        q_fin_lb[joint_sel] = q_fin_sel
        q_fin_ub[joint_sel] = q_fin_sel

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
                print('FAILED')
                all_ok = False # the solution is either not available or not guaranteed to be optimal
                break

        if all_ok: 
            count_solved += 1
            count_totpoints += N

            if joint_sel == 0 and vel_sel == -1:
                count_first_max += 1
            if joint_sel == 1 and vel_sel == -1:
                count_second_max += 1
            if joint_sel == 0 and vel_sel == 1:
                count_first_min += 1
            if joint_sel == 1 and vel_sel == 1:
                count_second_min += 1
            
            print('INITIAL OCP SOLVED')
            print(ocp.ocp_solver.get(0, "x"))
            print(p)
            if check_data:
                np.save('p.npy', p)

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

                    print('MIN TIME SOLVED')

            x_fin_out = np.copy(x_sol[N])
            x_fin_out[joint_sel] = x_fin_out[joint_sel] - vel_sel * eps

            if abs(x_sol[N][2]) > 1e-3 and abs(x_sol[N][3]) > 1e-3:
                # Save the last state of the optimal trajectory and its corresponding unviable one:
                X_save = np.append(X_save, [[x_sol[N][0], x_sol[N][1], x_sol[N][2], x_sol[N][3]]], axis = 0)

            # Generate the unviable sample in the cost direction:
            x_out = np.copy(x_sol[0][:4])
            x_out[2] = x_out[2] - eps * p[0]
            x_out[3] = x_out[3] - eps * p[1]
            x_sym = np.empty((N+1,4))
            x_sym[0] = x_out

            # Save the initial state of the optimal trajectory:
            X_save = np.append(X_save, [[x_sol[0][0], x_sol[0][1], x_sol[0][2], x_sol[0][3]]], axis = 0)

            # xv_state = np.full((N+1,1),2)
            # xv_state[N] = 1

            # Check if initial velocities lie on a limit:
            if x_out[2] > v_max or x_out[2] < v_min or x_out[3] > v_max or x_out[3] < v_min:
                is_x_at_limit = True # the state is on dX
                print('Initial state at the limit')
                # xv_state[0] = 1
            else:
                is_x_at_limit = False # the state is on dV
                print('Initial state not at the limit')
                # xv_state[0] = 0

            # Iterate through the trajectory to verify the location of the states with respect to V:
            for f in range(1, N):
                print('State ', f)

                if is_x_at_limit:
                    # If the previous state was on a limit, the current state location cannot be identified using
                    # the corresponding unviable state but it has to rely on the proximity to the state limits 
                    # (more restrictive):
                    if x_sol[f][0] > q_max - eps or x_sol[f][0] < q_min + eps or x_sol[f][1] > q_max - eps or x_sol[f][1] < q_min + eps or x_sol[f][2] > v_max - eps or x_sol[f][2] < v_min + eps or x_sol[f][3] > v_max - eps or x_sol[f][3] < v_min + eps:
                        is_x_at_limit = True # the state is on dX
                        print('State on dX')
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

                        # if x_sol[f-1][joint_sel] > q_max - eps or x_sol[f-1][joint_sel] < q_min + eps:
                        #     print('Detouching from the limit on the reference joint, break')
                        #     break

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
                            print('INTERMEDIATE OCP SOLVED')
                            print(x_sol[f], x_out)

                            # Compare the old and new velocity norms:
                            x0_new = ocp.ocp_solver.get(0, "x")
                            norm_old = norm(np.array([x_sol[f][2:4]]))    
                            norm_new = norm(np.array([x0_new[2:4]]))    

                            if norm_new > norm_old + 1e-6: # the state is inside V
                                print('new vel: ', x0_new[2:4], 'old vel: ', x_sol[f][2:4])
                                print('new vel norm: ', norm_new, 'old vel norm: ', norm_old)
                                print('Test failed, state inside V, break')

                                break
                            else:
                                print('State on dV, test passed')
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
                            print('INTERMEDIATE OCP FAILED, break')
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
                        print('State on dX')
                        is_x_at_limit = True # the state is on dX
                        # xv_state[f] = 1
                    else:
                        print('State on dV')
                        is_x_at_limit = False # the state is on dV
                        # xv_state[f] = 0

                if is_x_at_limit == False and abs(x_sol[f][2]) > 1e-3 and abs(x_sol[f][3]) > 1e-3:
                    # Save the viable and unviable states:
                    X_save = np.append(X_save, [[x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3]]], axis = 0)
                    print(x_sol[f], x_out)
                    count_added += 1

                # Empiric check if data correctness base on the classifier trained on the Active Learning data:
                if check_data and is_x_at_limit == False:
                    with torch.no_grad():
                        # Compute the entropy of the current viable and unviable samples wrt the AL classifier:
                        sigmoid = nn.Sigmoid()
                        out_v = model_al((torch.from_numpy(np.float32([x_sol[f][:4]])).to(device) - mean_al) / std_al).cpu().numpy()
                        out_uv = model_al((torch.from_numpy(np.float32([x_out[:4]])).to(device) - mean_al) / std_al).cpu().numpy()
                        prob_xu = sigmoid(model_al((torch.from_numpy(np.float32([x_out[:4]])).to(device) - mean_al) / std_al)).cpu()
                        etp = entropy(prob_xu, axis=1)

                        if out_uv[0,0] < out_v[0,0]: # the classifier would most likely classify as 0 the viable state than the unviable one
                            print('Possible direction error') 

                        if out_v[0,1] < 0:
                        # if etp < 1e-2: # the classifier would be almost certain in classifing the unviable sample as viable
                            print('Possible classification error')
                            print('1 set, at the limit, x: ', x_sol[f], out_v)
                            print('1 set, at the limit, x_out: ', x_out, out_uv)

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

                                if norm_new > norm_old + 1e-4:
                                    print('new vel: ', x0_new[2:4], 'old vel: ', x_sol[f][2:4])
                                    print('new vel norm: ', norm_new, 'old vel norm: ', norm_old)
                                    print('ERROR FOUND!!!!!!!!!!!!!!!!!!!!')
                                    error_found = 1

    print('Total OCP solved', count_solved/count_tot)
    print('Total saved samples', count_added/count_totpoints)
    print('Total OCP solved from q1 min', count_first_max/count_solved)
    print('Total OCP solved from q1 max', count_first_min/count_solved)
    print('Total OCP solved from q2 min', count_second_max/count_solved)
    print('Total OCP solved from q2 max', count_second_min/count_solved)

    print("Execution time: %s seconds" % (time.time() - start_time))

    np.save = np.save('data_reverse_100000_30_newlimits.npy', np.asarray(X_save))
    # X_save = np.load('data_reverse_500000_20.npy')

    plt.figure()
    plt.scatter(X_save[:,0],X_save[:,1],s=0.1)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("OCP dataset positions")

    plt.figure()
    plt.scatter(X_save[:,2],X_save[:,3],s=0.1)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("OCP dataset velocities")

    model_dir = NeuralNetRegression(4, 512, 1).to(device)
    criterion_dir = nn.MSELoss()
    optimizer_dir = torch.optim.Adam(model_dir.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_dir, gamma=0.98)

    # X_save = np.array([X_save[i] for i in range(len(X_save)) if abs(X_save[i][2]) > 1e-3 and abs(X_save[i][3]) > 1e-3])

    mean_dir, std_dir = torch.mean(torch.tensor(X_save[:,:2].tolist())).to(device).item(), torch.std(torch.tensor(X_save[:,:2].tolist())).to(device).item()
    std_out_dir = norm([v_max,v_max])

    X_save_dir = np.empty((X_save.shape[0],5))

    for i in range(X_save_dir.shape[0]):
        X_save_dir[i][0] = (X_save[i][0] - mean_dir) / std_dir
        X_save_dir[i][1] = (X_save[i][1] - mean_dir) / std_dir
        vel_norm = norm([X_save[i][2],X_save[i][3]])
        if vel_norm != 0:
            X_save_dir[i][2] = X_save[i][2] / vel_norm
            X_save_dir[i][3] = X_save[i][3] / vel_norm
        X_save_dir[i][4] = vel_norm / std_out_dir

    ind = random.sample(range(len(X_save_dir)), int(len(X_save_dir)*0.7))
    X_train_dir = np.array([X_save_dir[i] for i in ind])
    X_test_dir = np.array([X_save_dir[i] for i in range(len(X_save_dir)) if i not in ind])

    # model_dir.load_state_dict(torch.load('model_2pendulum_dir_20'))

    it = 1
    val = 1.

    beta = 0.95
    n_minibatch = 512
    B = int(X_save.shape[0]*100/n_minibatch) # number of iterations for 100 epoch
    it_max = B * 100

    training_evol = []

    # Train the model
    while val > 1e-6 and it < it_max:
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
            training_evol.append(val)

            # if it > it_max:
            #     current_mean = sum(training_evol[-50:]) / 50
            #     previous_mean = sum(training_evol[-100:-50]) / 50
            #     if current_mean > previous_mean - 1e-6:

            current_mean = sum(training_evol[-10:]) / 10
            previous_mean = sum(training_evol[-20:-10]) / 10
            if current_mean > previous_mean - 1e-6:
                scheduler.step()

    plt.figure()
    plt.plot(training_evol)
    plt.show()

    torch.save(model_dir.state_dict(), 'model_2pendulum_dir_30')

    with torch.no_grad():
        X_iter_tensor = torch.Tensor(X_train_dir[:,:4]).to(device)
        y_iter_tensor = torch.Tensor(X_train_dir[:,4:]).to(device)
        outputs = model_dir(X_iter_tensor)
        print('RMSE train data: ', torch.sqrt(criterion_dir(outputs, y_iter_tensor))*std_out_dir)

    with torch.no_grad():
        X_iter_tensor = torch.Tensor(X_test_dir[:,:4]).to(device)
        y_iter_tensor = torch.Tensor(X_test_dir[:,4:]).to(device)
        outputs = model_dir(X_iter_tensor)
        print('RMSE test data: ', torch.sqrt(criterion_dir(outputs, y_iter_tensor))*std_out_dir)

    X_dir_al = np.empty((data_al.shape[0],6))

    for i in range(X_dir_al.shape[0]):
        X_dir_al[i][0] = (data_al[i][0] - mean_dir) / std_dir
        X_dir_al[i][1] = (data_al[i][1] - mean_dir) / std_dir
        vel_norm = norm([data_al[i][2],data_al[i][3]])
        if vel_norm != 0:
            X_dir_al[i][2] = data_al[i][2] / vel_norm
            X_dir_al[i][3] = data_al[i][3] / vel_norm
        X_dir_al[i][4] = vel_norm / std_out_dir
        X_dir_al[i][5] = data_al[i][5]

    correct = 0

    with torch.no_grad():
        X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
        outputs = model_dir(X_iter_tensor).cpu().numpy()
        for i in range(X_dir_al.shape[0]):
            if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
                correct += 1
            if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
                correct += 1

        print('Accuracy AL data: ', correct/X_dir_al.shape[0])

    data_boundary = np.array([data_al[i] for i in range(data_al.shape[0]) if abs(model_al((torch.Tensor(data_al[i,:4]).to(device) - mean_al) / std_al)[0]) < 5])

    X_dir_al = np.empty((data_boundary.shape[0],6))

    for i in range(X_dir_al.shape[0]):
        X_dir_al[i][0] = (data_boundary[i][0] - mean_dir) / std_dir
        X_dir_al[i][1] = (data_boundary[i][1] - mean_dir) / std_dir
        vel_norm = norm([data_boundary[i][2],data_boundary[i][3]])
        if vel_norm != 0:
            X_dir_al[i][2] = data_boundary[i][2] / vel_norm
            X_dir_al[i][3] = data_boundary[i][3] / vel_norm
        X_dir_al[i][4] = vel_norm / std_out_dir
        X_dir_al[i][5] = data_boundary[i][5]

    correct = 0

    with torch.no_grad():
        X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
        outputs = model_dir(X_iter_tensor).cpu().numpy()
        for i in range(X_dir_al.shape[0]):
            if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
                correct += 1
            if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
                correct += 1

        print('Accuracy AL data on boundary: ', correct/X_dir_al.shape[0])

    data_boundary = np.array([data_al[i] for i in range(data_al.shape[0]) if abs(model_al((torch.Tensor(data_al[i,:4]).to(device) - mean_al) / std_al)[0]) > 5])

    X_dir_al = np.empty((data_boundary.shape[0],6))

    for i in range(X_dir_al.shape[0]):
        X_dir_al[i][0] = (data_boundary[i][0] - mean_dir) / std_dir
        X_dir_al[i][1] = (data_boundary[i][1] - mean_dir) / std_dir
        vel_norm = norm([data_boundary[i][2],data_boundary[i][3]])
        if vel_norm != 0:
            X_dir_al[i][2] = data_boundary[i][2] / vel_norm
            X_dir_al[i][3] = data_boundary[i][3] / vel_norm
        X_dir_al[i][4] = vel_norm / std_out_dir
        X_dir_al[i][5] = data_boundary[i][5]

    correct = 0

    with torch.no_grad():
        X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
        outputs = model_dir(X_iter_tensor).cpu().numpy()
        for i in range(X_dir_al.shape[0]):
            if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
                correct += 1
            if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
                correct += 1

        print('Accuracy AL data not on boundary: ', correct/X_dir_al.shape[0])
    
    with torch.no_grad():
        # Plots:
        h = 0.02
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
            inp[i][4] = vel_norm / std_out_dir
        out = (model_dir(torch.from_numpy(inp[:,:4].astype(np.float32)).to(device))).cpu().numpy() 
        y_pred = np.empty(out.shape)
        for i in range(len(out)):
            if inp[i][4] > out[i]:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        xit = []
        yit = []
        for i in range(X_save.shape[0]):
            if (
                norm(X_save[i][0] - (q_min + q_max) / 2) < 0.01
                and norm(X_save[i][2]) < 0.1
            ):
                xit.append(X_save[i][1])
                yit.append(X_save[i][3])
        plt.plot(
            xit,
            yit,
            "ko",
            markersize=2
        )
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.grid()
        plt.title("Second actuator")

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
            inp[i][4] = vel_norm / std_out_dir
        out = (model_dir(torch.from_numpy(inp[:,:4].astype(np.float32)).to(device))).cpu().numpy() 
        y_pred = np.empty(out.shape)
        for i in range(len(out)):
            if inp[i][4] > out[i]:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        xit = []
        yit = []
        for i in range(X_save.shape[0]):
            if (
                norm(X_save[i][1] - (q_min + q_max) / 2) < 0.01
                and norm(X_save[i][3]) < 0.1
            ):
                xit.append(X_save[i][0])
                yit.append(X_save[i][2])
        plt.plot(
            xit,
            yit,
            "ko",
            markersize=2
        )
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.grid()
        plt.title("First actuator")

        # Plots:
        h = 0.05
        xx, yy = np.meshgrid(np.arange(v_min, v_max+h, h), np.arange(v_min, v_max+h, h))
        xrav = xx.ravel()
        yrav = yy.ravel()

        for l in range(10):
            q1ran = q_min + random.random() * (q_max-q_min)
            q2ran = q_min + random.random() * (q_max-q_min)

            # Plot the results:
            plt.figure()
            inp = np.float32(
                    np.c_[
                        q1ran * np.ones(xrav.shape[0]),
                        q2ran * np.ones(xrav.shape[0]),
                        xrav,
                        yrav,
                        np.empty(yrav.shape[0]),
                    ]
                )
            for i in range(inp.shape[0]):
                vel_norm = norm([inp[i][2],inp[i][3]])
                inp[i][0] = (inp[i][0] - mean_dir) / std_dir
                inp[i][1] = (inp[i][1] - mean_dir) / std_dir
                if vel_norm != 0:
                    inp[i][2] = inp[i][2] / vel_norm
                    inp[i][3] = inp[i][3] / vel_norm
                inp[i][4] = vel_norm / std_out_dir
            out = (model_dir(torch.Tensor(inp[:,:4]).to(device))).cpu().numpy() 
            y_pred = np.empty(out.shape)
            for i in range(len(out)):
                if inp[i][4] > out[i]:
                    y_pred[i] = 0
                else:
                    y_pred[i] = 1
            Z = y_pred.reshape(yy.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
            xit = []
            yit = []
            for i in range(X_save.shape[0]):
                if (
                    norm(X_save[i][0] - q1ran) < 0.01
                    and norm(X_save[i][1] - q2ran) < 0.01
                ):
                    xit.append(X_save[i][2])
                    yit.append(X_save[i][3])
            plt.plot(
                xit,
                yit,
                "ko",
                markersize=2
            )
            plt.xlim([v_min, v_max])
            plt.ylim([v_min, v_max])
            plt.grid()
            plt.title("q1="+str(q1ran)+" q2="+str(q2ran))

    print("Execution time: %s seconds" % (time.time() - start_time))

    plt.show()
