import numpy as np
import random 
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
from triplependulum_class_fixedveldir import OCPtriplependulumINIT
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from my_nn import NeuralNet, NeuralNetRegression
import math
from multiprocessing import Pool
from torch.utils.data import DataLoader

def testing(v):
    # Reset the number of steps used in the OCP:
    N = ocp.N
    ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
    ocp.ocp_solver.update_qp_solver_cond_N(N)

    # Time step duration:
    dt_sym = 1e-2

    # Initial velocity optimization direction (the cost is in the form p[0] * dtheta1 + p[1] * dtheta2 + p[2] * dt):
    ran1 = random.choice([-1, 1]) * random.random()
    ran2 = random.choice([-1, 1]) * random.random() 
    ran3 = random.choice([-1, 1]) * random.random() 
    norm_weights = norm(np.array([ran1, ran2, ran3]))         
    p = np.array([ran1/norm_weights, ran2/norm_weights, ran3/norm_weights, 0.])

    # Bounds on the initial state:
    q_init_1 = q_min + random.random() * (q_max-q_min)
    q_init_2 = q_min + random.random() * (q_max-q_min)
    q_init_3 = q_min + random.random() * (q_max-q_min)
    q_init_lb = np.array([q_init_1, q_init_2, q_init_3, v_min, v_min, v_min, dt_sym])
    q_init_ub = np.array([q_init_1, q_init_2, q_init_3, v_max, v_max, v_max, dt_sym])

    # Bounds on the final state:
    q_fin_lb = np.array([q_min, q_min, q_min, 0., 0., 0., dt_sym])
    q_fin_ub = np.array([q_max, q_max, q_max, 0., 0., 0., dt_sym])

    # Guess:
    x_sol_guess = np.full((N, 7), np.array([q_init_1, q_init_2, q_init_3, 0., 0., 0., dt_sym]))
    u_sol_guess = np.full((N, 3), np.array([0.,0.,0.]))

    # Iteratively solve the OCP with an increased number of time steps until the solution converges:
    cost = 1e6
    while True:
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

        # If the solver finds a solution, compare it with the previous. If the cost has decresed, keep increasing N, alternatively keep increasing N.
        # If the solver fails, reinitialize N and restart the iterations with slight different initial conditions.
        if status == 0: 
            # Compare the current cost with the previous:
            cost_new = ocp.ocp_solver.get_cost()
            if cost_new > float(f'{cost:.3f}') - 1e-3:
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

            # Initial velocity optimization direction:
            ran1 = ran1 + random.random() * random.choice([-1, 1]) * 0.01
            ran2 = ran2 + random.random() * random.choice([-1, 1]) * 0.01
            ran3 = ran3 + random.random() * random.choice([-1, 1]) * 0.01
            norm_weights = norm(np.array([ran1, ran2, ran3]))         
            p = np.array([ran1/norm_weights, ran2/norm_weights, ran3/norm_weights, 0.])
            
            # Bounds on the initial state:
            q_init_1 = q_init_1 + random.random() * random.choice([-1, 1]) * 0.01
            q_init_2 = q_init_2 + random.random() * random.choice([-1, 1]) * 0.01
            q_init_3 = q_init_3 + random.random() * random.choice([-1, 1]) * 0.01
            q_init_lb = np.array([q_init_1, q_init_2, q_init_3, v_min, v_min, v_min, dt_sym])
            q_init_ub = np.array([q_init_1, q_init_2, q_init_3, v_max, v_max, v_max, dt_sym])

            # Guess:
            x_sol_guess = np.full((N, 7), np.array([q_init_1, q_init_2, q_init_3, 0., 0., 0., dt_sym]))
            u_sol_guess = np.full((N, 3), np.array([0.,0.,0.]))

            cost = 1e6

    return ocp.ocp_solver.get(0, "x")[:6]

# Ocp initialization:
ocp = OCPtriplependulumINIT()

# Position, velocity and torque bounds:
v_max = ocp.dthetamax
v_min = - ocp.dthetamax
q_max = ocp.thetamax
q_min = ocp.thetamin
tau_max = ocp.Cmax

n_minibatch_model = pow(2,15)

# Pytorch device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Test data generation:
# cpu_num = 30
# num_prob = 1000

# with Pool(cpu_num) as p:
#     data = p.map(testing, range(num_prob))

# X_test = np.array(data)
# np.save('data3_test_10.npy', X_test)

X_test = np.load('data3_test_10.npy')

# VBOC:
model_dir = NeuralNetRegression(6, 500, 1).to(device)
criterion_dir = nn.MSELoss()
model_dir.load_state_dict(torch.load('model_3dof_vboc_10_14400s'))
data_reverse = np.load('data_3dof_vboc_10_14400.npy')
# mean_dir, std_dir = torch.mean(torch.tensor(data_reverse[:,:2].tolist())).to(device).item(), torch.std(torch.tensor(data_reverse[:,:2].tolist())).to(device).item()
mean_dir = torch.load('mean_3dof_vboc_10')
std_dir = torch.load('std_3dof_vboc_10')

# Active Learning:
model_al = NeuralNet(6, 500, 2).to(device)
model_al.load_state_dict(torch.load('AL/model_3dof_al_10_14400s'))
mean_al = torch.load('AL/mean_3dof_al_10')
std_al = torch.load('AL/std_3dof_al_10')
data_al = np.load('AL/data_3dof_al_10_14400s.npy')

# RMSE evolutions:
times_al = np.load('AL/times_3dof_al.npy')
rmse_al = np.load('AL/rmse_3dof_al.npy')
times_vbocp = np.load('times_3dof_vbocp.npy')
rmse_vbocp = np.load('rmse_3dof_vbocp.npy')

plt.figure(figsize=(6, 4))
plt.plot(times_vbocp, rmse_vbocp, label='VBOC')
plt.plot(times_al, rmse_al, label='AL')
plt.title('RMSE evolution')
plt.legend(loc='upper right')
plt.ylabel('RMSE (rad/s)')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.grid(True, which="both")

# Compute the prediction errors over the test data:

X_training_dir = np.empty((data_reverse.shape[0],7))
for i in range(X_training_dir.shape[0]):
    X_training_dir[i][0] = (data_reverse[i][0] - mean_dir) / std_dir
    X_training_dir[i][1] = (data_reverse[i][1] - mean_dir) / std_dir
    X_training_dir[i][2] = (data_reverse[i][2] - mean_dir) / std_dir
    vel_norm = norm([data_reverse[i][4],data_reverse[i][3],data_reverse[i][5]])
    if vel_norm != 0:
        X_training_dir[i][5] = data_reverse[i][5] / vel_norm
        X_training_dir[i][4] = data_reverse[i][4] / vel_norm
        X_training_dir[i][3] = data_reverse[i][3] / vel_norm
    X_training_dir[i][6] = vel_norm 

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
    print('RMSE test data wrt VBOCP NN: ', rmse_vbocp[-1:])
    print('RMSE test data wrt AL NN: ', rmse_al[-1:])

    X_iter_tensor = torch.Tensor(X_training_dir[:,:6]).to(device)
    out = np.empty((len(X_training_dir),1))
    my_dataloader = DataLoader(X_iter_tensor,batch_size=n_minibatch_model,shuffle=False)
    for (idx, batch) in enumerate(my_dataloader):
        if n_minibatch_model*(idx+1) > len(X_training_dir):
            out[n_minibatch_model*idx:len(X_training_dir)] = model_dir(batch).cpu()
        else:
            out[n_minibatch_model*idx:n_minibatch_model*(idx+1)] = model_dir(batch).cpu()

    print('RMSE train data wrt VBOCP NN: ', math.sqrt(np.sum([(out[i] - X_training_dir[i,6])**2 for i in range(len(X_training_dir))])/len(X_training_dir)))

    X_iter_tensor = torch.Tensor(X_test_dir[:,:6]).to(device)
    out = np.empty((len(X_test_dir),1))
    my_dataloader = DataLoader(X_iter_tensor,batch_size=n_minibatch_model,shuffle=False)
    for (idx, batch) in enumerate(my_dataloader):
        if n_minibatch_model*(idx+1) > len(X_test_dir):
            out[n_minibatch_model*idx:len(X_test_dir)] = model_dir(batch).cpu()
        else:
            out[n_minibatch_model*idx:n_minibatch_model*(idx+1)] = model_dir(batch).cpu()

    print('RMSE test data wrt VBOCP NN: ', math.sqrt(np.sum([(out[i] - X_test_dir[i,6])**2 for i in range(len(X_test_dir))])/len(X_test_dir)))

plt.show()