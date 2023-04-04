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
    # Reset the number of steps used in the OCP:
    N = ocp.N
    ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
    ocp.ocp_solver.update_qp_solver_cond_N(N)

    # Time step duration:
    dt_sym = 1e-2

    # Initial velocity optimization direction (the cost is in the form p[0] * dtheta1 + p[1] * dtheta2 + p[2] * dt):
    ran1 = random.choice([-1, 1]) * random.random()
    ran2 = random.choice([-1, 1]) * random.random() 
    norm_weights = norm(np.array([ran1, ran2]))         
    p = np.array([ran1/norm_weights, ran2/norm_weights, 0.])

    # Bounds on the initial state:
    q_init_1 = q_min + random.random() * (q_max-q_min)
    q_init_2 = q_min + random.random() * (q_max-q_min)
    q_init_lb = np.array([q_init_1, q_init_2, v_min, v_min, dt_sym])
    q_init_ub = np.array([q_init_1, q_init_2, v_max, v_max, dt_sym])

    # Bounds on the final state:
    q_fin_lb = np.array([q_min, q_min, 0., 0., dt_sym])
    q_fin_ub = np.array([q_max, q_max, 0., 0., dt_sym])

    # Guess:
    x_sol_guess = np.full((N, 5), np.array([q_init_1, q_init_2, 0., 0., dt_sym]))
    u_sol_guess = np.full((N, 2), np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(q_init_1),ocp.g*ocp.l2*ocp.m2*math.sin(q_init_2)]))

    cost = 1e6

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
            if cost_new > float(f'{cost:.4f}') - 1e-4:
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
            # Reset the number of steps used in the OCP:
            N = ocp.N
            ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
            ocp.ocp_solver.update_qp_solver_cond_N(N)

            # Time step duration:
            dt_sym = 1e-2

            # Initial velocity optimization direction (the cost is in the form p[0] * dtheta1 + p[1] * dtheta2 + p[2] * dt):
            ran1 = ran1 + random.random() * random.choice([-1, 1]) * 0.01
            ran2 = ran2 + random.random() * random.choice([-1, 1]) * 0.01
            norm_weights = norm(np.array([ran1, ran2]))         
            p = np.array([ran1/norm_weights, ran2/norm_weights, 0.])
            
            # Bounds on the initial state:
            q_init_1 = q_init_1 + random.random() * random.choice([-1, 1]) * 0.01
            q_init_2 = q_init_2 + random.random() * random.choice([-1, 1]) * 0.01
            q_init_lb = np.array([q_init_1, q_init_2, v_min, v_min, dt_sym])
            q_init_ub = np.array([q_init_1, q_init_2, v_max, v_max, dt_sym])

            # Guess:
            x_sol_guess = np.full((N, 5), np.array([q_init_1, q_init_2, 0., 0., dt_sym]))
            u_sol_guess = np.full((N, 2), np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(q_init_1),ocp.g*ocp.l2*ocp.m2*math.sin(q_init_2)]))

            cost = 1e6

    return ocp.ocp_solver.get(0, "x")[:4]

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

cpu_num = 30

num_prob = 1000

# # Data generation:
# with Pool(cpu_num) as p:
#     data = p.map(testing, range(num_prob))

# X_test = np.array(data)
# np.save('data_test_20.npy', X_test)

X_test = np.load('data_test_20.npy')

# Reverse OCP model and data:
model_dir = NeuralNetRegression(4, 400, 1).to(device)
criterion_dir = nn.MSELoss()
model_dir.load_state_dict(torch.load('data_rt/model_2pendulum_dir_20'))
data_reverse = np.load('data_rt/data_reverse_20.npy')
mean_dir, std_dir = torch.mean(torch.tensor(data_reverse[:,:2].tolist())).to(device).item(), torch.std(torch.tensor(data_reverse[:,:2].tolist())).to(device).item()

# # Reverse OCP model with only x0 and data:
# model_dir_x0 = NeuralNetRegression(4, 400, 1).to(device)
# criterion_dir = nn.MSELoss()
# model_dir_x0.load_state_dict(torch.load('data_rt/model_2pendulum_dir_5_x0'))
# data_reverse_x0 = np.load('data_rt/data_reverse_5_x0.npy')
# mean_dir_x0, std_dir_x0 = torch.mean(torch.tensor(data_reverse_x0[:,:2].tolist())).to(device).item(), torch.std(torch.tensor(data_reverse_x0[:,:2].tolist())).to(device).item()

# Active Learning model and data:
model_al = NeuralNet(4, 400, 2).to(device)
model_al.load_state_dict(torch.load('data_al/model_2pendulum_20'))
mean_al = torch.load('data_al/mean_2pendulum_20')
std_al = torch.load('data_al/std_2pendulum_20')
data_al = np.load('data_al/data_al_20.npy')

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
    output_al_test = np.argmax(model_al((torch.Tensor(X_test).to(device) - mean_al) / std_al).cpu().numpy(), axis=1)

data_neg = np.array([X_test[i] for i in range(X_test.shape[0]) if output_al_test[i] == 0])
data_pos = np.array([X_test[i] for i in range(X_test.shape[0]) if output_al_test[i] == 1])

norm_error_neg = np.empty((len(data_neg),))
norm_relerror_neg = np.empty((len(data_neg),))

for i in range(len(data_neg)):
    vel_norm = norm([data_neg[i][2],data_neg[i][3]])
    v0 = data_neg[i][2]
    v1 = data_neg[i][3]

    out = 0
    while out == 0 and norm([v0,v1]) > 1e-2:
        v0 = v0 - 1e-2 * data_neg[i][2]/vel_norm
        v1 = v1 - 1e-2 * data_neg[i][3]/vel_norm
        out = np.argmax(model_al((torch.Tensor([[data_neg[i][0], data_neg[i][1], v0, v1]]).to(device) - mean_al) / std_al).cpu().detach().numpy(), axis=1)

    norm_error_neg[i] = vel_norm - norm([v0,v1])
    norm_relerror_neg[i] = norm_error_neg[i]/vel_norm

norm_error_pos = np.empty((len(data_pos),))
norm_relerror_pos = np.empty((len(data_pos),))

for i in range(len(data_pos)):
    vel_norm = norm([data_pos[i][2],data_pos[i][3]])
    v0 = data_pos[i][2]
    v1 = data_pos[i][3]

    out = 1
    while out == 1 and norm([v0,v1]) > 1e-2:
        v0 = v0 + 1e-2 * data_pos[i][2]/vel_norm
        v1 = v1 + 1e-2 * data_pos[i][3]/vel_norm
        out = np.argmax(model_al((torch.Tensor([[data_pos[i][0], data_pos[i][1], v0, v1]]).to(device) - mean_al) / std_al).cpu().detach().numpy(), axis=1)

    norm_error_pos[i] = vel_norm - norm([v0,v1]) 
    norm_relerror_pos[i] = norm_error_pos[i]/vel_norm

norm_error_al = np.concatenate((norm_error_neg, norm_error_pos))
norm_relerror_al = np.concatenate((norm_relerror_neg, norm_relerror_pos))

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_test_dir[:,:4]).to(device)
    y_iter_tensor = torch.Tensor(X_test_dir[:,4:]).to(device)
    outputs = model_dir(X_iter_tensor).cpu().numpy()
    print('RRMSE test data wrt new NN in %: ', math.sqrt(np.sum([((outputs[i] - X_test_dir[i,4])/X_test_dir[i,4])**2 for i in range(len(X_test_dir))])/len(X_test_dir))*100)
    print('RMSE test data wrt new NN: ', torch.sqrt(criterion_dir(model_dir(X_iter_tensor), y_iter_tensor)))
    # outputs = model_dir_x0(X_iter_tensor).cpu().numpy()
    # print('RRMSE test data wrt new NN with only x0 in %: ', math.sqrt(np.sum([((outputs[i] - X_test_dir[i,4])/X_test_dir[i,4])**2 for i in range(len(X_test_dir))])/len(X_test_dir))*100)
    # print('RMSE test data wrt new NN with only x0: ', torch.sqrt(criterion_dir(model_dir_x0(X_iter_tensor), y_iter_tensor)))

with torch.no_grad():
    print('RRMSE test data wrt AL NN in %: ', math.sqrt(np.sum(np.power(norm_relerror_al,2))/len(norm_relerror_al))*100)
    print('RMSE test data wrt AL NN: ', math.sqrt(np.sum(np.power(norm_error_al,2))/len(norm_error_al)))

plt.figure()
norm_error_rt=[X_test_dir[i,4] - outputs[i].tolist()[0] for i in range(len(X_test_dir))]
bins = np.linspace(-2, 2, 100)
plt.hist(norm_error_rt, bins, alpha=0.5, label='Reverse-time NN. Error distribution')
plt.hist(norm_error_al, bins, alpha=0.5, label='Active Learning NN. Error distribution')
plt.legend(loc='upper right')

plt.figure()
norm_relerror_rt=[(X_test_dir[i,4] - outputs[i].tolist()[0])/X_test_dir[i,4] for i in range(len(X_test_dir))]
bins = np.linspace(-1., 1., 100)
plt.hist(norm_relerror_rt, bins, alpha=0.5, label='Reverse-time NN. Relative error distribution')
plt.hist(norm_relerror_al, bins, alpha=0.5, label='Active Learning NN. Relative error distribution')
plt.legend(loc='upper right')

# X_dir_al = np.empty((data_al.shape[0],6))

# for i in range(X_dir_al.shape[0]):
#     X_dir_al[i][0] = (data_al[i][0] - mean_dir) / std_dir
#     X_dir_al[i][1] = (data_al[i][1] - mean_dir) / std_dir
#     vel_norm = norm([data_al[i][2],data_al[i][3]])
#     if vel_norm != 0:
#         X_dir_al[i][2] = data_al[i][2] / vel_norm
#         X_dir_al[i][3] = data_al[i][3] / vel_norm
#     X_dir_al[i][4] = vel_norm 
#     X_dir_al[i][5] = data_al[i][5]

# correct = 0

# with torch.no_grad():
#     X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
#     outputs = model_dir(X_iter_tensor).cpu().numpy()
#     for i in range(X_dir_al.shape[0]):
#         if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
#             correct += 1
#         if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
#             correct += 1

#     print('Accuracy AL data: ', correct/X_dir_al.shape[0])

# data_internal = np.array([data_al[i] for i in range(data_al.shape[0]) if model_al((torch.Tensor(data_al[i,:4]).to(device) - mean_al) / std_al)[1]>0])

# X_dir_al = np.empty((data_internal.shape[0],6))

# for i in range(X_dir_al.shape[0]):
#     X_dir_al[i][0] = (data_internal[i][0] - mean_dir) / std_dir
#     X_dir_al[i][1] = (data_internal[i][1] - mean_dir) / std_dir
#     vel_norm = norm([data_internal[i][2],data_internal[i][3]])
#     if vel_norm != 0:
#         X_dir_al[i][2] = data_internal[i][2] / vel_norm
#         X_dir_al[i][3] = data_internal[i][3] / vel_norm
#     X_dir_al[i][4] = vel_norm 
#     X_dir_al[i][5] = data_internal[i][5]

# correct = 0

# with torch.no_grad():
#     X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
#     outputs = model_dir(X_iter_tensor).cpu().numpy()
#     for i in range(X_dir_al.shape[0]):
#         if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
#             correct += 1
#         if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
#             correct += 1

#     print('Accuracy AL internal data: ', correct/X_dir_al.shape[0])

# data_boundary = np.array([data_al[i] for i in range(data_al.shape[0]) if abs(model_al((torch.Tensor(data_al[i,:4]).to(device) - mean_al) / std_al)[0]) < 10])

# X_dir_al = np.empty((data_boundary.shape[0],6))

# for i in range(X_dir_al.shape[0]):
#     X_dir_al[i][0] = (data_boundary[i][0] - mean_dir) / std_dir
#     X_dir_al[i][1] = (data_boundary[i][1] - mean_dir) / std_dir
#     vel_norm = norm([data_boundary[i][2],data_boundary[i][3]])
#     if vel_norm != 0:
#         X_dir_al[i][2] = data_boundary[i][2] / vel_norm
#         X_dir_al[i][3] = data_boundary[i][3] / vel_norm
#     X_dir_al[i][4] = vel_norm 
#     X_dir_al[i][5] = data_boundary[i][5]

# correct = 0

# with torch.no_grad():
#     X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
#     outputs = model_dir(X_iter_tensor).cpu().numpy()
#     for i in range(X_dir_al.shape[0]):
#         if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
#             correct += 1
#         if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
#             correct += 1

#     print('Accuracy AL data on boundary: ', correct/X_dir_al.shape[0])

# data_notboundary = np.array([data_al[i] for i in range(data_al.shape[0]) if abs(model_al((torch.Tensor(data_al[i,:4]).to(device) - mean_al) / std_al)[0]) > 10])

# X_dir_al = np.empty((data_notboundary.shape[0],6))

# for i in range(X_dir_al.shape[0]):
#     X_dir_al[i][0] = (data_notboundary[i][0] - mean_dir) / std_dir
#     X_dir_al[i][1] = (data_notboundary[i][1] - mean_dir) / std_dir
#     vel_norm = norm([data_notboundary[i][2],data_notboundary[i][3]])
#     if vel_norm != 0:
#         X_dir_al[i][2] = data_notboundary[i][2] / vel_norm
#         X_dir_al[i][3] = data_notboundary[i][3] / vel_norm
#     X_dir_al[i][4] = vel_norm 
#     X_dir_al[i][5] = data_notboundary[i][5]

# correct = 0

# with torch.no_grad():
#     X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
#     outputs = model_dir(X_iter_tensor).cpu().numpy()
#     for i in range(X_dir_al.shape[0]):
#         if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
#             correct += 1
#         if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
#             correct += 1

#     print('Accuracy AL data not on boundary: ', correct/X_dir_al.shape[0])

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
    xit = []
    yit = []
    for i in range(X_test.shape[0]):
        if (
            norm(X_test[i][0] - (q_min + q_max) / 2) < 0.01
            and norm(X_test[i][2]) < 0.1
        ):
            xit.append(X_test[i][1])
            yit.append(X_test[i][3])
    plt.plot(
        xit,
        yit,
        "ko",
        markersize=2
    )
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.grid()
    plt.title("Second actuator, RT")

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
    xit = []
    yit = []
    for i in range(X_test.shape[0]):
        if (
            norm(X_test[i][1] - (q_min + q_max) / 2) < 0.01
            and norm(X_test[i][3]) < 0.1
        ):
            xit.append(X_test[i][0])
            yit.append(X_test[i][2])
    plt.plot(
        xit,
        yit,
        "ko",
        markersize=2
    )
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.grid()
    plt.title("First actuator, RT")

    # plt.figure()
    # inp = np.c_[
    #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
    #             xrav,
    #             np.zeros(yrav.shape[0]),
    #             yrav,
    #             np.empty(yrav.shape[0]),
    #         ]
    # for i in range(inp.shape[0]):
    #     vel_norm = norm([inp[i][2],inp[i][3]])
    #     inp[i][0] = (inp[i][0] - mean_dir_x0) / std_dir_x0
    #     inp[i][1] = (inp[i][1] - mean_dir_x0) / std_dir_x0
    #     if vel_norm != 0:
    #         inp[i][2] = inp[i][2] / vel_norm
    #         inp[i][3] = inp[i][3] / vel_norm
    #     inp[i][4] = vel_norm
    # out = (model_dir_x0(torch.from_numpy(inp[:,:4].astype(np.float32)).to(device))).cpu().numpy() 
    # y_pred = np.empty(out.shape)
    # for i in range(len(out)):
    #     if inp[i][4] > out[i]:
    #         y_pred[i] = 0
    #     else:
    #         y_pred[i] = 1
    # Z = y_pred.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
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
    # plt.xlim([q_min, q_max])
    # plt.ylim([v_min, v_max])
    # plt.grid()
    # plt.title("Second actuator, RT x0")

    # plt.figure()
    # inp = np.c_[
    #             xrav,
    #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
    #             yrav,
    #             np.zeros(yrav.shape[0]),
    #             np.empty(yrav.shape[0]),
    #         ]
    # for i in range(inp.shape[0]):
    #     vel_norm = norm([inp[i][2],inp[i][3]])
    #     inp[i][0] = (inp[i][0] - mean_dir_x0) / std_dir_x0
    #     inp[i][1] = (inp[i][1] - mean_dir_x0) / std_dir_x0
    #     if vel_norm != 0:
    #         inp[i][2] = inp[i][2] / vel_norm
    #         inp[i][3] = inp[i][3] / vel_norm
    #     inp[i][4] = vel_norm
    # out = (model_dir_x0(torch.from_numpy(inp[:,:4].astype(np.float32)).to(device))).cpu().numpy() 
    # y_pred = np.empty(out.shape)
    # for i in range(len(out)):
    #     if inp[i][4] > out[i]:
    #         y_pred[i] = 0
    #     else:
    #         y_pred[i] = 1
    # Z = y_pred.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
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
    # plt.xlim([q_min, q_max])
    # plt.ylim([v_min, v_max])
    # plt.grid()
    # plt.title("First actuator, RT x0")

    # Plot the results:
    plt.figure()
    inp = torch.from_numpy(
        np.float32(
            np.c_[
                (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                xrav,
                np.zeros(yrav.shape[0]),
                yrav,
            ]
        )
    ).to(device)
    inp = (inp - mean_al) / std_al
    out = model_al(inp)
    y_pred = np.argmax(out.cpu().numpy(), axis=1)
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    for i in range(X_test.shape[0]):
        if (
            norm(X_test[i][0] - (q_min + q_max) / 2) < 0.01
            and norm(X_test[i][2]) < 0.1
        ):
            xit.append(X_test[i][1])
            yit.append(X_test[i][3])
    plt.plot(
        xit,
        yit,
        "ko",
        markersize=2
    )
    xit = []
    yit = []
    cit = []
    for i in range(len(data_al)):
        if (
            norm(data_al[i][0] - (q_min + q_max) / 2) < 0.1
            and norm(data_al[i][2]) < 0.1
        ):
            xit.append(data_al[i][1])
            yit.append(data_al[i][3])
            if data_al[i][4] == 1:
                cit.append(0)
            else:
                cit.append(1)
    plt.scatter(
        xit,
        yit,
        c=cit,
        marker=".",
        alpha=0.5,
        cmap=plt.cm.Paired,
    )
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.grid()
    plt.title("Second actuator, AL")

    plt.figure()
    inp = torch.from_numpy(
        np.float32(
            np.c_[
                xrav,
                (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                yrav,
                np.zeros(yrav.shape[0]),
            ]
        )
    ).to(device)
    inp = (inp - mean_al) / std_al
    out = model_al(inp)
    y_pred = np.argmax(out.cpu().numpy(), axis=1)
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    for i in range(X_test.shape[0]):
        if (
            norm(X_test[i][1] - (q_min + q_max) / 2) < 0.01
            and norm(X_test[i][3]) < 0.1
        ):
            xit.append(X_test[i][0])
            yit.append(X_test[i][2])
    plt.plot(
        xit,
        yit,
        "ko",
        markersize=2
    )
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.grid()
    plt.title("First actuator, AL")

    # Plots:
    h = 0.05
    xx, yy = np.meshgrid(np.arange(v_min, v_max+h, h), np.arange(v_min, v_max+h, h))
    xrav = xx.ravel()
    yrav = yy.ravel()

    # for l in range(5):
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
    #     for i in range(X_test.shape[0]):
    #         if (
    #             norm(X_test[i][0] - q1ran) < 0.01
    #             and norm(X_test[i][1] - q2ran) < 0.01
    #         ):
    #             xit.append(X_test[i][2])
    #             yit.append(X_test[i][3])
    #     plt.plot(
    #         xit,
    #         yit,
    #         "ko",
    #         markersize=2
    #     )
    #     plt.xlim([v_min, v_max])
    #     plt.ylim([v_min, v_max])
    #     plt.grid()
    #     plt.title("q1="+str(q1ran)+" q2="+str(q2ran)+" RT")

    #     plt.figure()
    #     inp = torch.from_numpy(
    #         np.float32(
    #             np.c_[
    #                 q1ran * np.ones(xrav.shape[0]),
    #                 q2ran * np.ones(xrav.shape[0]),
    #                 xrav,
    #                 yrav, 
    #             ]
    #         )
    #     ).to(device)
    #     inp = (inp - mean_al) / std_al
    #     out = model_al(inp)
    #     y_pred = np.argmax(out.cpu().numpy(), axis=1)
    #     Z = y_pred.reshape(xx.shape)
    #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #     xit = []
    #     yit = []
    #     for i in range(X_test.shape[0]):
    #         if (
    #             norm(X_test[i][0] - q1ran) < 0.01
    #             and norm(X_test[i][1] - q2ran) < 0.01
    #         ):
    #             xit.append(X_test[i][2])
    #             yit.append(X_test[i][3])
    #     plt.plot(
    #         xit,
    #         yit,
    #         "ko",
    #         markersize=2
    #     )
    #     plt.xlim([v_min, v_max])
    #     plt.ylim([v_min, v_max])
    #     plt.grid()
    #     plt.title("q1="+str(q1ran)+" q2="+str(q2ran)+" AL")

plt.show()