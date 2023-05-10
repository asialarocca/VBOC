import os
import sys
sys.path.insert(1, os.getcwd() + '/VBOC')

import numpy as np
import matplotlib.pyplot as plt
from pendulum_class_fixedveldir import OCPpendulum
import random
import torch
import torch.nn as nn
from my_nn import NeuralNet, NeuralNetRegression
import math
from numpy.linalg import norm as norm
from multiprocessing import Pool

def testing(v):
    # Reset the number of steps used in the OCP:
    N = ocp.N

    dt_sym = 1e-2

    # Initial velocity optimization direction (the cost is in the form p[0] * dtheta1 + p[1] * dtheta2 + p[2] * dt):
    ran = random.choice([-1, 1])   
    p = np.array([ran, 0.])

    # Bounds on the initial state:
    q_init = q_min + random.random() * (q_max-q_min)
    q_init_lb = np.array([q_init, v_min, dt_sym])
    q_init_ub = np.array([q_init, v_max, dt_sym])

    # Bounds on the final state:
    q_fin_lb = np.array([q_min, 0., dt_sym])
    q_fin_ub = np.array([q_max, 0., dt_sym])

    # Guess:
    x_sol_guess = np.full((N, 3), np.array([q_init, 0., dt_sym]))

    # Reset current iterate:
    ocp.ocp_solver.reset()

    # Set parameters, guesses and constraints:
    for i in range(N):
        ocp.ocp_solver.set(i, 'x', x_sol_guess[i])
        ocp.ocp_solver.set(i, 'p', p)
        ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, v_min, dt_sym])) 
        ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, v_max, dt_sym])) 

    ocp.ocp_solver.constraints_set(0, "lbx", q_init_lb)
    ocp.ocp_solver.constraints_set(0, "ubx", q_init_ub)

    ocp.ocp_solver.constraints_set(N, "lbx", q_fin_lb)
    ocp.ocp_solver.constraints_set(N, "ubx", q_fin_ub)
    ocp.ocp_solver.set(N, 'x', x_sol_guess[-1])
    ocp.ocp_solver.set(N, 'p', p)

    # Solve the OCP:
    status = ocp.ocp_solver.solve()

    if status == 0: 
        return ocp.ocp_solver.get(0, "x")[:2]
    else:
        return None

# Ocp initialization:
ocp = OCPpendulum()

# Position, velocity and torque bounds:
v_max = ocp.dthetamax
v_min = - ocp.dthetamax
q_max = ocp.thetamax
q_min = ocp.thetamin

# Pytorch device:
device = torch.device("cpu")

# # Test data generation:
# cpu_num = 30
# num_prob = 1000

# with Pool(cpu_num) as p:
#     data = p.map(testing, range(num_prob))

# X_test = np.array([i for i in data if i is not None])
# np.save('data1_test_10.npy', X_test)

X_test = np.load('data1_test_10.npy')

# Upload the models and training data of the different approaches:
# VBOCP:
model_dir = NeuralNetRegression(2, 100, 1).to(device)
criterion_dir = nn.MSELoss()
model_dir.load_state_dict(torch.load('VBOC/model_1dof_vboc_10'))
data_reverse = np.load('VBOC/data_1dof_vboc_10.npy')
mean_dir, std_dir = torch.mean(torch.tensor(data_reverse[:,:1].tolist())).to(device).item(), torch.std(torch.tensor(data_reverse[:,:1].tolist())).to(device).item()

# Active Learning:
model_al = NeuralNet(2, 100, 2).to(device)
model_al.load_state_dict(torch.load('AL/model_1pendulum_al_10'))
mean_al = torch.load('AL/mean_1dof_al_10')
std_al = torch.load('AL/std_1dof_al_10')
data_al = np.load('AL/data_1dof_al_10.npy')

# HJ Reachability:
model_hjr = NeuralNet(2, 100, 2).to(device)
model_hjr.load_state_dict(torch.load('HJR/model_1dof_hjr_10'))
mean_hjr = torch.load('HJR/mean_1dof_hjr_10')
std_hjr = torch.load('HJR/std_1dof_hjr_10')

X_test = np.load('data1_test_10.npy')
X_test_dir = np.empty((X_test.shape[0],3))
for i in range(X_test_dir.shape[0]):
    X_test_dir[i][0] = (X_test[i][0] - mean_dir) / std_dir
    vel_norm = abs(X_test[i][1])
    if vel_norm != 0:
        X_test_dir[i][1] = X_test[i][1] / vel_norm
    X_test_dir[i][2] = vel_norm 

# Compute the prediction errors over the test data:

# VBOC:
X_test_dir = np.empty((X_test.shape[0],3))
for i in range(X_test_dir.shape[0]):
    X_test_dir[i][0] = (X_test[i][0] - mean_dir) / std_dir
    vel_norm = abs(X_test[i][1])
    if vel_norm != 0:
        X_test_dir[i][1] = X_test[i][1] / vel_norm
    X_test_dir[i][2] = vel_norm 

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_test_dir[:,:2]).to(device)
    y_iter_tensor = torch.Tensor(X_test_dir[:,2:]).to(device)
    outputs = model_dir(X_iter_tensor).cpu().numpy()
    print('RRMSE test data wrt VBOCP NN in %: ', math.sqrt(np.sum([((outputs[i] - X_test_dir[i,2])/X_test_dir[i,2])**2 for i in range(len(X_test_dir))])/len(X_test_dir))*100)
    print('RMSE test data wrt VBOCP NN: ', torch.sqrt(criterion_dir(model_dir(X_iter_tensor), y_iter_tensor)))

# Active Learning:
output_al_test = np.argmax(model_al((torch.Tensor(X_test).to(device) - mean_al) / std_al).cpu().detach().numpy(), axis=1)
norm_error_al = np.empty((len(X_test),))
norm_relerror_al = np.empty((len(X_test),))
absnorm_error_al = np.empty((len(X_test),))
absnorm_relerror_al = np.empty((len(X_test),))
for i in range(len(X_test)):
    vel_norm = abs(X_test[i][1])
    v0 = X_test[i][1]

    if output_al_test[i] == 0:
        out = 0
        while out == 0 and abs(v0) > 1e-2:
            v0 = v0 - 1e-2 * X_test[i][1]/vel_norm
            out = np.argmax(model_al((torch.Tensor([[X_test[i][0], v0]]).to(device) - mean_al) / std_al).cpu().detach().numpy(), axis=1)
    else:
        out = 1
        while out == 1 and abs(v0) > 1e-2:
            v0 = v0 + 1e-2 * X_test[i][1]/vel_norm
            out = np.argmax(model_al((torch.Tensor([[X_test[i][0], v0]]).to(device) - mean_al) / std_al).cpu().detach().numpy(), axis=1)

    norm_error_al[i] = vel_norm - abs(v0)
    norm_relerror_al[i] = norm_error_al[i]/vel_norm
    absnorm_error_al[i] = abs(norm_error_al[i])
    absnorm_relerror_al[i] = abs(norm_relerror_al[i])

with torch.no_grad():
    print('RRMSE test data wrt AL NN in %: ', math.sqrt(np.sum(np.power(norm_relerror_al,2))/len(norm_relerror_al))*100)
    print('RMSE test data wrt AL NN: ', math.sqrt(np.sum(np.power(norm_error_al,2))/len(norm_error_al)))

# HJ Reachability:
output_hjr_test = np.argmax(model_hjr((torch.Tensor(X_test).to(device) - mean_hjr) / std_hjr).cpu().detach().numpy(), axis=1)
norm_error_hjr = np.empty((len(X_test),))
norm_relerror_hjr = np.empty((len(X_test),))
absnorm_error_hjr = np.empty((len(X_test),))
absnorm_relerror_hjr = np.empty((len(X_test),))
for i in range(len(X_test)):
    vel_norm = abs(X_test[i][1])
    v0 = X_test[i][1]

    if output_al_test[i] == 0:
        out = 0
        while out == 0 and abs(v0) > 1e-2:
            v0 = v0 - 1e-2 * X_test[i][1]/vel_norm
            out = np.argmax(model_hjr((torch.Tensor([[X_test[i][0], v0]]).to(device) - mean_hjr) / std_hjr).cpu().detach().numpy(), axis=1)
    else:
        out = 1
        while out == 1 and abs(v0) > 1e-2:
            v0 = v0 + 1e-2 * X_test[i][1]/vel_norm
            out = np.argmax(model_hjr((torch.Tensor([[X_test[i][0], v0]]).to(device) - mean_hjr) / std_hjr).cpu().detach().numpy(), axis=1)

    norm_error_hjr[i] = vel_norm - abs(v0)
    norm_relerror_hjr[i] = norm_error_hjr[i]/vel_norm
    absnorm_error_hjr[i] = abs(norm_error_hjr[i])
    absnorm_relerror_hjr[i] = abs(norm_relerror_hjr[i])

with torch.no_grad():
    print('RRMSE test data wrt HJR NN in %: ', math.sqrt(np.sum(np.power(norm_relerror_hjr,2))/len(norm_relerror_hjr))*100)
    print('RMSE test data wrt HJR NN: ', math.sqrt(np.sum(np.power(norm_error_hjr,2))/len(norm_error_hjr)))

# Comparison plots:
plt.figure(figsize=(6, 4))
bins = np.linspace(0, 1, 100)
plt.hist([abs(X_test_dir[i,2] - outputs[i].tolist()[0]) for i in range(len(X_test_dir))], bins, alpha=0.5, label='VBOCP', cumulative=True) #,density=True
plt.hist(absnorm_error_al, bins, alpha=0.5, label='AL', cumulative=True)
plt.hist(absnorm_error_hjr, bins, alpha=0.5, label='HJR', cumulative=True)
plt.title('Cumulative error distribution')
plt.legend(loc='lower right')
plt.ylabel('# test samples')
plt.xlabel('Error (rad/s)')

# plt.figure()
# bins = np.linspace(0, 0.5, 100)
# plt.hist([abs((X_test_dir[i,2] - outputs[i].tolist()[0])/X_test_dir[i,2]) for i in range(len(X_test_dir))], bins, alpha=0.5, label='VBOCP', cumulative=True)
# plt.hist(absnorm_relerror_al, bins, alpha=0.5, label='AL', cumulative=True)
# plt.hist(absnorm_relerror_hjr, bins, alpha=0.5, label='HJR', cumulative=True)
# plt.title('Cumulative relative error distribution')
# plt.legend(loc='lower right')

plt.figure(figsize=(6, 4))
bins = np.linspace(-1, 1, 200)
plt.hist([X_test_dir[i,2] - outputs[i].tolist()[0] for i in range(len(X_test_dir))], bins, alpha=0.5, label='VBOCP') #,density=True
plt.hist(norm_error_al, bins, alpha=0.5, label='AL')
plt.hist(norm_error_hjr, bins, alpha=0.5, label='HJR')
plt.title('Error distribution')
plt.legend(loc='upper right')
plt.ylabel('# test samples')
plt.xlabel('Error (rad/s)')

# plt.figure()
# bins = np.linspace(-0.5, 0.5, 200)
# plt.hist([(X_test_dir[i,2] - outputs[i].tolist()[0])/X_test_dir[i,2] for i in range(len(X_test_dir))], bins, alpha=0.5, label='VBOCP')
# plt.hist(norm_relerror_al, bins, alpha=0.5, label='AL')
# plt.hist(norm_relerror_hjr, bins, alpha=0.5, label='HJR')
# plt.title('Relative error distribution')
# plt.legend(loc='upper right')

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_test_dir[:,:2]).to(device)
    y_iter_tensor = torch.Tensor(X_test_dir[:,2:]).to(device)
    outputs = model_dir(X_iter_tensor)
        
    plt.figure()
    plt.plot(
        X_test[:,0], X_test[:,1], "ko", markersize=2
    )
    h = 0.01
    xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))
    inp = np.c_[xx.ravel(), yy.ravel(), yy.ravel()]
    for i in range(inp.shape[0]):
        inp[i][0] = (inp[i][0] - mean_dir) / std_dir
        vel_norm = abs(inp[i][1])
        if vel_norm != 0:
            inp[i][1] = inp[i][1] / vel_norm
        inp[i][2] = vel_norm
    out = (model_dir(torch.from_numpy(inp[:,:2].astype(np.float32)).to(device))).cpu().numpy() 
    y_pred = np.empty(out.shape)
    for i in range(len(out)):
        if inp[i][2] > out[i]:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.ylabel('$\dot{q}$')
    plt.xlabel('$q$')
    plt.grid()
    plt.title("VBOC")

    # Plot the results:
    plt.figure()
    h = 0.01
    x_min, x_max = q_min-(q_max-q_min)/100, q_max+(q_max-q_min)/100
    y_min, y_max = v_min-(v_max-v_min)/100, v_max+(v_max-v_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    inp = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    inp = (inp - mean_al) / std_al
    out = model_al(inp)
    y_pred = np.argmax(out.numpy(), axis=1)
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel("Initial position [rad]")
    plt.ylabel("Initial velocity [rad/s]")
    plt.title("AL")
    plt.grid(True)

    # Plot the results:
    plt.figure()
    h = 0.01
    xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))
    inp = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32)).to(device)
    inp = (inp - mean_hjr) / std_hjr
    out = model_hjr(inp)
    y_pred = np.argmax(out.cpu().numpy(), axis=1)
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.xlabel("Initial position [rad]")
    plt.ylabel("Initial velocity [rad/s]")
    plt.title("HJR")
    plt.grid(True)

plt.show()