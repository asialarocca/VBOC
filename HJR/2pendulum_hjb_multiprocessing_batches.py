import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from numpy.linalg import norm
import time
from doublependulum_hjr_class import OCPdoublependulum
import warnings
import math
from scipy.optimize import fsolve
import torch
import torch.nn as nn
from my_nn import NeuralNet
import random
from multiprocessing import Pool

warnings.filterwarnings("ignore")

def testing(v):
    x0 = Xu_iter[v:v+100]

    data = np.empty((0, 4))

    for l in range(x0.shape[0]):
        inp = (torch.Tensor([x0]).to(device) - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.detach().cpu().numpy(), axis=1)

        if x0[0] >= q_min and x0[0] <= q_max and x0[2] >= v_min and x0[2] <= v_max and x0[1] >= q_min and x0[1] <= q_max and x0[3] >= v_min and x0[3] <= v_max and y_pred == 1:
            res = ocp.compute_problem(x0)
            if res == 1:
                if ocp.ocp_solver.get_cost() < 0.:
                    data = np.append(data,[[x0[0], x0[1], x0[2], x0[3], 0, 1]], axis=0)
                else:
                    data = np.append(data,[[x0[0], x0[1], x0[2], x0[3], 1, 0]], axis=0)
        else:
            data = np.append(data,[[x0[0], x0[1], x0[2], x0[3], 1, 0]], axis=0)

    return data

start_time = time.time()

# Position and velocity bounds:
v_max = 10.
v_min = -10.
q_max = np.pi / 4 + np.pi
q_min = - np.pi / 4 + np.pi

# Hyper-parameters for nn:
input_size = 4
hidden_size = 100
output_size = 2
learning_rate = 1e-3

# Device configuration
device = torch.device("cpu")

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

B = pow(20, 4)  # batch size

loss_stop = 1e-3  # nn training stopping condition
beta = 0.95
n_minibatch = 4096
it_max = int(1e2 * B / n_minibatch)

Xu_iter = np.random.uniform(low=[q_min-(q_max-q_min)/10, q_min-(q_max-q_min)/10, v_min-(v_max-v_min)/10, v_min-(v_max-v_min)/10], high=[q_max+(q_max-q_min)/10, q_max+(q_max-q_min)/10, v_max+(v_max-v_min)/10, v_max+(v_max-v_min)/10], size=(B,4))
Xu_test = np.random.uniform(low=[q_min, q_min, v_min, v_min], high=[q_max, q_max, v_max, v_max], size=(B,4))
Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32)).to(device)
mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

X_iter = Xu_iter
y_iter = np.array([[0, 1] if Xu_iter[n][0] >= q_min and Xu_iter[n][0] <= q_max and Xu_iter[n][2] >= v_min and Xu_iter[n][2] <= v_max and Xu_iter[n][1] >= q_min and Xu_iter[n][1] <= q_max and Xu_iter[n][3] >= v_min and Xu_iter[n][3] <= v_max else [1, 0] for n in range(Xu_iter.shape[0])])

it = 0
val = 1

# Train the model
while val > loss_stop and it <= it_max:

    ind = random.sample(range(X_iter.shape[0]), n_minibatch)

    X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32)).to(device)
    y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32)).to(device)
    X_iter_tensor = (X_iter_tensor - mean) / std

    # Forward pass
    outputs = model(X_iter_tensor)
    loss = criterion(outputs, y_iter_tensor)

    # Backward and optimize
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    val = beta * val + (1 - beta) * loss.item()

    it += 1

pos_old = Xu_iter.shape[0]
y_test_pred = np.argmax(model((torch.from_numpy(Xu_test.astype(np.float32)).to(device) - mean) / std).detach().cpu().numpy(), axis=1)
pos_new = np.sum([1 for i in range(Xu_test.shape[0]) if y_test_pred[i]==1])/Xu_test.shape[0]

X_test = np.load('../data2_test_10.npy')

output_hjr_test = np.argmax(model((torch.Tensor(X_test).to(device) - mean) / std).cpu().detach().numpy(), axis=1)
norm_error = np.empty((len(X_test),))
for i in range(len(X_test)):
    vel_norm = norm([X_test[i][2],X_test[i][3]])
    v0 = X_test[i][2]
    v1 = X_test[i][3]

    if output_hjr_test[i] == 0:
        out = 0
        while out == 0 and norm([v0,v1]) > 1e-2:
            v0 = v0 - 1e-2 * X_test[i][2]/vel_norm
            v1 = v1 - 1e-2 * X_test[i][3]/vel_norm
            out = np.argmax(model((torch.Tensor([[X_test[i][0], X_test[i][1], v0, v1]]).to(device) - mean) / std).cpu().detach().numpy(), axis=1)
    else:
        out = 1
        while out == 1 and norm([v0,v1]) > 1e-2:
            v0 = v0 + 1e-2 * X_test[i][2]/vel_norm
            v1 = v1 + 1e-2 * X_test[i][3]/vel_norm
            out = np.argmax(model((torch.Tensor([[X_test[i][0], X_test[i][1], v0, v1]]).to(device) - mean) / std).cpu().detach().numpy(), axis=1)

    norm_error[i] = vel_norm - norm([v0,v1])

rmse = np.array([math.sqrt(np.sum(np.power(norm_error,2))/len(norm_error))])
times = np.array([time.time() - start_time])

init_times = 0
iterbreak = 0

while (pos_new <= pos_old or iterbreak > 10) and iterbreak < 100 and time.time() - start_time - init_times < 1800:
    iterbreak = iterbreak + 1

    time_before = time.time()
    ocp = OCPdoublependulum(mean.item(), std.item(), model.parameters())
    init_times = init_times + time.time() - time_before

    Xu_iter = np.random.uniform(low=[q_min, q_min, v_min-(v_max-v_min)/5, v_min-(v_max-v_min)/5], high=[q_max, q_max, v_max+(v_max-v_min)/5, v_max+(v_max-v_min)/5], size=(B,4))

    with Pool(30) as p:
        temp = p.map(testing, range(int(Xu_iter.shape[0]/100)))

    data_save = np.array([i for f in temp for i in f])
    X_iter, y_iter = data_save[:,:4], data_save[:,4:]

    it = 0
    val = 1

    # Train the model
    while val > loss_stop and it <= it_max:

        ind = random.sample(range(X_iter.shape[0]), n_minibatch)

        X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32)).to(device)
        y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32)).to(device)
        X_iter_tensor = (X_iter_tensor - mean) / std

        # Forward pass
        outputs = model(X_iter_tensor)
        loss = criterion(outputs, y_iter_tensor)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        val = beta * val + (1 - beta) * loss.item()

        it += 1

    pos_old = pos_new
    y_test_pred = np.argmax(model((torch.from_numpy(Xu_test.astype(np.float32)).to(device) - mean) / std).detach().cpu().numpy(), axis=1)
    pos_new = np.sum([1 for i in range(Xu_test.shape[0]) if y_test_pred[i]==1])/Xu_test.shape[0]

    del ocp

    times = np.append(times, [time.time() - start_time - init_times])

    output_al_test = np.argmax(model((torch.Tensor(X_test).to(device) - mean) / std).cpu().detach().numpy(), axis=1)
    norm_error_al = np.empty((len(X_test),))
    for i in range(len(X_test)):
        vel_norm = norm([X_test[i][2],X_test[i][3]])
        v0 = X_test[i][2]
        v1 = X_test[i][3]

        if output_al_test[i] == 0:
            out = 0
            while out == 0 and norm([v0,v1]) > 1e-2:
                v0 = v0 - 1e-2 * X_test[i][2]/vel_norm
                v1 = v1 - 1e-2 * X_test[i][3]/vel_norm
                out = np.argmax(model((torch.Tensor([[X_test[i][0], X_test[i][1], v0, v1]]).to(device) - mean) / std).cpu().detach().numpy(), axis=1)
        else:
            out = 1
            while out == 1 and norm([v0,v1]) > 1e-2:
                v0 = v0 + 1e-2 * X_test[i][2]/vel_norm
                v1 = v1 + 1e-2 * X_test[i][3]/vel_norm
                out = np.argmax(model((torch.Tensor([[X_test[i][0], X_test[i][1], v0, v1]]).to(device) - mean) / std).cpu().detach().numpy(), axis=1)

        norm_error_al[i] = vel_norm - norm([v0,v1])

    rmse = np.append(rmse, [math.sqrt(np.sum(np.power(norm_error_al,2))/len(norm_error_al))])

print(time.time() - start_time - init_times)

torch.save(model.state_dict(), 'model_2pendulum_hjr_10_1800_new')
torch.save(mean, 'mean_2pendulum_10_new')
torch.save(std, 'std_2pendulum_10_new')

print('RMSE test data: ', math.sqrt(np.sum(np.power(norm_error_al,2))/len(norm_error_al))) 

np.save('times_hjr_new.npy', np.asarray(times))
np.save('rmse_hjr_new.npy', np.asarray(rmse))

plt.figure()
plt.plot(times, rmse)

with torch.no_grad():
    # Plots:
    h = 0.02
    x_min, x_max = q_min-(q_max-q_min)/100, q_max+(q_max-q_min)/100
    y_min, y_max = v_min-(v_max-v_min)/100, v_max+(v_max-v_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max+h, h), np.arange(y_min, y_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()

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
    inp = (inp - mean) / std
    out = model(inp)
    y_pred = np.argmax(out.cpu().numpy(), axis=1)
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.grid()
    plt.title("Second actuator")

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
    inp = (inp - mean) / std
    out = model(inp)
    y_pred = np.argmax(out.cpu().numpy(), axis=1)
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.grid()
    plt.title("First actuator")

plt.show()