import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import matplotlib.pyplot as plt
import time
from pendulum_hjr_class import OCPpendulum
import warnings
import torch
import torch.nn as nn
from my_nn import NeuralNetCLS
import random
from multiprocessing import Pool

warnings.filterwarnings("ignore")

def data_generation(v):
    q0 = Xu_iter[v, 0]
    v0 = Xu_iter[v, 1]
    out_pred = out_iter[v]

    state = None
    output = None

    if q0>=q_min and q0<=q_max and v0>=v_min and v0<=v_max and out_pred == 1:
        res = ocp.compute_problem(q0, v0)
        if res == 1:
            state = [q0, v0]
            if ocp.ocp_solver.get_cost() < 0.:
                output = [0, 1]
            else:
                output = [1, 0]
    else:
        state = [q0, v0]
        output = [1, 0]

    return state, output

start_time = time.time()

# Position and velocity bounds:
v_max = 10.
v_min = -10.
q_max = np.pi / 4 + np.pi
q_min = - np.pi / 4 + np.pi

# Hyper-parameters for nn:
input_size = 2
hidden_size = 100
output_size = 2
learning_rate = 1e-3

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and optimizer:
model = NeuralNetCLS(input_size, hidden_size, output_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

B = pow(100, 2)  # batch size

loss_stop = 1e-3  # nn training stopping condition
beta = 0.95
n_minibatch = 64
it_max = int(1e2 * B / n_minibatch)

# Generate unlabeled and test dataset:
Xu_iter = np.random.uniform(low=[q_min-(q_max-q_min)/10, v_min-(v_max-v_min)/10], high=[q_max+(q_max-q_min)/10, v_max+(v_max-v_min)/10], size=(B,2))
Xu_test = np.random.uniform(low=[q_min, v_min], high=[q_max, v_max], size=(B,2))

# Mean and standard:
Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32)).to(device)
mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

# Initial train dataset:
X_iter = Xu_iter
y_iter = np.array([[0, 1] if Xu_iter[n, 0]>=q_min and Xu_iter[n, 0]<=q_max and Xu_iter[n, 1]>=v_min and Xu_iter[n, 1]<=v_max else [1, 0] for n in range(Xu_iter.shape[0])])

it = 0
val = 1

# Train the model:
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

# Compute number of positively classified test samples:
pos_old = Xu_iter.shape[0]
y_test_pred = np.argmax(model((torch.from_numpy(Xu_test.astype(np.float32)).to(device) - mean) / std).detach().cpu().numpy(), axis=1)
pos_new = np.sum([1 for i in range(Xu_test.shape[0]) if y_test_pred[i]==1])/Xu_test.shape[0]

init_times = 0
iterbreak = 0

# Iteratively update the model:
while pos_new <= pos_old and iterbreak < 100:
    iterbreak = iterbreak + 1

    time_before = time.time()
    ocp = OCPpendulum(mean.item(), std.item(), model.parameters())
    init_times = init_times + time.time() - time_before

    # Compute predictions wrt current model:
    Xu_iter = np.random.uniform(low=[q_min-(q_max-q_min)/10, v_min-(v_max-v_min)/10], high=[q_max+(q_max-q_min)/10, v_max+(v_max-v_min)/10], size=(B,2))
    inp = (torch.from_numpy(Xu_iter.astype(np.float32)).to(device) - mean) / std
    out = model(inp)
    out_iter = np.argmax(out.detach().cpu().numpy(), axis=1)

    # Data testing:
    with Pool(30) as p:
        temp = p.map(data_generation, range(Xu_iter.shape[0]))

    x, y = zip(*temp)
    X_iter, y_iter = np.array([i for i in x if i is not None]), np.array([i for i in y if i is not None])

    it = 0
    val = 1

    # Train the model:
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

    # Compute number of positively classified test samples:
    pos_old = pos_new
    y_test_pred = np.argmax(model((torch.from_numpy(Xu_test.astype(np.float32)).to(device) - mean) / std).detach().cpu().numpy(), axis=1)
    pos_new = np.sum([1 for i in range(Xu_test.shape[0]) if y_test_pred[i]==1])/Xu_test.shape[0]

    del ocp

    # Show the set approximation:
    with torch.no_grad():
        # Plot the results:
        plt.figure()
        h = 0.01
        xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))
        inp = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32)).to(device)
        inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.cpu().numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # plt.scatter(X_iter[:,0], X_iter[:,1], c=[0 if y_iter[n,0] == 1 else 1 for n in range(y_iter.shape[0])])
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Classifier")
        plt.grid(True)
        # plt.show()

print('Computation time:', time.time() - start_time - init_times)

torch.save(model.state_dict(), 'model_1dof_hjr')
torch.save(mean, 'mean_1dof_hjr')
torch.save(std, 'std_1dof_hjr')

plt.show()
