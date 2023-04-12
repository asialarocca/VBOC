import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from numpy.linalg import norm
import time
from doublependulum_ocp_class import OCPdoublependulum
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
    x0 = Xu_iter[v]

    X_iter = None
    y_iter = None

    if x0[0] >= q_min or x0[0] <= q_max or x0[2] >= v_min or x0[2] <= v_max or x0[1] >= q_min or x0[1] <= q_max or x0[3] >= v_min or x0[3] <= v_max:
        res = ocp.compute_problem(x0)
        if res == 1:
            X_iter = x0

            inp = (torch.Tensor([ocp.ocp_solver.get(0, 'x')]).to(device) - mean) / std
            out = model(inp)
            y_pred = np.argmax(out.detach().cpu().numpy(), axis=1)

            if y_pred == 1 and ocp.ocp_solver.get_cost() < 0.:
                y_iter = [0, 1]
            else:
                y_iter = [1, 0]
    else:
        X_iter = x0
        y_iter = [1, 0]

    return X_iter, y_iter

start_time = time.time()

# Position and velocity bounds:
v_max = 5.
v_min = -5.
q_max = np.pi / 4 + np.pi
q_min = - np.pi / 4 + np.pi

# Hyper-parameters for nn:
input_size = 4
hidden_size = 10
output_size = 2
learning_rate = 0.001

# Device configuration
device = torch.device("cpu")

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

B = pow(10, 4)  # batch size

loss_stop = 0.01  # nn training stopping condition
beta = 0.99
n_minibatch = 512
it_max = 1e3 * B / n_minibatch

Xu_iter = np.random.uniform(low=[q_min-(q_max-q_min)/10, q_min-(q_max-q_min)/10, v_min-(v_max-v_min)/10, v_min-(v_max-v_min)/10], high=[q_max+(q_max-q_min)/10, q_max+(q_max-q_min)/10, v_max+(v_max-v_min)/10, v_max+(v_max-v_min)/10], size=(B,4))

Xu_test = np.random.uniform(low=[q_min, q_min, v_min, v_min], high=[q_max, q_max, v_max, v_max], size=(B,4))

Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32)).to(device)
mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

X_iter = Xu_iter
y_iter = np.empty((0, 2))

for n in range(Xu_iter.shape[0]):
    x0 = Xu_iter[n]

    if x0[0] >= q_min or x0[0] <= q_max or x0[2] >= v_min or x0[2] <= v_max or x0[1] >= q_min or x0[1] <= q_max or x0[3] >= v_min or x0[3] <= v_max:
        y_iter = np.append(y_iter, [[0, 1]], axis=0)
    else:
        y_iter = np.append(y_iter, [[1, 0]], axis=0)

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

h = 0.01
xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))

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
    xit = []
    yit = []
    cit = []
    for i in range(len(X_iter)):
        if (
            norm(X_iter[i][0] - (q_min + q_max) / 2) < 0.1
            and norm(X_iter[i][2]) < 0.1
        ):
            xit.append(X_iter[i][1])
            yit.append(X_iter[i][3])
            if y_iter[i][0] == 1:
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
    xit = []
    yit = []
    cit = []
    for i in range(len(X_iter)):
        if (
            norm(X_iter[i][1] - (q_min + q_max) / 2) < 0.1
            and norm(X_iter[i][3]) < 0.1
        ):
            xit.append(X_iter[i][0])
            yit.append(X_iter[i][2])
            if y_iter[i][0] == 1:
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
    plt.title("First actuator")

pos_old = Xu_iter.shape[0]
y_test_pred = np.argmax(model((torch.from_numpy(Xu_test.astype(np.float32)).to(device) - mean) / std).detach().cpu().numpy(), axis=1)
pos_new = np.sum([1 for i in range(Xu_test.shape[0]) if y_test_pred[i]==1])/Xu_test.shape[0]

init_times = 0

while pos_new <= pos_old:
    time_before = time.time()
    ocp = OCPdoublependulum(mean.item(), std.item(), model.parameters())
    init_times = init_times + time.time() - time_before

    Xu_iter = np.random.uniform(low=[q_min-(q_max-q_min)/5, q_min-(q_max-q_min)/5, v_min-(v_max-v_min)/5, v_min-(v_max-v_min)/5], high=[q_max+(q_max-q_min)/5, q_max+(q_max-q_min)/5, v_max+(v_max-v_min)/5, v_max+(v_max-v_min)/5], size=(B,4))

    with Pool(30) as p:
        temp = p.map(testing, range(Xu_iter.shape[0]))

    x, y = zip(*temp)
    X_iter, y_iter = np.array([i for i in x if i is not None]), np.array([i for i in y if i is not None])

    # del model, optimizer
    # model = NeuralNet(input_size, hidden_size, output_size).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    
    with torch.no_grad():
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
        xit = []
        yit = []
        cit = []
        for i in range(len(X_iter)):
            if (
                norm(X_iter[i][0] - (q_min + q_max) / 2) < 0.1
                and norm(X_iter[i][2]) < 0.1
            ):
                xit.append(X_iter[i][1])
                yit.append(X_iter[i][3])
                if y_iter[i][0] == 1:
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
        xit = []
        yit = []
        cit = []
        for i in range(len(X_iter)):
            if (
                norm(X_iter[i][1] - (q_min + q_max) / 2) < 0.1
                and norm(X_iter[i][3]) < 0.1
            ):
                xit.append(X_iter[i][0])
                yit.append(X_iter[i][2])
                if y_iter[i][0] == 1:
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
        plt.title("First actuator")

    del ocp

print(time.time() - start_time - init_times)
plt.show()