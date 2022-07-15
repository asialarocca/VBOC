import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from utils import plot_pendulum
import time
from pendulum_ocp_class import OCPpendulumINIT
import warnings
import random
import queue
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from my_nn import NeuralNet
import math
import sys

warnings.filterwarnings("ignore")

with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPpendulumINIT()

    ocp_dim = ocp.nx  # state space dimension

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = -ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    # Hyper-parameters for nn:
    input_size = ocp_dim
    hidden_size = ocp_dim * 20
    output_size = 2
    learning_rate = 0.01

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Active learning parameters:
    N_init = pow(10, ocp_dim)  # size of initial labeled set
    B = pow(10, ocp_dim)  # batch size
    loss_stop = 0.1  # nn training stopping condition
    beta = 0.8
    n_minibatch = 64
    etp_stop = 0.1  # active learning stopping condition
    it_max = 1e2 * B / n_minibatch

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(50, ocp_dim))
    l_bounds = [q_min, v_min]
    u_bounds = [q_max, v_max]
    data = qmc.scale(sample, l_bounds, u_bounds)

    Xu_iter = np.float32(data)  # Unlabeled set
    Xu_iter_tensor = torch.from_numpy(Xu_iter)
    # mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

    # Generate the initial set of labeled samples:
    n = pow(10, ocp_dim)
    X_iter = np.empty((n, ocp_dim), dtype="f")
    y_iter = np.full((n, 2), [1, 0], dtype="f")
    r = np.random.random(size=(n, ocp_dim - 1))
    k = np.random.randint(ocp_dim, size=(n, 1))
    j = np.random.randint(2, size=(n, 1))
    x = np.zeros((n, ocp_dim))
    for i in np.arange(n):
        x[i, np.arange(ocp_dim)[np.arange(ocp_dim) != k[i]]] = r[i, :]
        x[i, k[i]] = j[i]

    X_iter[:, 0] = np.float32(x[:, 0] * (q_max + 0.1 - (q_min - 0.1)) + q_min - 0.1)
    X_iter[:, 1] = np.float32(x[:, 1] * (v_max + 1 - (v_min - 1)) + v_min - 1)

    # Get solution of positive samples:
    N = ocp.N
    simX_vec = np.ndarray((1, N + 1, 2))
    simX_vec.fill(0.0)
    simU_vec = np.ndarray((1, N, 1))
    simU_vec.fill(0.0)
    simX = np.ndarray((N + 1, 2))
    simU = np.ndarray((N, 1))

    # Training of an initial classifier:
    for n in range(N_init):
        q0 = Xu_iter[n, 0]
        v0 = Xu_iter[n, 1]

        # Data testing:
        res = ocp.compute_problem(q0, v0)
        if res != 2:
            X_iter = np.append(X_iter, [[q0, v0]], axis=0)
            if res == 1:
                y_iter = np.append(y_iter, [[0, 1]], axis=0)
            else:
                y_iter = np.append(y_iter, [[1, 0]], axis=0)

        # Add intermediate states of succesfull initial conditions
        if res == 1:
            for f in range(1, N, int(N / 3)):
                current_val = np.float32(ocp.ocp_solver.get(f, "x"))
                X_iter = np.append(X_iter, [current_val], axis=0)
                y_iter = np.append(y_iter, [[0, 1]], axis=0)

                for i in range(N):
                    simX[i, :] = ocp.ocp_solver.get(i, "x")
                    simU[i] = ocp.ocp_solver.get(i, "u")
                simX[N, :] = ocp.ocp_solver.get(N, "x")

                simX_vec = np.append(simX_vec, [simX], axis=0)
                simU_vec = np.append(simU_vec, [simU], axis=0)

    # Delete tested data from the unlabeled set:
    Xu_iter = np.delete(Xu_iter, range(N_init), axis=0)

    val = 1
    it = 0

    # Train the model
    while val > loss_stop and it <= it_max:

        ind = random.sample(range(X_iter.shape[0]), n_minibatch)
        X_iter_tensor = torch.from_numpy(X_iter[ind])
        y_iter_tensor = torch.from_numpy(y_iter[ind])
        # X_iter_tensor = (X_iter_tensor - mean) / std

        # Forward pass
        outputs = model(X_iter_tensor)
        loss = criterion(outputs, y_iter_tensor)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        val = beta * val + (1 - beta) * loss.item()

        it += 1

    print("INITIAL CLASSIFIER TRAINED")

    with torch.no_grad():
        # Plot the results:
        plt.figure()
        x_min, x_max = 0.0, np.pi / 2
        y_min, y_max = -10.0, 10.0
        h = 0.02
        xx, yy = np.float32(
            np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        )
        inp = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()])
        # inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        z = [
            0 if np.array_equal(y_iter[x], [1, 0]) else 1
            for x in range(y_iter.shape[0])
        ]
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        scatter = plt.scatter(
            X_iter[:, 0], X_iter[:, 1], c=z, marker=".", alpha=0.5, cmap=plt.cm.Paired
        )
        plt.xlim([0.0, np.pi / 2 - 0.01])
        plt.ylim([-10.0, 10.0])
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Classifier")
        hand = scatter.legend_elements()[0]
        plt.legend(handles=hand, labels=("Non viable", "Viable"))

        # input_data = np.array([0.8, 10.0])
        # input_data = (input_data - mean.numpy()) / std.numpy()

        # output = input_data
        # it = 2

        # for param in model.parameters():

        #     if it % 2 == 0:
        #         output = np.dot(param.numpy(), output)
        #     else:
        #         output = np.add(param.numpy(), output)

        #         if it == 3:
        #             for i in range(np.shape(output)[0]):
        #                 output[i] = max(0.0, output[i])
        #         elif it == 5:
        #             for k in range(np.shape(output)[0]):
        #                 output[k] = np.tanh(output[k])
        #         else:
        #             for k in range(np.shape(output)[0]):
        #                 output[k] = 1 / (1 + math.exp(-output[k]))

        #     it += 1

    # Active learning:
    k = 0  # iteration number
    etpmax = 1
    performance_history = [etpmax]

    while not (etpmax < etp_stop or Xu_iter.shape[0] == 0):

        if Xu_iter.shape[0] < B:
            B = Xu_iter.shape[0]

        with torch.no_grad():
            sigmoid = nn.Sigmoid()
            Xu_iter_tensor = torch.from_numpy(Xu_iter)
            # Xu_iter_tensor = (Xu_iter_tensor - mean) / std
            prob_xu = sigmoid(model(Xu_iter_tensor)).numpy()
            etp = entropy(prob_xu, axis=1)

        maxindex = np.argpartition(etp, -B)[-B:]  # indexes of the uncertain samples

        etpmax = max(etp)  # max entropy used for the stopping condition
        performance_history.append(etpmax)

        k += 1

        # Add the B most uncertain samples to the labeled set:
        for x in range(B):
            q0 = Xu_iter[maxindex[x], 0]
            v0 = Xu_iter[maxindex[x], 1]

            # Data testing:
            res = ocp.compute_problem_withGUESS(q0, v0, simX_vec, simU_vec)
            # res = ocp.compute_problem(q0, v0)
            if res != 2:
                X_iter = np.append(X_iter, [[q0, v0]], axis=0)
                if res == 1:
                    y_iter = np.append(y_iter, [[0, 1]], axis=0)
                else:
                    y_iter = np.append(y_iter, [[1, 0]], axis=0)

        # Delete tested data from the unlabeled set:
        Xu_iter = np.delete(Xu_iter, maxindex, axis=0)

        it = 0
        val = 1

        # Train the model
        while val > loss_stop and it <= it_max:

            ind = random.sample(range(X_iter.shape[0] - B), int(n_minibatch / 2))
            ind.extend(
                random.sample(
                    range(X_iter.shape[0] - B, X_iter.shape[0]),
                    int(n_minibatch / 2),
                )
            )
            X_iter_tensor = torch.from_numpy(X_iter[ind])
            y_iter_tensor = torch.from_numpy(y_iter[ind])
            # X_iter_tensor = (X_iter_tensor - mean) / std

            # Forward pass
            outputs = model(X_iter_tensor)
            loss = criterion(outputs, y_iter_tensor)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            val = beta * val + (1 - beta) * loss.item()

            it += 1

        print("CLASSIFIER", k, "TRAINED")

    with torch.no_grad():
        # Plot the results:
        plt.figure()
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        z = [
            0 if np.array_equal(y_iter[x], [1, 0]) else 1
            for x in range(y_iter.shape[0])
        ]
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        scatter = plt.scatter(
            X_iter[:, 0],
            X_iter[:, 1],
            c=z,
            marker=".",
            alpha=0.5,
            cmap=plt.cm.Paired,
        )
        plt.xlim([0.0, np.pi / 2 - 0.01])
        plt.ylim([-10.0, 10.0])
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Classifier")
        hand = scatter.legend_elements()[0]
        plt.legend(handles=hand, labels=("Non viable", "Viable"))

        # # Plot of the entropy:
        # plt.figure()
        # prob_xu = sigmoid(out).numpy()
        # etxu = entropy(prob_xu, axis=1)
        # out = etxu.reshape(xx.shape)
        # levels = np.linspace(out.min(), out.max(), 10)
        # plt.contourf(xx, yy, out, levels=levels)
        # this = plt.contour(xx, yy, out, levels=levels, colors=("k",), linewidths=(1,))
        # plt.clabel(this, fmt="%2.1f", colors="w", fontsize=11)
        # plt.xlim([0.0, np.pi / 2 - 0.01])
        # plt.ylim([-10.0, 10.0])
        # plt.xlabel("Initial position [rad]")
        # plt.ylabel("Initial velocity [rad/s]")
        # plt.title("Entropy")

#     del (X_iter, y_iter, ocp)

#     # Ocp initialization:
#     ocp = OCPpendulumNN(model)  # , mean.item(), std.item())

#     del model

#     model = NeuralNet(input_size, hidden_size, output_size).to(device)

#     # Loss and optimizer
#     criterion = nn.BCELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     # Active learning parameters:
#     N_init = pow(10, ocp_dim)  # size of initial labeled set
#     B = pow(10, ocp_dim)  # batch size
#     loss_stop = 0.05  # nn training stopping condition
#     beta = 0.8
#     n_minibatch = 64

#     # Generate low-discrepancy unlabeled samples:
#     sampler = qmc.Halton(d=ocp_dim, scramble=False)
#     sample = sampler.random(n=pow(20, ocp_dim))
#     l_bounds = [q_min, v_min]
#     u_bounds = [q_max, v_max]
#     Xu_iter = qmc.scale(sample, l_bounds, u_bounds)

#     Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32))

#     # Generate the initial set of labeled samples:
#     n = pow(10, ocp_dim)
#     X_iter = np.empty((n, ocp_dim))
#     y_iter = np.full((n, 1), [[0]])
#     r = np.random.random(size=(n, ocp_dim - 1))
#     k = np.random.randint(ocp_dim, size=(n, 1))
#     j = np.random.randint(2, size=(n, 1))
#     x = np.zeros((n, ocp_dim))
#     for i in np.arange(n):
#         x[i, np.arange(ocp_dim)[np.arange(ocp_dim) != k[i]]] = r[i, :]
#         x[i, k[i]] = j[i]

#     X_iter[:, 0] = x[:, 0] * (q_max + 0.1 - (q_min - 0.1)) + q_min - 0.1
#     X_iter[:, 1] = x[:, 1] * (v_max + 1 - (v_min - 1)) + v_min - 1

#     # Training of an initial classifier:
#     for n in range(Xu_iter.shape[0]):
#         q0 = Xu_iter[n, 0]
#         v0 = Xu_iter[n, 1]

#         # Data testing:
#         res = ocp.compute_problem(q0, v0)
#         if res != 2:
#             X_iter = np.append(X_iter, [[q0, v0]], axis=0)
#             y_iter = np.append(y_iter, [[res]], axis=0)
#         else:
#             raise Exception("Max iteration reached")

#         # Add intermediate states of succesfull initial conditions
#         if res == 1:
#             for f in range(1, ocp.N, int(ocp.N / 3)):
#                 current_val = ocp.ocp_solver.get(f, "x")
#                 X_iter = np.append(X_iter, [current_val], axis=0)
#                 y_iter = np.append(y_iter, [[1]], axis=0)

#     val = 1

#     # Train the model
#     while val > loss_stop:

#         ind = random.sample(range(X_iter.shape[0]), n_minibatch)
#         X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32))
#         y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32))
#         # X_iter_tensor = (X_iter_tensor - mean) / std

#         # Forward pass
#         outputs = model(X_iter_tensor)
#         loss = criterion(outputs, y_iter_tensor)

#         # Backward and optimize
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         val = beta * val + (1 - beta) * loss.item()

#     print("INITIAL CLASSIFIER TRAINED")

#     with torch.no_grad():
#         # Plot the results:
#         plt.figure()
#         x_min, x_max = 0.0, np.pi / 2
#         y_min, y_max = -10.0, 10.0
#         h = 0.02
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#         inp = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
#         # inp = (inp - mean) / std
#         out = model(inp)
#         out = out.numpy()
#         out = out.reshape(xx.shape)

#         def my_fun(x):
#             return 1 if x >= 0.5 else 0

#         out = np.vectorize(my_fun)(out)
#         plt.contourf(xx, yy, out, cmap=plt.cm.coolwarm, alpha=0.8)
#         scatter = plt.scatter(
#             X_iter[:, 0],
#             X_iter[:, 1],
#             c=y_iter,
#             marker=".",
#             alpha=0.5,
#             cmap=plt.cm.Paired,
#         )
#         plt.xlim([0.0, np.pi / 2 - 0.01])
#         plt.ylim([-10.0, 10.0])
#         plt.xlabel("Initial position [rad]")
#         plt.ylabel("Initial velocity [rad/s]")
#         plt.title("Classifier")
#         hand = scatter.legend_elements()[0]
#         plt.legend(handles=hand, labels=("Non viable", "Viable"))

#     print("Execution time: %s seconds" % (time.time() - start_time))

pr.print_stats(sort="cumtime")

# del ocp
# ocp = OCPpendulumNN(model)

# N = ocp.N
# Tf = ocp.Tf
# Fmax = ocp.Fmax

# ocp.compute_problem(1.4, -9.0)

# # get solution
# simX = np.ndarray((N + 1, 2))
# simU = np.ndarray((N, 1))

# for i in range(N):
#     simX[i, :] = ocp.ocp_solver.get(i, "x")
#     simU[i, :] = ocp.ocp_solver.get(i, "u")
# simX[N, :] = ocp.ocp_solver.get(N, "x")

# ocp.ocp_solver.print_statistics()
print("Execution time: %s seconds" % (time.time() - start_time))

# plot_pendulum(np.linspace(0, Tf, N + 1), Fmax, simU, simX, latexify=False)

plt.show()
