import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from utils import plot_pendulum
import time
from pendulum_ocp_class import OCPpendulumINIT, OCPpendulumNN
import warnings
import random
import queue
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from my_nn import NeuralNet
import math
import sys
from casadi import SX, MX, vertcat, sin, exp, norm_2, fmax, tanh
from statistics import mean


warnings.filterwarnings("ignore")

with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPpendulumINIT()

    ocp_dim = ocp.nx  # state space dimension
    N = ocp.N

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = -ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    # Hyper-parameters for nn:
    input_size = ocp_dim
    hidden_size = ocp_dim * 50
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
    loss_stop = 0.01  # nn training stopping condition
    beta = 0.8
    n_minibatch = 64
    etp_stop = 0.1  # active learning stopping condition
    it_max = 1e2 * B / n_minibatch

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(100, ocp_dim))
    l_bounds = [q_min, v_min]
    u_bounds = [q_max, v_max]
    data = qmc.scale(sample, l_bounds, u_bounds)

    Xu_iter = np.float32(data)  # Unlabeled set
    Xu_iter_tensor = torch.from_numpy(Xu_iter)
    mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

    # Generate the initial set of labeled samples:
    res = ocp.compute_problem((q_max + q_min) / 2, 0.0)
    if res != 2:
        X_iter = np.array([[(q_max + q_min) / 2, 0.0]])
        if res == 1:
            y_iter = np.array([[0, 1]])
        else:
            y_iter = np.array([[1, 0]])

    # # Generate the initial set of labeled samples:
    # n = pow(10, ocp_dim)
    # X_iter = np.empty((n, ocp_dim))
    # y_iter = np.full((n, 2), [1, 0])
    # r = np.random.random(size=(n, ocp_dim - 1))
    # k = np.random.randint(ocp_dim, size=(n, 1))
    # j = np.random.randint(2, size=(n, 1))
    # x = np.zeros((n, ocp_dim))
    # for i in np.arange(n):
    #     x[i, np.arange(ocp_dim)[np.arange(ocp_dim) != k[i]]] = r[i, :]
    #     x[i, k[i]] = j[i]

    # X_iter[:, 0] = x[:, 0] * (q_max + 0.01 - (q_min - 0.01)) + q_min - 0.01
    # X_iter[:, 1] = x[:, 1] * (v_max + 0.1 - (v_min - 1)) + v_min - 0.1

    # Training of an initial classifier:
    for p in range(N_init):
        q0 = Xu_iter[p, 0]
        v0 = Xu_iter[p, 1]

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

    # Delete tested data from the unlabeled set:
    Xu_iter = np.delete(Xu_iter, range(N_init), axis=0)

    val = 1
    it = 0

    # Train the model
    while val > loss_stop and it <= it_max:

        ind = random.sample(range(X_iter.shape[0]), n_minibatch)
        X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32))
        y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32))
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

    # Active learning:
    k = 0  # iteration number
    etpmax = 1
    performance_history = []

    while not (etpmax < etp_stop or Xu_iter.shape[0] == 0):

        if Xu_iter.shape[0] < B:
            B = Xu_iter.shape[0]

        with torch.no_grad():
            sigmoid = nn.Sigmoid()
            Xu_iter_tensor = torch.from_numpy(Xu_iter)
            Xu_iter_tensor = (Xu_iter_tensor - mean) / std
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
            res = ocp.compute_problem(q0, v0)
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
            X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32))
            y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32))
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

        print("CLASSIFIER", k, "TRAINED")

    with torch.no_grad():
        # Plot the results:
        plt.figure()
        h = 0.02
        x_min, x_max = q_min-h, q_max+h
        y_min, y_max = v_min, v_max
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        inp = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
        inp = (inp - mean) / std
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
        plt.xlim([0.0, np.pi / 2])
        plt.ylim([-10.0, 10.0])
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Classifier")
        hand = scatter.legend_elements()[0]
        plt.legend(handles=hand, labels=("Non viable", "Viable"))
        plt.grid(True)

        # input_data = (data - mean.item()) / std.item()
        # output_data = np.empty(np.shape(input_data)[0])

        # for i in range(np.shape(input_data)[0]):

        #     init = input_data[i]

        #     output = init
        #     it = 2

        #     for param in model.parameters():

        #         if it % 2 == 0:
        #             output = np.dot(param.numpy(), output)
        #         else:
        #             output = np.add(param.numpy(), output)

        #             if it == 3:
        #                 for k in range(np.shape(output)[0]):
        #                     output[k] = max(0.0, output[k])
        #             elif it == 5:
        #                 for k in range(np.shape(output)[0]):
        #                     output[k] = np.tanh(output[k])
        #             else:
        #                 for k in range(np.shape(output)[0]):
        #                     output[k] = 1 / (1 + math.exp(-output[k]))

        #                     if output[1]-output[0] >= 0:
        #                         output_data[i] = 1
        #                     else:
        #                         output_data[i] = 0

        #         it += 1

        # plt.figure()
        # scatter = plt.scatter(
        #     data[:, 0], data[:, 1], c=output_data, marker=".", alpha=0.5, cmap=plt.cm.Paired
        # )
        # plt.xlim([0.0, np.pi / 2])
        # plt.ylim([-10.0, 10.0])
        # plt.xlabel("Initial position [rad]")
        # plt.ylabel("Initial velocity [rad/s]")
        # plt.title("Classifier")
        # hand = scatter.legend_elements()[0]
        # plt.legend(handles=hand, labels=("Non viable", "Viable"))
        # plt.grid(True)

        # input_data = ((data - mean.item()) / std.item()).tolist()
        # output_data = np.empty(np.shape(input_data)[0])
        # z = np.empty(np.shape(input_data)[0])

        # for i in range(np.shape(data)[0]):

        #     init = input_data[i]

        #     out = init
        #     it = 2

        #     for param in model.parameters():
        #         param = SX(param.tolist())
        #         if it % 2 == 0:
        #             out = param @ out
        #         else:
        #             out = param + out

        #             if it == 3:
        #                 out = fmax(0.0, out)
        #             elif it == 5:
        #                 out = tanh(out)
        #             else:
        #                 out = 1 / (1 + exp(-out))
        #                 output_data[i] = out[1]-out[0]
        #         it += 1

        # for i in range(np.shape(data)[0]):
        #     if output_data[i] >= 0:
        #         z[i] = 1
        #     else:
        #         z[i] = 0

        # plt.figure()

        # scatter = plt.scatter(

        #     data[:, 0], data[:, 1], c=z, marker=".", alpha=0.5, cmap=plt.cm.Paired
        # )
        # plt.xlim([0.0, np.pi / 2])
        # plt.ylim([-10.0, 10.0])
        # plt.xlabel("Initial position [rad]")
        # plt.ylabel("Initial velocity [rad/s]")
        # plt.title("Classifier")
        # hand = scatter.legend_elements()[0]
        # plt.legend(handles=hand, labels=("Non viable", "Viable"))
        # plt.grid(True)

        # used for the stopping criterion
        n_sum = sum(
            np.argmax(model((torch.Tensor(data)-mean)/std).numpy(), axis=1))

    while True:

        del ocp

        # Ocp initialization:
        ocp = OCPpendulumNN(model, mean.item(), std.item())

        with torch.no_grad():
            # The unlabeled set is redefined with the data that resulted unfeasible from testing or classification:
            y_argmax = np.argmax(y_iter, axis=1)
            data_xinit = X_iter[y_argmax == 0]
            xu_pred = np.argmax(
                model((torch.Tensor(Xu_iter)-mean)/std).numpy(), axis=1)
            data_xuinit = Xu_iter[xu_pred == 0]
            Xu_iter = np.float32(np.concatenate([data_xinit, data_xuinit]))

            # The labeled set is redefined with the data previously tested as feasible:
            X_iter = np.array(X_iter[y_argmax == 1])
            y_iter = np.array(y_iter[y_argmax == 1])

        Xu_iter_tensor = torch.Tensor(Xu_iter)
        mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

        del model

        model = NeuralNet(input_size, hidden_size, output_size).to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training of an initial classifier:
        for p in range(N_init):
            q0 = Xu_iter[p, 0]
            v0 = Xu_iter[p, 1]

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

        # Delete tested data from the unlabeled set:
        Xu_iter = np.delete(Xu_iter, range(N_init), axis=0)

        val = 1
        it = 0

        # Train the model
        while val > loss_stop and it <= it_max:

            ind = random.sample(range(X_iter.shape[0]), n_minibatch)
            X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32))
            y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32))
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

        k += 1

        # Active learning:
        etpmax = 1

        while not (etpmax < etp_stop or Xu_iter.shape[0] == 0):

            if Xu_iter.shape[0] < B:
                B = Xu_iter.shape[0]

            with torch.no_grad():
                sigmoid = nn.Sigmoid()
                Xu_iter_tensor = torch.from_numpy(Xu_iter)
                Xu_iter_tensor = (Xu_iter_tensor - mean) / std
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
                res = ocp.compute_problem(q0, v0)
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
                X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32))
                y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32))
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

            print("CLASSIFIER", k, "TRAINED")

        with torch.no_grad():
            # Plot the results:
            plt.figure()
            h = 0.02
            x_min, x_max = q_min-h, q_max+h
            y_min, y_max = v_min, v_max
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            inp = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
            inp = (inp - mean) / std
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
            plt.xlim([0.0, np.pi / 2])
            plt.ylim([-10.0, 10.0])
            plt.xlabel("Initial position [rad]")
            plt.ylabel("Initial velocity [rad/s]")
            plt.title("Classifier")
            hand = scatter.legend_elements()[0]
            plt.legend(handles=hand, labels=("Non viable", "Viable"))
            plt.grid(True)

            # Stopping condition: compare two consecutive classifiers by the number of positively classified
            # data from a predefined pool of samples:
            # number of positively classified data
            n_temp = sum(
                np.argmax(model((torch.Tensor(data)-mean)/std).numpy(), axis=1))
            if n_temp - n_sum < 100:
                break
            n_sum = n_temp

# pr.print_stats(sort="cumtime")

print("Execution time: %s seconds" % (time.time() - start_time))

plt.show()
