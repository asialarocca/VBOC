import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
import time
from pendulum_ocp_class import OCPpendulumINIT
import warnings
import random
import torch
import torch.nn as nn
from my_nn import NeuralNet
import math
from scipy.optimize import fsolve

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
    etp_stop = 0.5  # active learning stopping condition
    loss_stop = 0.01  # nn training stopping condition
    beta = 0.8
    n_minibatch = 64
    it_max = 1e2 * B / n_minibatch

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(50, ocp_dim))
    l_bounds = [q_min, v_min]
    u_bounds = [q_max, v_max]
    data = qmc.scale(sample, l_bounds, u_bounds)

    Xu_iter = data  # Unlabeled set
    Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32))
    mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

    # Generate the initial set of labeled samples:
    contour = pow(10, ocp_dim)
    X_iter = np.empty((contour, ocp_dim))
    y_iter = np.full((contour, 2), [1, 0])
    r = np.random.random(size=(contour, ocp_dim - 1))
    k = np.random.randint(ocp_dim, size=(contour, 1))
    j = np.random.randint(2, size=(contour, 1))
    x = np.zeros((contour, ocp_dim))
    for i in np.arange(contour):
        x[i, np.arange(ocp_dim)[np.arange(ocp_dim) != k[i]]] = r[i, :]
        x[i, k[i]] = j[i]

    X_iter[:, 0] = x[:, 0] * (q_max + (q_max-q_min)/50 -
                              (q_min - (q_max-q_min)/50)) + q_min - (q_max-q_min)/50
    X_iter[:, 1] = x[:, 1] * (v_max + (v_max-v_min)/50 -
                              (v_min - (v_max-v_min)/50)) + v_min - (v_max-v_min)/50

    # # Generate the initial set of labeled samples:
    # res = ocp.compute_problem((q_max + q_min) / 2, 0.0)
    # if res != 2:
    #     X_iter = np.double([[(q_max + q_min) / 2, 0.0]])
    #     if res == 1:
    #         y_iter = [[0, 1]]
    #     else:
    #         y_iter = [[1, 0]]
    # else:
    #     raise Exception("Max iteration reached")

    # Training of an initial classifier:
    for n in range(N_init):
        q0 = data[n, 0]
        v0 = data[n, 1]

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
            for f in range(1, ocp.N, int(ocp.N / 3)):
                current_val = ocp.ocp_solver.get(f, "x")
                X_iter = np.append(X_iter, [current_val], axis=0)
                y_iter = np.append(y_iter, [[0, 1]], axis=0)

    # Delete tested data from the unlabeled set:
    Xu_iter = np.delete(Xu_iter, range(N_init), axis=0)

    it = 0
    val = 1

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

    print("INITIAL CLASSIFIER TRAINED")

    with torch.no_grad():
        # Plot the results:
        plt.figure()
        h = 0.01
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

        # # Plot of the entropy:
        # sigmoid = nn.Sigmoid()
        # Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32))
        # Xu_iter_tensor = (Xu_iter_tensor - mean) / std
        # prob_xu = sigmoid(model(Xu_iter_tensor)).numpy()
        # etp = entropy(prob_xu, axis=1)
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

    # Active learning:
    k = 0  # iteration number
    etpmax = 1
    performance_history = []

    while not (etpmax < etp_stop or Xu_iter.shape[0] == 0):
        if Xu_iter.shape[0] < B:
            B = Xu_iter[S].shape[0]

        with torch.no_grad():
            sigmoid = nn.Sigmoid()
            Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32))
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
                    
            # Add intermediate states of succesfull initial conditions
            if res == 1:
                for f in range(1, ocp.N, int(ocp.N / 3)):
                    current_val = ocp.ocp_solver.get(f, "x")
                    X_iter = np.append(X_iter, [current_val], axis=0)
                    y_iter = np.append(y_iter, [[0, 1]], axis=0)

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
        plt.xlim([0.0, np.pi / 2])
        plt.ylim([-10.0, 10.0])
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Classifier")
        hand = scatter.legend_elements()[0]
        plt.legend(handles=hand, labels=("Non viable", "Viable"))
        plt.grid(True)

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

    plt.figure()
    plt.plot(performance_history)
    plt.scatter(range(len(performance_history)), performance_history)
    plt.xlabel("Iteration number")
    plt.ylabel("Maximum entropy")
    plt.title("Maximum entropy evolution")

    print("Execution time: %s seconds" % (time.time() - start_time))

pr.print_stats(sort="cumtime")

plt.show()
