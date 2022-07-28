import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
import time
from double_pendulum_ocp_class_mpc import OCPdoublependulumINIT, OCPdoublependulumNN
import random
import warnings
import torch
import torch.nn as nn
from my_nn import NeuralNet
from statistics import mean

warnings.filterwarnings("ignore")


with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPdoublependulumINIT()

    ocp_dim = ocp.nx

    # Hyper-parameters for nn:
    input_size = ocp_dim
    hidden_size = ocp_dim * 50
    output_size = 2
    learning_rate = 0.001

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Active learning parameters:
    N_init = pow(10, ocp_dim)  # size of initial labeled set
    B = pow(10, ocp_dim)  # batch size
    etp_stop = 0.1  # active learning stopping condition
    loss_stop = 0.1  # nn training stopping condition
    beta = 0.8
    n_minibatch = 512
    it_max = int(100 * B / n_minibatch)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(20, ocp_dim))
    l_bounds = ocp.xmin.tolist()
    u_bounds = ocp.xmax.tolist()
    Xu_iter = qmc.scale(sample, l_bounds, u_bounds).tolist()

    # Generate the initial set of labeled samples:
    n = pow(10, ocp_dim)
    X_iter = np.empty((n, ocp_dim))
    y_iter = np.full((n, 2), [1, 0])
    r = np.random.random(size=(n, ocp_dim - 1))
    k = np.random.randint(ocp_dim, size=(n, 1))
    j = np.random.randint(2, size=(n, 1))
    x = np.zeros((n, ocp_dim))
    for i in np.arange(n):
        x[i, np.arange(ocp_dim)[np.arange(ocp_dim) != k[i]]] = r[i, :]
        x[i, k[i]] = j[i]

    X_iter[:, 0] = np.float32(x[:, 0] * (ocp.xmax[0] + 0.5 -
                              (ocp.xmin[0] - 0.5)) + ocp.xmin[0] - 0.5)
    X_iter[:, 1] = np.float32(x[:, 1] * (ocp.xmax[1] + 0.5 -
                              (ocp.xmin[1] - 0.5)) + ocp.xmin[1] - 0.5)
    X_iter[:, 2] = np.float32(
        x[:, 2] * (ocp.xmax[2] + 0.5 - (ocp.xmin[2] - 0.5)) + ocp.xmin[2] - 0.5)
    X_iter[:, 3] = np.float32(
        x[:, 3] * (ocp.xmax[3] + 0.5 - (ocp.xmin[3] - 0.5)) + ocp.xmin[3] - 0.5)

    X_iter = X_iter.tolist()
    y_iter = y_iter.tolist()

    # # Generate the initial set of labeled samples:
    # X_iter = [[(q_max + q_min) / 2, (q_max + q_min) / 2, 0.0, 0.0]]
    # res = ocp.compute_problem([(q_max + q_min) / 2, (q_max + q_min) / 2], [0.0, 0.0])
    # if res == 1:
    #     y_iter = [[0, 1]]
    # else:
    #     y_iter = [[1, 0]]

    Xu_iter_tensor = torch.Tensor(Xu_iter)
    mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

    # Training of an initial classifier:
    for n in range(N_init):
        q0 = Xu_iter[n][:2]
        v0 = Xu_iter[n][2:]

        # Data testing:
        res = ocp.compute_problem(q0, v0)
        #res = ocp.compute_problem_withGUESSPID(q0, v0)
        if res == 1:
            X_iter.append([q0[0], q0[1], v0[0], v0[1]])
            y_iter.append([0, 1])
        elif res == 0:
            X_iter.append([q0[0], q0[1], v0[0], v0[1]])
            y_iter.append([1, 0])

        # Add intermediate states of succesfull initial conditions:
        if res == 1:
            for f in range(1, ocp.N, int(ocp.N / 3)):
                current_val = ocp.ocp_solver.get(f, "x").tolist()
                X_iter.append(current_val)
                y_iter.append([0, 1])

    # Delete tested data from the unlabeled set:
    del Xu_iter[:N_init]

    print("INITIAL CLASSIFIER IN TRAINING")

    it = 0
    val = 1

    # Train the model
    while val > loss_stop and it <= it_max:

        ind = random.sample(range(len(X_iter)), n_minibatch)

        X_iter_tensor = torch.Tensor([X_iter[i] for i in ind])
        y_iter_tensor = torch.Tensor([y_iter[i] for i in ind])
        X_iter_tensor = (X_iter_tensor - mean) / std

        # Zero the gradients
        for param in model.parameters():
            param.grad = None

        # Forward pass
        outputs = model(X_iter_tensor)
        loss = criterion(outputs, y_iter_tensor)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        val = beta * val + (1 - beta) * loss.item()

        it += 1

    print("INITIAL CLASSIFIER TRAINED")

    # Plots:
    h = 0.01
    xx1, yy1 = np.meshgrid(np.arange(ocp.xmin[0], ocp.xmax[0], h),
                           np.arange(ocp.xmin[2], ocp.xmax[2], h))
    xrav1 = xx1.ravel()
    yrav1 = yy1.ravel()
    xx2, yy2 = np.meshgrid(np.arange(ocp.xmin[1], ocp.xmax[1], h),
                           np.arange(ocp.xmin[3], ocp.xmax[3], h))
    xrav2 = xx2.ravel()
    yrav2 = yy2.ravel()

    with torch.no_grad():
        # Plot the results:
        plt.figure()
        inp = torch.from_numpy(
            np.float32(
                np.c_[
                    (ocp.xmin[0] + ocp.xmax[0]) / 2 * np.ones(xrav2.shape[0]),
                    xrav2,
                    np.zeros(yrav2.shape[0]),
                    yrav2,
                ]
            )
        )
        inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx2.shape)
        plt.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        xit = []
        yit = []
        cit = []
        for i in range(len(X_iter)):
            if (
                norm(X_iter[i][0] - (ocp.xmin[0] + ocp.xmax[0]) / 2) < 0.1
                and norm(X_iter[i][2]) < 0.01
            ):
                xit.append(X_iter[i][1])
                yit.append(X_iter[i][3])
                if y_iter[i] == [1, 0]:
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
        plt.xlim([ocp.xmin[1], ocp.xmax[1]])
        plt.ylim([ocp.xmin[3], ocp.xmax[3]])
        plt.grid()
        plt.title("Second actuator")

        plt.figure()
        inp = torch.from_numpy(
            np.float32(
                np.c_[
                    xrav1,
                    (ocp.xmin[1] + ocp.xmax[1]) / 2 * np.ones(xrav1.shape[0]),
                    yrav1,
                    np.zeros(yrav1.shape[0]),
                ]
            )
        )
        inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx1.shape)
        plt.contourf(xx1, yy1, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        xit = []
        yit = []
        cit = []
        for i in range(len(X_iter)):
            if (
                norm(X_iter[i][1] - (ocp.xmin[1] + ocp.xmax[1]) / 2) < 0.1
                and norm(X_iter[i][3]) < 0.01
            ):
                xit.append(X_iter[i][0])
                yit.append(X_iter[i][2])
                if y_iter[i] == [1, 0]:
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
        plt.xlim([ocp.xmin[0], ocp.xmax[0]])
        plt.ylim([ocp.xmin[2], ocp.xmax[2]])
        plt.grid()
        plt.title("First actuator")

    # Active learning:
    k = 0  # iteration number
    etpmax = 1
    performance_history = []

    sigmoid = nn.Sigmoid()

    while not (etpmax < etp_stop or len(Xu_iter) == 0):

        if len(Xu_iter) < B:
            B = len(Xu_iter)

        with torch.no_grad():
            Xu_iter_tensor = torch.Tensor(Xu_iter)
            Xu_iter_tensor = (Xu_iter_tensor - mean) / std
            prob_xu = sigmoid(model(Xu_iter_tensor))
            etp = entropy(prob_xu, axis=1)

        maxindex = np.argpartition(etp, -B)[
            -B:
        ].tolist()  # indexes of the uncertain samples
        maxindex.sort(reverse=True)

        etpmax = max(etp[maxindex])  # max entropy used for the stopping condition
        performance_history.append(etpmax)

        k += 1

        # Add the B most uncertain samples to the labeled set:
        for x in range(B):
            x0 = Xu_iter.pop(maxindex[x])
            q0 = x0[:2]
            v0 = x0[2:]

            # Data testing:
            res = ocp.compute_problem(q0, v0)
            if res == 1:
                X_iter.append([q0[0], q0[1], v0[0], v0[1]])
                y_iter.append([0, 1])
            elif res == 0:
                X_iter.append([q0[0], q0[1], v0[0], v0[1]])
                y_iter.append([1, 0])

        print("CLASSIFIER", k, "IN TRAINING")

        it = 0
        val = 1

        # Train the model
        while val > loss_stop and it <= it_max:
            ind = random.sample(range(len(X_iter) - B), int(n_minibatch / 2))
            ind.extend(
                random.sample(
                    range(len(X_iter) - B, len(X_iter)),
                    int(n_minibatch / 2),
                )
            )

            X_iter_tensor = torch.Tensor([X_iter[i] for i in ind])
            y_iter_tensor = torch.Tensor([y_iter[i] for i in ind])
            X_iter_tensor = (X_iter_tensor - mean) / std

            # Zero the gradients
            for param in model.parameters():
                param.grad = None

            # Forward pass
            outputs = model(X_iter_tensor)
            loss = criterion(outputs, y_iter_tensor)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            val = beta * val + (1 - beta) * loss.item()

            it += 1

        print("CLASSIFIER", k, "TRAINED")

        print("etpmax:", etpmax)

    with torch.no_grad():
        # Plot the results:
        plt.figure()
        inp = torch.from_numpy(
            np.float32(
                np.c_[
                    (ocp.xmin[0] + ocp.xmax[0]) / 2 * np.ones(xrav2.shape[0]),
                    xrav2,
                    np.zeros(yrav2.shape[0]),
                    yrav2,
                ]
            )
        )
        inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx2.shape)
        plt.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        xit = []
        yit = []
        cit = []
        for i in range(len(X_iter)):
            if (
                norm(X_iter[i][0] - (ocp.xmin[0] + ocp.xmax[0]) / 2) < 0.1
                and norm(X_iter[i][2]) < 0.01
            ):
                xit.append(X_iter[i][1])
                yit.append(X_iter[i][3])
                if y_iter[i] == [1, 0]:
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
        plt.xlim([ocp.xmin[1], ocp.xmax[1]])
        plt.ylim([ocp.xmin[3], ocp.xmax[3]])
        plt.grid()
        plt.title("Second actuator")

        plt.figure()
        inp = torch.from_numpy(
            np.float32(
                np.c_[
                    xrav1,
                    (ocp.xmin[1] + ocp.xmax[1]) / 2 * np.ones(xrav1.shape[0]),
                    yrav1,
                    np.zeros(yrav1.shape[0]),
                ]
            )
        )
        inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx1.shape)
        plt.contourf(xx1, yy1, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        xit = []
        yit = []
        cit = []
        for i in range(len(X_iter)):
            if (
                norm(X_iter[i][1] - (ocp.xmin[1] + ocp.xmax[1]) / 2) < 0.1
                and norm(X_iter[i][3]) < 0.01
            ):
                xit.append(X_iter[i][0])
                yit.append(X_iter[i][2])
                if y_iter[i] == [1, 0]:
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
        plt.xlim([ocp.xmin[0], ocp.xmax[0]])
        plt.ylim([ocp.xmin[2], ocp.xmax[2]])
        plt.grid()
        plt.title("First actuator")

        plt.figure()
        plt.plot(performance_history)
        plt.scatter(range(len(performance_history)), performance_history)
        plt.xlabel("Iteration number")
        plt.ylabel("Maximum entropy")
        plt.title("Maximum entropy evolution")
        
        torch.save(model, 'model_save')
       

    print("Execution time: %s seconds" % (time.time() - start_time))

pr.print_stats(sort="cumtime")

plt.show()
