import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from numpy.linalg import norm as norm
import time
from double_pendulum_ocp_class import OCPdoublependulumINIT, OCPdoublependulumNN
import random
import math
import warnings
import torch
import torch.nn as nn
from my_nn import NeuralNet
from torch.utils.data import TensorDataset, DataLoader
from statistics import mean

warnings.filterwarnings("ignore")


with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPdoublependulumINIT()

    ocp_dim = ocp.nx

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
    etp_stop = 0.1  # active learning stopping condition
    loss_stop = 0.1  # nn training stopping condition
    beta = 0.8
    n_minibatch = 1024
    it_max = 100 * B / n_minibatch

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(50, ocp_dim))
    l_bounds = [q_min, q_min, v_min, v_min]
    u_bounds = [q_max, q_max, v_max, v_max]
    Xu_iter = qmc.scale(sample, l_bounds, u_bounds).tolist()

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

    X_iter[:, 0] = np.float32(x[:, 0] * (q_max + 0.05 - (q_min - 0.05)) + q_min - 0.05)
    X_iter[:, 1] = np.float32(x[:, 1] * (q_max + 0.05 - (q_min - 0.05)) + q_min - 0.05)
    X_iter[:, 2] = np.float32(x[:, 2] * (v_max + 0.5 - (v_min - 0.5)) + v_min - 0.5)
    X_iter[:, 3] = np.float32(x[:, 3] * (v_max + 0.5 - (v_min - 0.5)) + v_min - 0.5)

    X_iter = X_iter.tolist()
    y_iter = y_iter.tolist()

    # # Generate the initial set of labeled samples:
    # X_iter = [[(q_max + q_min) / 2, (q_max + q_min) / 2, 0.0, 0.0]]
    # res = ocp.compute_problem([(q_max + q_min) / 2, (q_max + q_min) / 2], [0.0, 0.0])
    # if res == 1:
    #     y_iter = [[0, 1]]
    # else:
    #     y_iter = [[1, 0]]

    # Xu_iter_tensor = torch.Tensor(Xu_iter)
    # mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

    # # Get solution of positive samples:
    # N = ocp.N
    # count = 0
    # N_guess = 100
    # simX = np.ndarray((N_guess, N + 1, ocp_dim))
    # simU = np.ndarray((N_guess, N, int(ocp_dim / 2)))

    # Training of an initial classifier:
    for n in range(N_init):
        q0 = Xu_iter[n][:2]
        v0 = Xu_iter[n][2:]

        # Data testing:
        res = ocp.compute_problem(q0, v0)
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

        # if count < N_guess:
        #     for i in range(N):
        #         simX[count, i, :] = ocp.ocp_solver.get(i, "x")
        #         simU[count, i] = ocp.ocp_solver.get(i, "u")
        #     simX[count, N, :] = ocp.ocp_solver.get(N, "x")

        # count += 1

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
        # X_iter_tensor = (X_iter_tensor - mean) / std

        # Forward pass
        outputs = model(X_iter_tensor)
        loss = criterion(outputs, y_iter_tensor)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        for param in model.parameters():
            param.grad = None

        # ind = random.sample(range(X_iter.shape[0]), n_minibatch)
        # X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32))
        # y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32))
        # X_iter_tensor = (X_iter_tensor - mean) / std

        # loss = criterion(model(X_iter_tensor), y_iter_tensor)

        val = beta * val + (1 - beta) * loss.item()

        it += 1

    # val = 1

    # X_iter_tensor = torch.from_numpy(X_iter.astype(np.float32))
    # y_iter_tensor = torch.from_numpy(y_iter.astype(np.float32))
    # X_iter_tensor = (X_iter_tensor - mean) / std

    # my_dataset = TensorDataset(X_iter_tensor, y_iter_tensor)
    # my_dataloader = DataLoader(my_dataset, batch_size=n_minibatch, shuffle=True)

    # for it in range(n_epoch):  # loop over the dataset multiple times

    #     for i, data in enumerate(my_dataloader):
    #         inputs, labels = data

    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         val = beta * val + (1 - beta) * loss.item()

    #         if val <= loss_stop:
    #             break

    #     if val <= loss_stop:
    #         break

    print("INITIAL CLASSIFIER TRAINED")

    # Plots:
    h = 0.02
    xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()

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
        )
        # inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
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
        plt.xlim([q_min, q_max - 0.01])
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
        )
        # inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
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
        plt.xlim([q_min, q_max - 0.01])
        plt.ylim([v_min, v_max])
        plt.grid()
        plt.title("First actuator")

    # Active learning:
    k = 0  # iteration number
    etpmax = 1
    performance_history = []

    while not (etpmax < etp_stop or len(Xu_iter) == 0):

        if len(Xu_iter) < B:
            B = len(Xu_iter)

        with torch.no_grad():
            sigmoid = nn.Sigmoid()
            Xu_iter_tensor = torch.Tensor(Xu_iter)
            # Xu_iter_tensor = (Xu_iter_tensor - mean) / std
            prob_xu = sigmoid(model(Xu_iter_tensor))
            etp = entropy(prob_xu, axis=1)

        maxindex = np.argpartition(etp, -B)[
            -B:
        ].tolist()  # indexes of the uncertain samples
        maxindex.sort(reverse=True)

        etpmax = max(etp[maxindex])  # max entropy used for the stopping condition
        performance_history.append(etpmax)

        k += 1
        # count = 0

        # Add the B most uncertain samples to the labeled set:
        for x in range(B):
            x0 = Xu_iter.pop(maxindex[x])
            q0 = x0[:2]
            v0 = x0[2:]

            # Data testing:
            res = ocp.compute_problem(q0, v0)
            # res = ocp.compute_problem_withGUESS(q0, v0, simX, simU)
            # res = ocp.compute_problem_withGUESSPID(q0, v0)
            if res == 1:
                X_iter.append([q0[0], q0[1], v0[0], v0[1]])
                y_iter.append([0, 1])
            elif res == 0:
                X_iter.append([q0[0], q0[1], v0[0], v0[1]])
                y_iter.append([1, 0])

            # # Add intermediate states of succesfull initial conditions:
            # if res == 1:
            #     if count < N_guess:
            #         for i in range(N):
            #             simX[count, i, :] = ocp.ocp_solver.get(i, "x")
            #             simU[count, i] = ocp.ocp_solver.get(i, "u")
            #         simX[count, N, :] = ocp.ocp_solver.get(N, "x")

            #     count += 1

        # # Delete tested data from the unlabeled set:
        # for i in sorted(maxindex, reverse=True):
        #     del Xu_iter[i]

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
            # X_iter_tensor = (X_iter_tensor - mean) / std

            # Forward pass
            outputs = model(X_iter_tensor)
            loss = criterion(outputs, y_iter_tensor)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            for param in model.parameters():
                param.grad = None

            # ind = random.sample(range(X_iter.shape[0]), n_minibatch)
            # X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32))
            # y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32))
            # X_iter_tensor = (X_iter_tensor - mean) / std

            # loss = criterion(model(X_iter_tensor), y_iter_tensor)

            val = beta * val + (1 - beta) * loss.item()

            it += 1

        # val = 1

        # selec = [i for i in range(X_iter.shape[0] - B) if np.random.uniform() <= 1 / k]
        # selec.extend([i for i in range(X_iter.shape[0] - B, X_iter.shape[0])])

        # X_iter_tensor = torch.from_numpy(X_iter[selec].astype(np.float32))
        # X_iter_tensor = (X_iter_tensor - mean) / std
        # y_iter_tensor = torch.from_numpy(y_iter[selec].astype(np.float32))

        # my_dataset = TensorDataset(X_iter_tensor, y_iter_tensor)
        # my_dataloader = DataLoader(my_dataset, batch_size=n_minibatch, shuffle=True)

        # for it in range(n_epoch):  # loop over the dataset multiple times
        #     for i, data in enumerate(my_dataloader):
        #         inputs, labels = data

        #         # zero the parameter gradients
        #         optimizer.zero_grad()

        #         # forward + backward + optimize
        #         outputs = model(inputs)
        #         loss = criterion(outputs, labels)
        #         loss.backward()
        #         optimizer.step()

        #         val = beta * val + (1 - beta) * loss.item()

        #         if val <= loss_stop:
        #             break

        #     if val <= loss_stop:
        #         break

        print("CLASSIFIER", k, "TRAINED")

        print("etpmax:", etpmax)

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
        )
        # inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
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
        plt.xlim([q_min, q_max - 0.01])
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
        )
        # inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
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
        plt.xlim([q_min, q_max - 0.01])
        plt.ylim([v_min, v_max])
        plt.grid()
        plt.title("First actuator")

    plt.figure()
    plt.plot(performance_history)
    plt.scatter(range(len(performance_history)), performance_history)
    plt.xlabel("Iteration number")
    plt.ylabel("Maximum entropy")
    plt.title("Maximum entropy evolution")

    # del ocp

    # # Ocp initialization:
    # ocp = OCPdoublependulumNN(model)  # , mean.item(), std.item())

    # indx = np.array(0)
    # indxu = np.array(0)
    # for i in range(X_iter.shape[0]):
    #     if np.array_equal(y_iter[i], [0, 1]):
    #         indx = np.append(indx, i)
    #     else:
    #         indxu = np.append(indxu, i)

    # data_xinit = X_iter[indxu[1:]]
    # data_xuinit = Xu_iter[
    #     np.argmax(model(torch.from_numpy(Xu_iter)).detach().numpy(), axis=1) == 0
    # ]
    # Xu_iter = np.concatenate([data_xinit, data_xuinit])
    # X_iter = X_iter[indx[1:]]
    # y_iter = y_iter[indx[1:]]
    # Xu_iter_tensor = torch.from_numpy(Xu_iter)
    # # mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

    # # Training of an initial classifier:
    # for n in range(N_init):
    #     q0 = Xu_iter[n, :2]
    #     v0 = Xu_iter[n, 2:]

    #     # Data testing:
    #     res = ocp.compute_problem(q0, v0)
    #     if res == 1:
    #         X_iter = np.append(X_iter, [[q0[0], q0[1], v0[0], v0[1]]], axis=0)
    #         y_iter = np.append(y_iter, [[0, 1]], axis=0)
    #     elif res == 0:
    #         X_iter = np.append(X_iter, [[q0[0], q0[1], v0[0], v0[1]]], axis=0)
    #         y_iter = np.append(y_iter, [[1, 0]], axis=0)

    #     # Add intermediate states of succesfull initial conditions:
    #     if res == 1:
    #         for f in range(1, ocp.N, int(ocp.N / 3)):
    #             current_val = np.float32(ocp.ocp_solver.get(f, "x"))
    #             X_iter = np.append(X_iter, [current_val], axis=0)
    #             y_iter = np.append(y_iter, [[0, 1]], axis=0)

    #         # if count < N_guess:
    #         #     for i in range(N):
    #         #         simX[count, i, :] = ocp.ocp_solver.get(i, "x")
    #         #         simU[count, i] = ocp.ocp_solver.get(i, "u")
    #         #     simX[count, N, :] = ocp.ocp_solver.get(N, "x")

    #         # count += 1

    # # Delete tested data from the unlabeled set:
    # Xu_iter = np.delete(Xu_iter, range(N_init), axis=0)

    # print("INITIAL CLASSIFIER IN TRAINING")

    # it = 0
    # val = 1

    # # Train the model
    # while val > loss_stop and it <= it_max:

    #     ind = random.sample(range(X_iter.shape[0]), n_minibatch)
    #     X_iter_tensor = torch.from_numpy(X_iter[ind])
    #     y_iter_tensor = torch.from_numpy(y_iter[ind])
    #     # X_iter_tensor = (X_iter_tensor - mean) / std

    #     # Forward pass
    #     outputs = model(X_iter_tensor)
    #     loss = criterion(outputs, y_iter_tensor)

    #     # Backward and optimize
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()

    #     # ind = random.sample(range(X_iter.shape[0]), n_minibatch)
    #     # X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32))
    #     # y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32))
    #     # X_iter_tensor = (X_iter_tensor - mean) / std

    #     # loss = criterion(model(X_iter_tensor), y_iter_tensor)

    #     val = beta * val + (1 - beta) * loss.item()

    #     it += 1

    # # val = 1

    # # X_iter_tensor = torch.from_numpy(X_iter.astype(np.float32))
    # # y_iter_tensor = torch.from_numpy(y_iter.astype(np.float32))
    # # X_iter_tensor = (X_iter_tensor - mean) / std

    # # my_dataset = TensorDataset(X_iter_tensor, y_iter_tensor)
    # # my_dataloader = DataLoader(my_dataset, batch_size=n_minibatch, shuffle=True)

    # # for it in range(n_epoch):  # loop over the dataset multiple times

    # #     for i, data in enumerate(my_dataloader):
    # #         inputs, labels = data

    # #         # zero the parameter gradients
    # #         optimizer.zero_grad()

    # #         # forward + backward + optimize
    # #         outputs = model(inputs)
    # #         loss = criterion(outputs, labels)
    # #         loss.backward()
    # #         optimizer.step()

    # #         val = beta * val + (1 - beta) * loss.item()

    # #         if val <= loss_stop:
    # #             break

    # #     if val <= loss_stop:
    # #         break

    # print("INITIAL CLASSIFIER TRAINED")

    # # Plots:
    # h = 0.02
    # xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))
    # xrav = xx.ravel()
    # yrav = yy.ravel()

    # with torch.no_grad():
    #     # Plot the results:
    #     plt.figure()
    #     inp = torch.from_numpy(
    #         np.float32(
    #             np.c_[
    #                 (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
    #                 xrav,
    #                 np.zeros(yrav.shape[0]),
    #                 yrav,
    #             ]
    #         )
    #     )
    #     # inp = (inp - mean) / std
    #     out = model(inp)
    #     y_pred = np.argmax(out.numpy(), axis=1)
    #     Z = y_pred.reshape(xx.shape)
    #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #     Xit = [X_iter[0, :]]
    #     yit = [0]
    #     for i in range(X_iter.shape[0]):
    #         if (
    #             norm(X_iter[i, 0] - (q_min + q_max) / 2) < 0.1
    #             and norm(X_iter[i, 2]) < 0.1
    #         ):
    #             Xit = np.append(Xit, [X_iter[i, :]], axis=0)
    #             if np.array_equal(y_iter[i], [1, 0]):
    #                 yit = np.append(yit, 0)
    #             else:
    #                 yit = np.append(yit, 1)
    #     plt.scatter(
    #         Xit[1:, 1],
    #         Xit[1:, 3],
    #         c=yit[1:],
    #         marker=".",
    #         alpha=0.5,
    #         cmap=plt.cm.Paired,
    #     )
    #     plt.xlim([q_min, q_max - 0.01])
    #     plt.ylim([v_min, v_max])
    #     plt.grid()
    #     plt.title("Second actuator")

    #     plt.figure()
    #     inp = torch.from_numpy(
    #         np.float32(
    #             np.c_[
    #                 xrav,
    #                 (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
    #                 yrav,
    #                 np.zeros(yrav.shape[0]),
    #             ]
    #         )
    #     )
    #     # inp = (inp - mean) / std
    #     out = model(inp)
    #     y_pred = np.argmax(out.numpy(), axis=1)
    #     Z = y_pred.reshape(xx.shape)
    #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #     Xit = [X_iter[0, :]]
    #     yit = [0]
    #     for i in range(X_iter.shape[0]):
    #         if (
    #             norm(X_iter[i, 1] - (q_min + q_max) / 2) < 0.1
    #             and norm(X_iter[i, 3]) < 0.1
    #         ):
    #             Xit = np.append(Xit, [X_iter[i, :]], axis=0)
    #             if np.array_equal(y_iter[i], [1, 0]):
    #                 yit = np.append(yit, 0)
    #             else:
    #                 yit = np.append(yit, 1)
    #     plt.scatter(
    #         Xit[1:, 0],
    #         Xit[1:, 2],
    #         c=yit[1:],
    #         marker=".",
    #         alpha=0.5,
    #         cmap=plt.cm.Paired,
    #     )
    #     plt.xlim([q_min, q_max - 0.01])
    #     plt.ylim([v_min, v_max])
    #     plt.grid()
    #     plt.title("First actuator")

    # # Active learning:
    # k = 0  # iteration number
    # etpmax = 1
    # performance_history = [etpmax]

    # while not (etpmax < etp_stop or Xu_iter.shape[0] == 0):

    #     if Xu_iter.shape[0] < B:
    #         B = Xu_iter.shape[0]

    #     with torch.no_grad():
    #         sigmoid = nn.Sigmoid()
    #         Xu_iter_tensor = torch.from_numpy(Xu_iter)
    #         # Xu_iter_tensor = (Xu_iter_tensor - mean) / std
    #         prob_xu = sigmoid(model(Xu_iter_tensor)).numpy()
    #         etp = entropy(prob_xu, axis=1)

    #     maxindex = np.argpartition(etp, -B)[-B:]  # indexes of the uncertain samples

    #     etpmax = max(etp[maxindex])  # max entropy used for the stopping condition
    #     performance_history.append(etpmax)

    #     k += 1
    #     # count = 0

    #     # Add the B most uncertain samples to the labeled set:
    #     for x in range(B):
    #         q0 = Xu_iter[maxindex[x], :2]
    #         v0 = Xu_iter[maxindex[x], 2:]

    #         # Data testing:
    #         res = ocp.compute_problem(q0, v0)
    #         # res = ocp.compute_problem_withGUESS(q0, v0, simX, simU)
    #         if res == 1:
    #             X_iter = np.append(X_iter, [[q0[0], q0[1], v0[0], v0[1]]], axis=0)
    #             y_iter = np.append(y_iter, [[0, 1]], axis=0)
    #         elif res == 0:
    #             X_iter = np.append(X_iter, [[q0[0], q0[1], v0[0], v0[1]]], axis=0)
    #             y_iter = np.append(y_iter, [[1, 0]], axis=0)

    #         # # Add intermediate states of succesfull initial conditions:
    #         # if res == 1:
    #         #     if count < N_guess:
    #         #         for i in range(N):
    #         #             simX[count, i, :] = ocp.ocp_solver.get(i, "x")
    #         #             simU[count, i] = ocp.ocp_solver.get(i, "u")
    #         #         simX[count, N, :] = ocp.ocp_solver.get(N, "x")

    #         #     count += 1

    #     # Delete tested data from the unlabeled set:
    #     Xu_iter = np.delete(Xu_iter, maxindex, axis=0)

    #     print("CLASSIFIER", k, "IN TRAINING")

    #     it = 0
    #     val = 1

    #     # Train the model
    #     while val > loss_stop and it <= it_max:

    #         ind = random.sample(range(X_iter.shape[0] - B), int(n_minibatch / 2))
    #         ind.extend(
    #             random.sample(
    #                 range(X_iter.shape[0] - B, X_iter.shape[0]),
    #                 int(n_minibatch / 2),
    #             )
    #         )
    #         X_iter_tensor = torch.from_numpy(X_iter[ind])
    #         y_iter_tensor = torch.from_numpy(y_iter[ind])
    #         # X_iter_tensor = (X_iter_tensor - mean) / std

    #         # Forward pass
    #         outputs = model(X_iter_tensor)
    #         loss = criterion(outputs, y_iter_tensor)

    #         # Backward and optimize
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()

    #         # ind = random.sample(range(X_iter.shape[0]), n_minibatch)
    #         # X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32))
    #         # y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32))
    #         # X_iter_tensor = (X_iter_tensor - mean) / std

    #         # loss = criterion(model(X_iter_tensor), y_iter_tensor)

    #         val = beta * val + (1 - beta) * loss.item()

    #         it += 1

    #     # val = 1

    #     # selec = [i for i in range(X_iter.shape[0] - B) if np.random.uniform() <= 1 / k]
    #     # selec.extend([i for i in range(X_iter.shape[0] - B, X_iter.shape[0])])

    #     # X_iter_tensor = torch.from_numpy(X_iter[selec].astype(np.float32))
    #     # X_iter_tensor = (X_iter_tensor - mean) / std
    #     # y_iter_tensor = torch.from_numpy(y_iter[selec].astype(np.float32))

    #     # my_dataset = TensorDataset(X_iter_tensor, y_iter_tensor)
    #     # my_dataloader = DataLoader(my_dataset, batch_size=n_minibatch, shuffle=True)

    #     # for it in range(n_epoch):  # loop over the dataset multiple times
    #     #     for i, data in enumerate(my_dataloader):
    #     #         inputs, labels = data

    #     #         # zero the parameter gradients
    #     #         optimizer.zero_grad()

    #     #         # forward + backward + optimize
    #     #         outputs = model(inputs)
    #     #         loss = criterion(outputs, labels)
    #     #         loss.backward()
    #     #         optimizer.step()

    #     #         val = beta * val + (1 - beta) * loss.item()

    #     #         if val <= loss_stop:
    #     #             break

    #     #     if val <= loss_stop:
    #     #         break

    #     print("CLASSIFIER", k, "TRAINED")

    #     print("etpmax:", etpmax)

    # with torch.no_grad():
    #     # Plot the results:
    #     plt.figure()
    #     inp = torch.from_numpy(
    #         np.float32(
    #             np.c_[
    #                 (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
    #                 xrav,
    #                 np.zeros(yrav.shape[0]),
    #                 yrav,
    #             ]
    #         )
    #     )
    #     # inp = (inp - mean) / std
    #     out = model(inp)
    #     y_pred = np.argmax(out.numpy(), axis=1)
    #     Z = y_pred.reshape(xx.shape)
    #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #     Xit = [X_iter[0, :]]
    #     yit = [0]
    #     for i in range(X_iter.shape[0]):
    #         if (
    #             norm(X_iter[i, 0] - (q_min + q_max) / 2) < 0.1
    #             and norm(X_iter[i, 2]) < 0.1
    #         ):
    #             Xit = np.append(Xit, [X_iter[i, :]], axis=0)
    #             if np.array_equal(y_iter[i], [1, 0]):
    #                 yit = np.append(yit, 0)
    #             else:
    #                 yit = np.append(yit, 1)
    #     plt.scatter(
    #         Xit[1:, 1],
    #         Xit[1:, 3],
    #         c=yit[1:],
    #         marker=".",
    #         alpha=0.5,
    #         cmap=plt.cm.Paired,
    #     )
    #     plt.xlim([q_min, q_max - 0.01])
    #     plt.ylim([v_min, v_max])
    #     plt.grid()
    #     plt.title("Second actuator")

    #     plt.figure()
    #     inp = torch.from_numpy(
    #         np.float32(
    #             np.c_[
    #                 xrav,
    #                 (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
    #                 yrav,
    #                 np.zeros(yrav.shape[0]),
    #             ]
    #         )
    #     )
    #     # inp = (inp - mean) / std
    #     out = model(inp)
    #     y_pred = np.argmax(out.numpy(), axis=1)
    #     Z = y_pred.reshape(xx.shape)
    #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #     Xit = [X_iter[0, :]]
    #     yit = [0]
    #     for i in range(X_iter.shape[0]):
    #         if (
    #             norm(X_iter[i, 1] - (q_min + q_max) / 2) < 0.1
    #             and norm(X_iter[i, 3]) < 0.1
    #         ):
    #             Xit = np.append(Xit, [X_iter[i, :]], axis=0)
    #             if np.array_equal(y_iter[i], [1, 0]):
    #                 yit = np.append(yit, 0)
    #             else:
    #                 yit = np.append(yit, 1)
    #     plt.scatter(
    #         Xit[1:, 0],
    #         Xit[1:, 2],
    #         c=yit[1:],
    #         marker=".",
    #         alpha=0.5,
    #         cmap=plt.cm.Paired,
    #     )
    #     plt.xlim([q_min, q_max - 0.01])
    #     plt.ylim([v_min, v_max])
    #     plt.grid()
    #     plt.title("First actuator")

    print("Execution time: %s seconds" % (time.time() - start_time))

pr.print_stats(sort="cumtime")

plt.show()
