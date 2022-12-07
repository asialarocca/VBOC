import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
import time
from double_pendulum_ocp_class import OCPdoublependulumINIT
import random
import warnings
import torch
import torch.nn as nn
from my_nn import NeuralNet
from statistics import mean
import math
from torch.utils.data import TensorDataset, DataLoader

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
    N_init = pow(5, ocp_dim)  # size of initial labeled set
    B = pow(5, ocp_dim)  # batch size
    etp_stop = 0.5  # active learning stopping condition
    loss_stop = 0.1  # nn training stopping condition
    beta = 0.8
    n_minibatch = 512
    it_max = int(100 * B / n_minibatch)
    gridp = 20

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(gridp, ocp_dim))
    l_bounds = [ocp.xmin[0]-0.05, ocp.xmin[1]-0.05, ocp.xmin[2], ocp.xmin[3]]
    u_bounds = [ocp.xmax[0]+0.05, ocp.xmax[1]+0.05, ocp.xmax[2], ocp.xmax[3]]
    Xu_iter = qmc.scale(sample, l_bounds, u_bounds).tolist()

    # Get solution of positive samples:
    simX_vec = np.ndarray((0, ocp.N + 1, ocp.nx))
    # simX_vec.fill(0.0)
    simU_vec = np.ndarray((0, ocp.N, ocp.nu))
    # simU_vec.fill(0.0)
    simX = np.ndarray((ocp.N + 1, ocp.nx))
    simU = np.ndarray((ocp.N, ocp.nu))

    # Generate the initial set of labeled samples:
    X_iter = [[(ocp.xmax[0] + ocp.xmin[0]) / 2, (ocp.xmax[1] + ocp.xmin[1]) / 2, 0.0, 0.0]]
    res = ocp.compute_problem([(ocp.xmax[0] + ocp.xmin[0]) / 2,
                              (ocp.xmax[1] + ocp.xmin[1]) / 2], [0.0, 0.0])
    if res == 1:
        y_iter = [[0, 1]]
    else:
        y_iter = [[1, 0]]

    Xu_iter_tensor = torch.Tensor(Xu_iter)
    mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

    # Training of an initial classifier:
    for n in range(N_init):
        q0 = Xu_iter[n][:2]
        v0 = Xu_iter[n][2:]

        # Data testing:
        res = ocp.compute_problem(q0, v0)
        if res == 1:
            X_iter.append([q0[0], q0[1], v0[0], v0[1]])
            y_iter.append([0, 1])

            for i in range(ocp.N):
                simX[i, :] = ocp.ocp_solver.get(i, "x")
                simU[i, :] = ocp.ocp_solver.get(i, "u")
            simX[ocp.N, :] = ocp.ocp_solver.get(ocp.N, "x")

            simX_vec = np.append(simX_vec, [simX], axis=0)
            simU_vec = np.append(simU_vec, [simU], axis=0)

        elif res == 0:
            X_iter.append([q0[0], q0[1], v0[0], v0[1]])
            y_iter.append([1, 0])

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

    # Active learning:
    k = 0  # iteration number
    etpmax = 1
    performance_history = []

    sigmoid = nn.Sigmoid()

    while not (etpmax < etp_stop or len(Xu_iter) == 0):

        if len(Xu_iter) < B:
            B = len(Xu_iter)

        etp = np.empty((len(Xu_iter),))

        with torch.no_grad():
            Xu_iter_tensor = torch.Tensor(Xu_iter)
            Xu_iter_tensor = (Xu_iter_tensor - mean) / std
            my_data = DataLoader(Xu_iter_tensor, batch_size=n_minibatch, shuffle=False)
            for (idx, batch) in enumerate(my_data):
                if n_minibatch*(idx+1) > len(Xu_iter):
                    prob_xu = sigmoid(model(batch))
                    etp[n_minibatch*idx:len(Xu_iter)] = entropy(prob_xu, axis=1)
                else:
                    prob_xu = sigmoid(model(batch))
                    etp[n_minibatch*idx:n_minibatch*(idx+1)] = entropy(prob_xu, axis=1)

        maxindex = np.argpartition(etp, -B)[
            -B:
        ].tolist()  # indexes of the uncertain samples
        maxindex.sort(reverse=True)

        etpmax = max(etp[maxindex])  # max entropy used for the stopping condition
        performance_history.append(etpmax)

        k += 1

        # Add the B most uncertain samples to the labeled set:
        for x in range(B):
            x0 = Xu_iter[maxindex[x]]
            del Xu_iter[maxindex[x]]
            q0 = x0[:2]
            v0 = x0[2:]

            # Data testing:
            res = ocp.compute_problem_withGUESS(q0, v0,  simX_vec, simU_vec)
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

    # rad_q = 1 / gridp
    # rad_v = 1 / gridp

    # n_points = ocp_dim

    # etp = np.empty((len(X_iter),))

    # with torch.no_grad():
    #     X_tensor = torch.Tensor(X_iter)
    #     X_tensor = (X_tensor - mean) / std
    #     my_dataloader = DataLoader(X_tensor,batch_size=n_minibatch,shuffle=False)
    #     for (idx, batch) in enumerate(my_dataloader):
    #         if n_minibatch*(idx+1) > len(Xu_iter):
    #             prob_x = sigmoid(model(batch))
    #             etp[n_minibatch*idx:len(X_iter)] = entropy(prob_x, axis=1)
    #         else:
    #             prob_x = sigmoid(model(batch))
    #             etp[n_minibatch*idx:n_minibatch*(idx+1)] = entropy(prob_x, axis=1)

    # xu = np.array(X_iter)[etp > etp_stop]

    # Xu_it = np.empty((xu.shape[0], n_points, ocp_dim))

    # # Generate other random samples:
    # for i in range(xu.shape[0]):
    #     for n in range(n_points):

    #         # random angle
    #         alpha = math.pi * random.random()
    #         beta = math.pi * random.random()
    #         gamma = 2 * math.pi * random.random()
    #         # random radius
    #         tmp = math.sqrt(random.random())
    #         r_x = rad_q * tmp
    #         r_y = rad_v * tmp
    #         # calculating coordinates
    #         x1 = r_x * math.cos(alpha) + xu[i, 0]
    #         x2 = r_x * math.sin(alpha) * math.cos(beta) + xu[i, 1]
    #         x3 = r_y * math.sin(alpha) * math.sin(beta) * math.cos(gamma) + xu[i, 2]
    #         x4 = r_y * math.sin(alpha) * math.sin(beta) * math.sin(gamma) + xu[i, 3]

    #         Xu_it[i, n, :] = [x1, x2, x3, x4]

    # Xu_it.shape = (xu.shape[0] * n_points, ocp_dim)
    # Xu_iter = Xu_it.tolist()

    # etpmax = 1

    # while not (etpmax < etp_stop or len(Xu_iter) == 0):

    #     if len(Xu_iter) < B:
    #         B = len(Xu_iter)

    #     etp = np.empty((len(Xu_iter),))

    #     with torch.no_grad():
    #         Xu_iter_tensor = torch.Tensor(Xu_iter)
    #         Xu_iter_tensor = (Xu_iter_tensor - mean) / std
    #         my_data = DataLoader(Xu_iter_tensor,batch_size=n_minibatch,shuffle=False)
    #         for (idx,batch) in enumerate(my_data):
    #             if n_minibatch*(idx+1) > len(Xu_iter):
    #                 prob_xu = sigmoid(model(batch))
    #                 etp[n_minibatch*idx:len(Xu_iter)] = entropy(prob_xu, axis=1)
    #             else:
    #                 prob_xu = sigmoid(model(batch))
    #                 etp[n_minibatch*idx:n_minibatch*(idx+1)] = entropy(prob_xu, axis=1)

    #     maxindex = np.argpartition(etp, -B)[
    #         -B:
    #     ].tolist()  # indexes of the uncertain samples
    #     maxindex.sort(reverse=True)

    #     etpmax = max(etp[maxindex])  # max entropy used for the stopping condition
    #     performance_history.append(etpmax)

    #     k += 1

    #     # Add the B most uncertain samples to the labeled set:
    #     for x in range(B):
    #         x0 = Xu_iter[maxindex[x]]
    #         del Xu_iter[maxindex[x]]
    #         q0 = x0[:2]
    #         v0 = x0[2:]

    #         # Data testing:
    #         res = ocp.compute_problem(q0, v0)
    #         if res == 1:
    #             X_iter.append([q0[0], q0[1], v0[0], v0[1]])
    #             y_iter.append([0, 1])
    #         elif res == 0:
    #             X_iter.append([q0[0], q0[1], v0[0], v0[1]])
    #             y_iter.append([1, 0])

    #     print("CLASSIFIER", k, "IN TRAINING")

    #     it = 0
    #     val = 1

    #     # Train the model
    #     while val > loss_stop and it <= it_max:
    #         ind = random.sample(range(len(X_iter) - B), int(n_minibatch / 2))
    #         ind.extend(
    #             random.sample(
    #                 range(len(X_iter) - B, len(X_iter)),
    #                 int(n_minibatch / 2),
    #             )
    #         )

    #         X_iter_tensor = torch.Tensor([X_iter[i] for i in ind])
    #         y_iter_tensor = torch.Tensor([y_iter[i] for i in ind])
    #         X_iter_tensor = (X_iter_tensor - mean) / std

    #         # Zero the gradients
    #         for param in model.parameters():
    #             param.grad = None

    #         # Forward pass
    #         outputs = model(X_iter_tensor)
    #         loss = criterion(outputs, y_iter_tensor)

    #         # Backward and optimize
    #         loss.backward()
    #         optimizer.step()

    #         val = beta * val + (1 - beta) * loss.item()

    #         it += 1

    #     print("CLASSIFIER", k, "TRAINED")

    #     print("etpmax:", etpmax)

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
                norm(X_iter[i][0] - (ocp.xmin[0] + ocp.xmax[0]) / 2) < 0.01
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
                norm(X_iter[i][1] - (ocp.xmin[1] + ocp.xmax[1]) / 2) < 0.01
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

        with open('mean.txt', 'w') as f:
            f.write(str(mean.item()))

        with open('std.txt', 'w') as f:
            f.write(str(std.item()))

        np.save('X_iter.npy', np.asarray(X_iter))

        np.save('y_iter.npy', np.asarray(y_iter))

        X_tensor = (torch.Tensor(X_iter) - mean) / std
        out = model(X_tensor)
        y_pred = torch.from_numpy(np.argmax(out.numpy(), axis=1))
        target = torch.Tensor(np.argmax(y_iter, axis=1))

        train_acc = torch.sum(y_pred == target)/len(X_iter)

        print(train_acc)

    print("Execution time: %s seconds" % (time.time() - start_time))

pr.print_stats(sort="cumtime")

plt.show()
