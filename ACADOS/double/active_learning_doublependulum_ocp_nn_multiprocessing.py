import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
import time
from double_pendulum_ocp_class import OCPdoublependulumINIT, OCPdoublependulumNN
import random
import math
import warnings
import torch
import torch.nn as nn
from my_nn import NeuralNet
from statistics import mean
from multiprocessing import Pool
from pstats import Stats

warnings.filterwarnings("ignore")

with cProfile.Profile() as pr:

    start_time = time.time()
    cpu_num = 4

    # Ocp initialization:
    ocp = OCPdoublependulumINIT()

    ocp_dim = ocp.nx

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = -ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    # test_grid = pow(100, 2)

    # q2 = np.random.uniform(q_min, q_max, size=test_grid)
    # q1 = np.random.uniform(q_min, q_max, size=test_grid)
    # v2 = np.random.uniform(v_min, v_max, size=test_grid)
    # v1 = np.random.uniform(v_min, v_max, size=test_grid)

    # delta_t = 0.01
    # M1 = ocp.l1**2 * (ocp.m1+ocp.m2)
    # M2 = ocp.l2**2 * ocp.m2

    # def func1(x, *arg):
    #     thetamin_new, q_dotdot_max, thetamax_new, q_dotdot_min = x
    #     q2, q2_dot = arg

    #     h_min = -math.sin(-q2+thetamin_new)*q2_dot**2*ocp.l1*ocp.l2*ocp.m2 - \
    #         ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(thetamin_new)

    #     h_max = -math.sin(-q2+thetamax_new)*q2_dot**2*ocp.l1*ocp.l2*ocp.m2 - \
    #         ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(thetamax_new)

    #     return (q_dotdot_max - (ocp.Cmax - h_min)/M1,
    #             - thetamin_new + q_min + 1/2 * delta_t**2 * q_dotdot_max,
    #             q_dotdot_min - (-ocp.Cmax - h_max)/M1,
    #             - thetamax_new + q_max + 1/2 * delta_t**2 * q_dotdot_min)

    # def func2(x, *arg):
    #     thetamin_new, q_dotdot_max, thetamax_new, q_dotdot_min = x
    #     q1, q1_dot = arg

    #     h_min = ocp.m2*math.sin(-thetamin_new+q1)*ocp.l1*q1_dot**2 * \
    #         ocp.l2-ocp.l2*math.sin(thetamin_new)*ocp.m2*ocp.g

    #     h_max = ocp.m2*math.sin(-thetamax_new+q1)*ocp.l1*q1_dot**2 * \
    #         ocp.l2-ocp.l2*math.sin(thetamax_new)*ocp.m2*ocp.g

    #     return (q_dotdot_max - (ocp.Cmax - h_min)/M2,
    #             - thetamin_new + q_min + 1/2 * delta_t**2 * q_dotdot_max,
    #             q_dotdot_min - (-ocp.Cmax - h_max)/M2,
    #             - thetamax_new + q_max + 1/2 * delta_t**2 * q_dotdot_min)

    # guess1 = (q_min, ocp.Cmax/M1, q_max, -ocp.Cmax/M1)
    # guess2 = (q_min, ocp.Cmax/M2, q_max, -ocp.Cmax/M2)

    # thetamin_new = [q_min, q_min]
    # thetamax_new = [q_max, q_max]

    # for i in range(test_grid):
    #     thetamin1, q_dotdot_max1, thetamax1, q_dotdot_min1 = fsolve(
    #         func1, guess1, args=(q2[i], v2[i]))
    #     thetamin2, q_dotdot_max2, thetamax2, q_dotdot_min2 = fsolve(
    #         func2, guess2, args=(q1[i], v1[i]))

    #     if thetamin1 > thetamin_new[0]:
    #         thetamin_new[0] = thetamin1
    #     if thetamax1 < thetamax_new[0]:
    #         thetamax_new[0] = thetamax1
    #     if thetamin2 > thetamin_new[1]:
    #         thetamin_new[1] = thetamin2
    #     if thetamax2 < thetamax_new[0]:
    #         thetamax_new[1] = thetamax2

    # print(q_min, thetamin_new, q_max, thetamax_new)

    # ocp.set_bounds(thetamin_new, thetamax_new)

    # Hyper-parameters for nn:
    input_size = ocp_dim
    hidden_size = ocp_dim * 100
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

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(10, ocp_dim))
    l_bounds = [q_min-(q_max-q_min)/100, q_min-(q_max-q_min)/100, v_min-(v_max-v_min)/100, v_min-(v_max-v_min)/100]
    u_bounds = [q_max+(q_max-q_min)/100, q_max+(q_max-q_min)/100, v_max+(v_max-v_min)/100, v_max+(v_max-v_min)/100]
    Xu_iter = qmc.scale(sample, l_bounds, u_bounds).tolist()

    # # Generate the initial set of labeled samples:
    # n = pow(19, ocp_dim)
    # X_iter = np.empty((n, ocp_dim))
    # y_iter = np.full((n, 2), [1, 0])
    # r = np.random.random(size=(n, ocp_dim - 1))
    # k = np.random.randint(ocp_dim, size=(n, 1))
    # j = np.random.randint(2, size=(n, 1))
    # x = np.zeros((n, ocp_dim))
    # for i in np.arange(n):
    #     x[i, np.arange(ocp_dim)[np.arange(ocp_dim) != k[i]]] = r[i, :]
    #     x[i, k[i]] = j[i]

    # X_iter[:, 0] = np.float32(x[:, 0] * (q_max + 0.1 - (q_min - 0.1)) + q_min - 0.1)
    # X_iter[:, 1] = np.float32(x[:, 1] * (q_max + 0.1 - (q_min - 0.1)) + q_min - 0.1)
    # X_iter[:, 2] = np.float32(x[:, 2] * (v_max + 1. - (v_min - 1.) + v_min - 1.)
    # X_iter[:, 3] = np.float32(x[:, 3] * (v_max + 1. - (v_min - 1.)) + v_min - 1.)

    # X_iter = X_iter.tolist()
    # y_iter = y_iter.tolist()

    # # Generate the initial set of labeled samples:
    # X_iter = [[(q_max + q_min) / 2, (q_max + q_min) / 2, 0.0, 0.0]]
    # res = ocp.compute_problem([(q_max + q_min) / 2, (q_max + q_min) / 2], [0.0, 0.0])
    # if res == 1:
    #     y_iter = [[0, 1]]
    # else:
    #     y_iter = [[1, 0]]

    Xu_iter_tensor = torch.Tensor(Xu_iter).to(device)
    mean, std = torch.mean(Xu_iter_tensor).to(device), torch.std(Xu_iter_tensor).to(device)

    def testing(s0):
        q0,v0 = s0[:2], s0[2:]

        # Data testing:
        res = ocp.compute_problem(q0, v0)

        if res == 1:
            return [q0[0], q0[1], v0[0], v0[1], 0, 1]
        elif res == 0:
            return [q0[0], q0[1], v0[0], v0[1], 1, 0]

    with Pool(cpu_num) as p:
        X_iter = list(p.map(testing, Xu_iter[:N_init]))

    # Delete tested data from the unlabeled set:
    del Xu_iter[:N_init]

    print("INITIAL CLASSIFIER IN TRAINING")

    it = 0
    val = 1

    # Train the model
    while val > loss_stop and it <= it_max:

        ind = random.sample(range(len(X_iter)), n_minibatch)

        X_iter_tensor = torch.Tensor([X_iter[i][:4] for i in ind]).to(device)
        y_iter_tensor = torch.Tensor([X_iter[i][4:] for i in ind]).to(device)
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
    h = 0.02
    x_min, x_max = q_min-(q_max-q_min)/100, q_max+(q_max-q_min)/100
    y_min, y_max = v_min-(v_max-v_min)/100, v_max+(v_max-v_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max+h, h), np.arange(y_min, y_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()

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
    #     ).to(device)
    #     inp = (inp - mean) / std
    #     out = model(inp)
    #     y_pred = np.argmax(out.numpy(), axis=1)
    #     Z = y_pred.reshape(xx.shape)
    #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #     xit = []
    #     yit = []
    #     cit = []
    #     for i in range(len(X_iter)):
    #         if (
    #             norm(X_iter[i][0] - (q_min + q_max) / 2) < 0.1
    #             and norm(X_iter[i][2]) < 0.1
    #         ):
    #             xit.append(X_iter[i][1])
    #             yit.append(X_iter[i][3])
    #             if X_iter[i][4] == 1:
    #                 cit.append(0)
    #             else:
    #                 cit.append(1)
    #     plt.scatter(
    #         xit,
    #         yit,
    #         c=cit,
    #         marker=".",
    #         alpha=0.5,
    #         cmap=plt.cm.Paired,
    #     )
    #     plt.xlim([x_min, x_max])
    #     plt.ylim([y_min, y_max])
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
    #     ).to(device)
    #     inp = (inp - mean) / std
    #     out = model(inp)
    #     y_pred = np.argmax(out.numpy(), axis=1)
    #     Z = y_pred.reshape(xx.shape)
    #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #     xit = []
    #     yit = []
    #     cit = []
    #     for i in range(len(X_iter)):
    #         if (
    #             norm(X_iter[i][1] - (q_min + q_max) / 2) < 0.1
    #             and norm(X_iter[i][3]) < 0.1
    #         ):
    #             xit.append(X_iter[i][0])
    #             yit.append(X_iter[i][2])
    #             if X_iter[i][4] == 1:
    #                 cit.append(0)
    #             else:
    #                 cit.append(1)
    #     plt.scatter(
    #         xit,
    #         yit,
    #         c=cit,
    #         marker=".",
    #         alpha=0.5,
    #         cmap=plt.cm.Paired,
    #     )
    #     plt.xlim([x_min, x_max])
    #     plt.ylim([y_min, y_max])
    #     plt.grid()
    #     plt.title("First actuator")

    # Active learning:
    k = 0  # iteration number
    etpmax = 1
    # performance_history = []

    sigmoid = nn.Sigmoid()

    while not (etpmax < etp_stop or len(Xu_iter) == 0):

        if len(Xu_iter) < B:
            B = len(Xu_iter)

        with torch.no_grad():
            Xu_iter_tensor = torch.Tensor(Xu_iter).to(device)
            Xu_iter_tensor = (Xu_iter_tensor - mean) / std
            prob_xu = sigmoid(model(Xu_iter_tensor)).cpu()
            etp = entropy(prob_xu, axis=1)

        maxindex = np.argpartition(etp, -B)[
            -B:
        ].tolist()  # indexes of the uncertain samples
        maxindex.sort(reverse=True)

        etpmax = max(etp[maxindex])  # max entropy used for the stopping condition
        # performance_history.append(etpmax)

        k += 1
        #count = 0

        # Take and delete data to test from the unlabeled set:
        elems = [None] * B
        for i in range(B):
            elems[i] = Xu_iter[maxindex[i]]
            del Xu_iter[maxindex[i]]

        with Pool(cpu_num) as p:
            X_iter = X_iter + list(p.map(testing, elems))

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

            X_iter_tensor = torch.Tensor([X_iter[i][:4] for i in ind]).to(device)
            y_iter_tensor = torch.Tensor([X_iter[i][4:] for i in ind]).to(device)
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
                    (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                    xrav,
                    np.zeros(yrav.shape[0]),
                    yrav,
                ]
            )
        ).to(device)
        inp = (inp - mean) / std
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
                if X_iter[i][4] == 1:
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
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
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
                if X_iter[i][4] == 1:
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
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.grid()
        plt.title("First actuator")

    print("Execution time: %s seconds" % (time.time() - start_time))

# stats = Stats(pr)
# stats.sort_stats('cumtime').print_stats(20)

plt.show()
