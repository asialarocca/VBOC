import cProfile
from pstats import Stats
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
from multiprocessing import Pool
from time import sleep

warnings.filterwarnings("ignore")

with cProfile.Profile() as pr:

    start_time = time.time()

    cpu_num = 31

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
    hidden_size = ocp_dim * 100
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
    etp_stop = 0.2  # active learning stopping condition
    loss_stop = 0.01  # nn training stopping condition
    beta = 0.8
    n_minibatch = 64
    it_max = 1e2 * B / n_minibatch

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(50, ocp_dim))
    l_bounds = [q_min-(q_max-q_min)/100, v_min-(v_max-v_min)/100]
    u_bounds = [q_max+(q_max-q_min)/100, v_max+(v_max-v_min)/100]
    data = qmc.scale(sample, l_bounds, u_bounds)

    Xu_iter = data.tolist()  # Unlabeled set
    Xu_iter_tensor = torch.Tensor(Xu_iter)
    mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

    def testing(s0):
        q0,v0 = s0[0], s0[1]

        # Data testing:
        res = ocp.compute_problem(q0, v0)

        if res == 1:
            return [q0, v0, 0, 1]
        elif res == 0:
            return [q0, v0, 1, 0]

    with Pool(cpu_num) as p:
        X_iter = list(p.map(testing, Xu_iter[:N_init]))

    # Delete tested data from the unlabeled set:
    del Xu_iter[:N_init]

    it = 0
    val = 1

    # Train the model
    while val > loss_stop and it <= it_max:

        ind = random.sample(range(len(X_iter)), n_minibatch)

        X_iter_tensor = torch.Tensor([X_iter[i][:2] for i in ind])
        y_iter_tensor = torch.Tensor([X_iter[i][2:] for i in ind])
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

    # with torch.no_grad():
    #     # Plot the results:
    #     plt.figure()
    #     h = 0.01
    #     x_min, x_max = q_min-(q_max-q_min)/100, q_max+(q_max-q_min)/100
    #     y_min, y_max = v_min-(v_max-v_min)/100, v_max+(v_max-v_min)/100
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #     inp = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    #     inp = (inp - mean) / std
    #     out = model(inp)
    #     y_pred = np.argmax(out.numpy(), axis=1)
    #     Z = y_pred.reshape(xx.shape)
    #     z = [
    #         0 if X_iter[x][2] == 1 else 1
    #         for x in range(len(X_iter))
    #     ]
    #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #     scatter = plt.scatter(
    #         [X_iter[n][0] for n in range(len(X_iter))], [X_iter[n][1] for n in range(len(X_iter))], c=z, marker=".", alpha=0.5, cmap=plt.cm.Paired
    #     )
    #     plt.xlim([x_min, x_max])
    #     plt.ylim([y_min, y_max])
    #     plt.xlabel("Initial position [rad]")
    #     plt.ylabel("Initial velocity [rad/s]")
    #     plt.title("Classifier")
    #     hand = scatter.legend_elements()[0]
    #     plt.legend(handles=hand, labels=("Non viable", "Viable"))
    #     plt.grid(True)

    # Active learning:
    k = 0  # iteration number
    etpmax = 1

    # performance_history = []

    while not (etpmax < etp_stop or len(Xu_iter) == 0):
        if len(Xu_iter) < B:
            B = len(Xu_iter)

        with torch.no_grad():
            sigmoid = nn.Sigmoid()
            Xu_iter_tensor = torch.Tensor(Xu_iter)
            Xu_iter_tensor = (Xu_iter_tensor - mean) / std
            prob_xu = sigmoid(model(Xu_iter_tensor))
            etp = entropy(prob_xu, axis=1)

        maxindex = np.argpartition(etp, -B)[-B:].tolist()  # indexes of the uncertain samples
        maxindex.sort(reverse=True)

        etpmax = max(etp)  # max entropy used for the stopping condition
        # performance_history.append(etpmax)

        k += 1

        # Take and delete data to test from the unlabeled set:
        elems = [None] * B
        for i in range(B):
            elems[i] = Xu_iter[maxindex[i]]
            del Xu_iter[maxindex[i]]

        with Pool(cpu_num) as p:
            X_iter = X_iter + list(p.map(testing, elems))

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

            X_iter_tensor = torch.Tensor([X_iter[i][:2] for i in ind])
            y_iter_tensor = torch.Tensor([X_iter[i][2:] for i in ind])
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

        # with torch.no_grad():
        #     # Plot the results:
        #     plt.figure()
        #     h = 0.01
        #     x_min, x_max = q_min-(q_max-q_min)/100, q_max+(q_max-q_min)/100
        #     y_min, y_max = v_min-(v_max-v_min)/100, v_max+(v_max-v_min)/100
        #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        #     inp = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
        #     inp = (inp - mean) / std
        #     out = model(inp)
        #     y_pred = np.argmax(out.numpy(), axis=1)
        #     Z = y_pred.reshape(xx.shape)
        #     z = [
        #         0 if X_iter[x][2] == 1 else 1
        #         for x in range(len(X_iter))
        #     ]
        #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        #     scatter = plt.scatter(
        #         [X_iter[n][0] for n in range(len(X_iter))], [X_iter[n][1] for n in range(len(X_iter))], c=z, marker=".", alpha=0.5, cmap=plt.cm.Paired
        #     )
        #     plt.xlim([x_min, x_max])
        #     plt.ylim([y_min, y_max])
        #     plt.xlabel("Initial position [rad]")
        #     plt.ylabel("Initial velocity [rad/s]")
        #     plt.title("Classifier")
        #     hand = scatter.legend_elements()[0]
        #     plt.legend(handles=hand, labels=("Non viable", "Viable"))
        #     plt.grid(True)

    # plt.figure()
    # plt.plot(performance_history)
    # plt.scatter(range(len(performance_history)), performance_history)
    # plt.xlabel("Iteration number")
    # plt.ylabel("Maximum entropy")
    # plt.title("Maximum entropy evolution")

    print("Execution time: %s seconds" % (time.time() - start_time))

stats = Stats(pr)
stats.sort_stats('cumtime').print_stats(20)

# plt.show()
