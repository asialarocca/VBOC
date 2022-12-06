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

    delta_t = 0.01

    def func_min_val(x):
        thetamin_new, q_dotdot_max = x
        q_dot = - delta_t * q_dotdot_max
        M = ocp.d**2 * ocp.m
        h = ocp.b * q_dot - ocp.d * math.sin(thetamin_new) * ocp.m * ocp.g
        return (q_dotdot_max - (ocp.Fmax - h)/M,
                - thetamin_new + q_min + 1/2 * delta_t**2 * q_dotdot_max)

    def func_max_val(x):
        thetamax_new, q_dotdot_min = x
        q_dot = - delta_t * q_dotdot_min
        M = ocp.d**2 * ocp.m
        h = ocp.b * q_dot - ocp.d * math.sin(thetamax_new) * ocp.m * ocp.g
        return (q_dotdot_min - (-ocp.Fmax - h)/M,
                - thetamax_new + q_max + 1/2 * delta_t**2 * q_dotdot_min)

    guess_min = (q_min, ocp.Fmax/ocp.d**2 * ocp.m)
    guess_max = (q_max, -ocp.Fmax/ocp.d**2 * ocp.m)

    thetamin_new, q_dotdot_max = fsolve(func_min_val, guess_min)
    thetamax_new, q_dotdot_min = fsolve(func_max_val, guess_max)

    # thetamin_new = q_min - delta__t ** 2 * \
    #     (ocp.g * ocp.d * ocp.m + ocp.Fmax) / (-ocp.d ** 2 * ocp.m + ocp.b * delta__t) / 2
    # thetamax_new = q_max - delta__t ** 2 * \
    #     (ocp.g * ocp.d * ocp.m - ocp.Fmax) / (-ocp.d ** 2 * ocp.m + ocp.b * delta__t) / 2

    # print(q_min, thetamin_new, q_max, thetamax_new)

    ocp.set_bounds(thetamin_new, thetamax_new)

    # q_max = thetamax_new
    # q_min = thetamin_new

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
    n_minibatch = pow(10, ocp_dim)
    it_max = 1e3 * B / n_minibatch

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(50, ocp_dim))
    l_bounds = [q_min, v_min]
    u_bounds = [q_max, v_max]
    data = qmc.scale(sample, l_bounds, u_bounds).tolist()

    Xu_iter = data  # Unlabeled set
    Xu_iter_tensor = torch.Tensor(Xu_iter)
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

    X_iter[:, 0] = x[:, 0] * (q_max + (q_max-q_min)/10 -
                              (q_min - (q_max-q_min)/10)) + q_min - (q_max-q_min)/10
    X_iter[:, 1] = x[:, 1] * (v_max + (v_max-v_min)/10 -
                              (v_min - (v_max-v_min)/10)) + v_min - (v_max-v_min)/10
                              
    X_iter = X_iter.tolist()
    y_iter = y_iter.tolist()

    # # Generate the initial set of labeled samples:
    # res = ocp.compute_problem((q_max + q_min) / 2, 0.0)
    # if res != 2:
    #     X_iter = np.double([[(q_max + q_min) / 2, 0.0]])
    #     if res == 1:
    #         y_iter = [[0, 1]]
    #     else:
    #         y_iter = [[1, 0]]
    # else:
    #     raise Exception("Max iteration reached"

    # Training of an initial classifier:
    for n in range(N_init):
        q0 = Xu_iter[n][0]
        v0 = Xu_iter[n][1]

        # Data testing:
        res = ocp.compute_problem(q0, v0)
        if res != 2:
            X_iter.append([q0, v0])
            if res == 1:
                y_iter.append([0, 1])
            else:
                y_iter.append([1, 0])

        # Add intermediate states of succesfull initial conditions:
        if res == 1:
            for f in range(1, ocp.N, int(ocp.N / 3)):
                current_val = ocp.ocp_solver.get(f, "x").tolist()
                X_iter.append(current_val)
                y_iter.append([0, 1])

    # Delete tested data from the unlabeled set:
    del Xu_iter[:N_init]

    it = 0
    val = 1
    
    init_time = time.time()

    # Train the model
    while val > loss_stop and it <= it_max:

        #Xn = np.array([i for i in range(len(X_iter)) if y_iter[i] == [1, 0]])
        #Xp = np.array([i for i in range(len(X_iter)) if y_iter[i] == [0, 1]])
        #ind = Xp[random.sample(range(Xp.shape[0]), int(n_minibatch/4))].tolist()
        #ind.extend(Xn[random.sample(range(Xn.shape[0]), int(3*n_minibatch/4))].tolist())
        
        ind = random.sample(range(len(X_iter)), n_minibatch)

        X_iter_tensor = torch.Tensor([X_iter[i] for i in ind])
        y_iter_tensor = torch.Tensor([y_iter[i] for i in ind])
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
        
    training_times = [time.time() - init_time]

    print("INITIAL CLASSIFIER TRAINED")

    with torch.no_grad():
        # Plot the results:
        plt.figure(figsize=(6, 5))
        h = 0.002
        x_min, x_max = q_min-h, q_max+h
        y_min, y_max = v_min, v_max
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        inp = torch.from_numpy(np.float32(np.c_[xx.ravel(), yy.ravel()]))
        inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        z = [
            0 if y_iter[x] == [1, 0] else 1
            for x in range(len(y_iter))
        ]
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.contour(xx, yy, y_pred.reshape(xx.shape), levels=[0], linewidths=(2,), colors=("k",))
        scatter = plt.scatter(
            [item[0] for item in X_iter], [item[1] for item in X_iter], c=z, marker=".", alpha=0.5, cmap=plt.cm.Paired
        )
        plt.xlim([0.0, np.pi / 2])
        plt.ylim([-10.0, 10.0])
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Classifier")
        hand = scatter.legend_elements()[0]
        plt.legend(handles=hand, labels=("Non viable", "Viable"), loc='upper right')
        plt.grid(True)

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
            Xu_iter_tensor = (Xu_iter_tensor - mean) / std
            prob_xu = sigmoid(model(Xu_iter_tensor))
            etp = entropy(prob_xu, axis=1)

        maxindex = np.argpartition(etp, -B)[
            -B:
        ].tolist()  # indexes of the uncertain samples
        maxindex.sort(reverse=True)

        etpmax = max(etp)  # max entropy used for the stopping condition
        performance_history.append(etpmax)

        k += 1

        # Add the B most uncertain samples to the labeled set:
        for x in range(B):
            x0 = Xu_iter.pop(maxindex[x])
            q0 = x0[0]
            v0 = x0[1]

            # Data testing:
            res = ocp.compute_problem(q0, v0)
            if res != 2:
                X_iter.append([q0, v0])
                if res == 1:
                    y_iter.append([0, 1])
                else:
                    y_iter.append([1, 0])

        it = 0
        val = 1
        
        init_time = time.time()

        # Train the model
        while val > loss_stop and it <= it_max:

            #Xn = np.array([i for i in range(len(X_iter) - B)
            #              if y_iter[i] == [1, 0]])
            #Xp = np.array([i for i in range(len(X_iter) - B)
            #              if y_iter[i] == [0, 1]])
                          
            #if int(3*(n_minibatch / 2)/4) >= Xn.shape[0]:
            #    ind = Xn
            #else:
            #    ind = Xn[random.sample(range(Xn.shape[0]), int(3*(n_minibatch / 2)/4))].tolist()
            #if int((n_minibatch / 2)/4) >= Xp.shape[0]:
            #    ind.extend(Xp)
            #else:
            #    ind.extend(Xp[random.sample(range(Xp.shape[0]), int((n_minibatch / 2)/4))].tolist())
                
            #Xn = np.array([i for i in range(len(X_iter) - B, len(X_iter))
            #              if y_iter[i] == [1, 0]])
            #Xp = np.array([i for i in range(len(X_iter) - B, len(X_iter))
            #              if y_iter[i] == [0, 1]])
                          
            #if int(3*(n_minibatch / 2)/4) >= Xn.shape[0]:
            #    ind.extend(Xn)
            #else:
            #    ind.extend(Xn[random.sample(range(Xn.shape[0]),
            #               int(3*(n_minibatch / 2)/4))].tolist())
            #if int((n_minibatch / 2)/4) >= Xp.shape[0]:
            #    ind.extend(Xp)
            #else:
            #    ind.extend(Xp[random.sample(range(Xp.shape[0]), int((n_minibatch / 2)/4))].tolist())
            
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

            # Forward pass
            outputs = model(X_iter_tensor)
            loss = criterion(outputs, y_iter_tensor)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            val = beta * val + (1 - beta) * loss.item()

            it += 1
            
        training_times.append(time.time() - init_time)

        print("CLASSIFIER", k, "TRAINED")

        with torch.no_grad():
            # Plot the results:
            plt.figure(figsize=(6, 5))
            
            inp = torch.from_numpy(np.float32(np.c_[xx.ravel(), yy.ravel()]))
            inp = (inp - mean) / std
            out = model(inp)
            y_pred = np.argmax(out.numpy(), axis=1)
            Z = y_pred.reshape(xx.shape)
            z = [
                0 if y_iter[x] == [1, 0] else 1
                for x in range(len(y_iter))
            ]
            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
            plt.contour(xx, yy, y_pred.reshape(xx.shape), levels=[0], linewidths=(2,), colors=("k",))
            scatter = plt.scatter(
                [item[0] for item in X_iter], [item[1] for item in X_iter], c=z, marker=".", alpha=0.5, cmap=plt.cm.Paired
            )
            plt.xlim([0.0, np.pi / 2])
            plt.ylim([-10.0, 10.0])
            plt.xlabel("Initial position [rad]")
            plt.ylabel("Initial velocity [rad/s]")
            plt.title("Classifier")
            hand = scatter.legend_elements()[0]
            plt.legend(handles=hand, labels=("Non viable", "Viable"), loc='upper right')
            plt.grid(True)

    #plt.figure()
    #plt.plot(performance_history)
    #plt.scatter(range(len(performance_history)), performance_history)
    #plt.xlabel("Iteration number")
    #plt.ylabel("Maximum entropy")
    #plt.title("Maximum entropy evolution")
    
    #plt.figure(figsize=(6, 5))
    #plt.bar(range(k+1),training_times)
    #plt.xlabel("Iteration")
    #plt.ylabel("Time")
    #plt.title("Training times")
    
    with torch.no_grad():
        inp = torch.Tensor(X_iter)
        inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1).tolist()
        sum_correct = 0
        z = [
                0 if y_iter[x] == [1, 0] else 1
                for x in range(len(y_iter))
            ]
        for i in range(len(X_iter)):
            if y_pred[i] == z[i]:
                sum_correct = sum_correct + 1
        accuracy = sum_correct/len(X_iter)

    print('The classifier accuracy is:', accuracy)
  
    print("Execution time: %s seconds" % (time.time() - start_time))

#pr.print_stats(sort="cumtime")

plt.show()
