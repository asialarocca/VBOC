import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
import time
from triple_pendulum_ocp_class import OCPtriplependulumINIT
import random
import math
import warnings
import torch
import torch.nn as nn
from my_nn import NeuralNet
from statistics import mean

warnings.filterwarnings("ignore")


with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPtriplependulumINIT()

    ocp_dim = ocp.nx

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = -ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    #with open("mean.txt", "r") as f:
    #    val = float(f.readlines()[0])
    #mean = torch.Tensor([val])
    #with open("std.txt", "r") as f:
    #    val = float(f.readlines()[0])
    #std = torch.Tensor([val])
    model = torch.load('model_save')
    X_iter = np.load('X_iter.npy').tolist()
    y_iter = np.load('y_iter.npy').tolist()
    
    gridp = 10
    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(gridp, ocp_dim))
    l_bounds = [q_min, q_min, q_min, v_min, v_min, v_min]
    u_bounds = [q_max, q_max, q_max, v_max, v_max, v_max]
    data = qmc.scale(sample, l_bounds, u_bounds).tolist()
    Xu_iter_tensor = torch.Tensor(data)
    mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

    # Plots:
    h = 0.02
    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()

    with torch.no_grad():
        # Plot the results:
        plt.figure()
        inp = torch.from_numpy(
            np.float32(
                np.c_[(q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                      (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                      xrav,
                      np.zeros(yrav.shape[0]),
                      np.zeros(yrav.shape[0]),
                      yrav,
                      ]
            )
        )
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
                and norm(X_iter[i][1] - (q_min + q_max) / 2) < 0.1
                and norm(X_iter[i][3]) < 0.1 and norm(X_iter[i][4]) < 0.1
            ):
                xit.append(X_iter[i][2])
                yit.append(X_iter[i][5])
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
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.grid()
        plt.title("Third actuator")

        plt.figure()
        inp = torch.from_numpy(
            np.float32(
                np.c_[(q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                      xrav,
                      (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                      np.zeros(yrav.shape[0]),
                      yrav,
                      np.zeros(yrav.shape[0]),
                      ]
            )
        )
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
                and norm(X_iter[i][2] - (q_min + q_max) / 2) < 0.1
                and norm(X_iter[i][3]) < 0.1 and norm(X_iter[i][5]) < 0.1
            ):
                xit.append(X_iter[i][1])
                yit.append(X_iter[i][4])
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
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.grid()
        plt.title("Second actuator")

        plt.figure()
        inp = torch.from_numpy(
            np.float32(
                np.c_[xrav,
                      (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                      (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                      yrav,
                      np.zeros(yrav.shape[0]),
                      np.zeros(yrav.shape[0]),
                      ]
            )
        )
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
                and norm(X_iter[i][2] - (q_min + q_max) / 2) < 0.1
                and norm(X_iter[i][4]) < 0.1 and norm(X_iter[i][5]) < 0.1
            ):
                xit.append(X_iter[i][0])
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
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.grid()
        plt.title("First actuator")

    print("Execution time: %s seconds" % (time.time() - start_time))

pr.print_stats(sort="cumtime")

plt.show()
