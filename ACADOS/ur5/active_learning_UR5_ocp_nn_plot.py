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

warnings.filterwarnings("ignore")


with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPdoublependulumINIT()

    ocp_dim = ocp.nx

    model = torch.load('model_save_2')

    X_iter = np.load('X_iter.npy').tolist()
    y_iter = np.load('y_iter.npy').tolist()

    gridp = 10
    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(gridp, ocp_dim))
    l_bounds = ocp.xmin.tolist()
    u_bounds = ocp.xmax.tolist()
    Xu_iter = qmc.scale(sample, l_bounds, u_bounds).tolist()
    Xu_iter_tensor = torch.Tensor(Xu_iter)
    mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

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
                    -0.80631 * np.ones(xrav2.shape[0]),
                    xrav2,
                    0.03253 * np.ones(yrav2.shape[0]),
                    yrav2,
                ]
            )
        )
        inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx2.shape)
        plt.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.8)
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

    print("Execution time: %s seconds" % (time.time() - start_time))

pr.print_stats(sort="cumtime")

plt.show()
