import time
from double_pendulum_ocp_class_mpc_reduced import OCPdoublependulumNN, OCPdoublependulumINIT
import numpy as np
import random
import scipy.linalg as lin
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template import AcadosModel
from casadi import SX, vertcat, exp, fmax, tanh, Function
import matplotlib.pyplot as plt
import os
import casadi as cs
import urdf2casadi.urdfparser as u2c
import pinocchio as pin
from scipy.stats import entropy, qmc
import cProfile
import pstats
import io
from pstats import SortKey
import params as conf
import torch
import casadi as cs
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.image as mpimg

model = torch.load('model_save_4')
with open("mean_4.txt", "r") as f:
    mean = float(f.readlines()[0])
with open("std_4.txt", "r") as f:
    std = float(f.readlines()[0])

mpc = OCPdoublependulumNN(model, mean, std)
#mpc = OCPdoublependulumINIT()

iterations = 30

resX = np.array([[-0.7805794638446351, -2.5675506591796875, 0., 0.]])
resU = np.empty((0, 2))

x_guess = np.array([[-0.78057946, -2.56755066,  0.,          0., ],
                    [-0.78062751, -2.56469023, -0.00998005,  0.57213944],
                    [-0.78079842, -2.55610535, -0.02531239,  1.14500612],
                    [-0.78114041, -2.5417873, -0.04495622,  1.71889891],
                    [-0.78176021, -2.52403118, -0.07923391,  1.83254324],
                    [-0.78273124, -2.50528143, -0.11507285,  1.91761949],
                    [-0.78404497, -2.4859226, -0.14756228,  1.95432127],
                    [-0.78565702, -2.46637009, -0.17459829,  1.95631292],
                    [-0.78750815, -2.44687972, -0.19533434,  1.94185648],
                    [-0.78952837, -2.42758542, -0.20841896,  1.91706241],
                    [-0.79163445, -2.40857733, -0.21252997,  1.884574]])
u_guess = np.array([[13.26957224,  19.99999908],
                    [13.2022724,   19.99996529],
                    [13.54906212,  19.99973718],
                    [10.49622068,  -8.7080096],
                    [10.27389041, -10.5645691],
                    [10.20351619, -13.61691246],
                    [10.28869353, -15.79602997],
                    [10.53130723, -16.82052262],
                    [10.92888379, -17.43647414],
                    [11.4764367, -17.85519724]])

# Plots:
h = 0.01
xx2, yy2 = np.meshgrid(np.arange(mpc.xmin[1], mpc.xmax[1]+h, h),
                       np.arange(mpc.xmin[3], mpc.xmax[3], h))
xrav2 = xx2.ravel()
yrav2 = yy2.ravel()

with torch.no_grad():
    # Plot the results:
    plt.figure()
    inp = torch.from_numpy(
        np.float32(
            np.c_[
                x_guess[0, 0] * np.ones(xrav2.shape[0]),
                xrav2,
                x_guess[0, 2] * np.ones(xrav2.shape[0]),
                yrav2,
            ]
        )
    )
    inp = (inp - mean) / std
    out = model(inp)
    y_pred = np.argmax(out.numpy(), axis=1)
    Z = y_pred.reshape(xx2.shape)
    plt.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.plot(
        x_guess[:, 1],
        x_guess[:, 3],
        '-o'
    )
    plt.xlim([mpc.xmin[1], mpc.xmax[1]])
    plt.ylim([mpc.xmin[3], mpc.xmax[3]])
    plt.grid()
    plt.title("Elbow joint")

frames = []

for it in range(1, iterations):
    status = mpc.compute_problem(resX[-1, :2], resX[-1, 2:], x_guess, u_guess)

    if status == 1:
        for i in range(1, mpc.N+1):
            x_guess[i-1, :] = mpc.ocp_solver.get(i, "x")
        x_guess[mpc.N, :] = mpc.ocp.cost.yref_e

        for i in range(1, mpc.N):
            u_guess[i-1, :] = mpc.ocp_solver.get(i, "u")

        with torch.no_grad():
            inp = torch.Tensor(x_guess)
            inp = (inp - torch.Tensor([mean])) / torch.Tensor([std])
            out = model(inp)
            y_pred = np.argmax(out.numpy(), axis=1)
            print(y_pred)

        print(mpc.ocp_solver.get(1, "x"))

        # with torch.no_grad():
        #     # Plot the results:
        #     plt.figure()
        #     inp = torch.from_numpy(
        #         np.float32(
        #             np.c_[
        #                 mpc.ocp_solver.get(1, "x")[0] * np.ones(xrav2.shape[0]),
        #                 xrav2,
        #                 mpc.ocp_solver.get(1, "x")[2] * np.ones(xrav2.shape[0]),
        #                 yrav2,
        #             ]
        #         )
        #     )
        #     inp = (inp - mean) / std
        #     out = model(inp)
        #     y_pred = np.argmax(out.numpy(), axis=1)
        #     Z = y_pred.reshape(xx2.shape)
        #     plt.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        #     plt.plot(
        #         x_guess[:, 1],
        #         x_guess[:, 3],
        #         '-o'
        #     )
        #     plt.xlim([mpc.xmin[1], mpc.xmax[1]])
        #     plt.ylim([mpc.xmin[3], mpc.xmax[3]])
        #     plt.grid()
        #     plt.title("Elbow joint")

        resX = np.append(resX, np.array([mpc.ocp_solver.get(1, "x")]), axis=0)
        resU = np.append(resU, np.array([mpc.ocp_solver.get(0, "u")]), axis=0)

        with torch.no_grad():
            fig = plt.figure()
            # Plot the results:
            inp = torch.from_numpy(
                np.float32(
                    np.c_[
                        mpc.ocp_solver.get(1, "x")[0] * np.ones(xrav2.shape[0]),
                        xrav2,
                        mpc.ocp_solver.get(1, "x")[2] * np.ones(xrav2.shape[0]),
                        yrav2,
                    ]
                )
            )
            inp = (inp - mean) / std
            out = model(inp)
            y_pred = np.argmax(out.numpy(), axis=1)
            Z = y_pred.reshape(xx2.shape)
            plt.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.8)
            plt.plot(
                resX[:, 1],
                resX[:, 3],
                '-o'
            )
            plt.xlim([mpc.xmin[1], mpc.xmax[1]])
            plt.ylim([mpc.xmin[3], mpc.xmax[3]])
            plt.grid()
            plt.title("Elbow joint")

        ite = 0

    else:
        # mpc.ocp_solver.print_statistics()

        ite = ite + 1

        resX = np.append(resX, np.array([x_guess[ite, :]]), axis=0)
        resU = np.append(resU, np.array([u_guess[ite-1, :]]), axis=0)

        if ite == 10:
            break

with torch.no_grad():
    inp = torch.Tensor(resX)
    inp = (inp - torch.Tensor([mean])) / torch.Tensor([std])
    out = model(inp)
    y_pred = np.argmax(out.numpy(), axis=1)
    print(y_pred)

with torch.no_grad():
    # Plot the results:
    plt.figure()
    inp = torch.from_numpy(
        np.float32(
            np.c_[
                -0.7805794638446351 * np.ones(xrav2.shape[0]),
                xrav2,
                np.zeros(xrav2.shape[0]),
                yrav2,
            ]
        )
    )
    inp = (inp - mean) / std
    out = model(inp)
    y_pred = np.argmax(out.numpy(), axis=1)
    Z = y_pred.reshape(xx2.shape)
    plt.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.plot(
        resX[:, 1],
        resX[:, 3],
        '-o'
    )
    plt.xlim([mpc.xmin[1], mpc.xmax[1]])
    plt.ylim([mpc.xmin[3], mpc.xmax[3]])
    plt.grid()
    plt.title("Elbow joint")

print(resX)

plt.show(block=True)
