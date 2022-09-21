import time
from double_pendulum_ocp_class import OCPdoublependulumNN, OCPdoublependulumINIT
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
from numpy.linalg import norm as norm

X_iter = np.load('X_iter_5.npy').tolist()
y_iter = np.load('y_iter_5.npy').tolist()

model = torch.load('model_save_5')
with open("mean_5.txt", "r") as f:
    mean = float(f.readlines()[0])
with open("std_5.txt", "r") as f:
    std = float(f.readlines()[0])

mpc = OCPdoublependulumINIT()

# Plots:
h = 0.01
xx2, yy2 = np.meshgrid(np.arange(mpc.xmin[1], mpc.xmax[1], h),
                       np.arange(mpc.xmin[3], mpc.xmax[3], h))
xrav2 = xx2.ravel()
yrav2 = yy2.ravel()

x_guess = np.empty((mpc.N+1, 4))

points = [[-0.82723211, -2.51791589, -2.29683664, 2.51291526],
          [-0.85279,  -2.49020793, -2.80994608,  3.03283546],
          [-0.88256443, -2.45935946, -3.13999782,  3.13999985],
          [-0.91397452, -2.42796004, -3.13987576,  3.13999971],
          [-0.94538445, -2.39656035, -3.13999003,  3.13999977],
          [-0.97675137, -2.36515995, -3.13139059,  3.13999994],
          [-1.00799479, -2.33375888, -3.11539738,  3.13999994],
          [-1.03877983, -2.30242855, -3.04027365,  3.12517924],
          [-1.06754358, -2.27162241, -2.71329728,  3.03261182],
          [-1.09306217, -2.24184264, -2.39122505,  2.92015648],
          [-1.11538387, -2.21328032, -2.07388791,  2.7893968],
          [-1.13455514, -2.18610992, -1.76109283,  2.64207302],
          [-1.15061967, -2.16048807, -1.45249437,  2.47999805],
          [-1.16361821, -2.13655264, -1.14784945,  2.3051213],
          [-1.17358901, -2.11442158, -0.84691075,  2.11946793],
          [-1.18056812, -2.09419221, -0.5494839,  1.92514002],
          [-1.18458997, -2.07594048, -0.25544279,  1.72430491]]

for k in range(len(points)):
    with torch.no_grad():

        inp = torch.Tensor([points[k]])
        inp = (inp - torch.Tensor([mean])) / torch.Tensor([std])
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        print('y_pred: ', y_pred)

        status = mpc.compute_problem(points[k][:2], points[k][2:])

        print('status: ', status)

        # mpc.ocp_solver.print_statistics()

        for i in range(0, mpc.N+1):
            x_guess[i, :] = mpc.ocp_solver.get(i, "x")

        # Plot the results:
        plt.figure()
        inp = torch.from_numpy(
            np.float32(
                np.c_[
                    points[k][0] * np.ones(xrav2.shape[0]),
                    xrav2,
                    points[k][2] * np.ones(xrav2.shape[0]),
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
        plt.title("Point")


plt.show(block=True)
