import time
from triple_pendulum_ocp_class import OCPtriplependulumINIT
import random
import math
import warnings
import torch.nn as nn
import numpy as np
import torch
import queue
from my_nn import NeuralNet
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Ocp initialization:
ocp = OCPtriplependulumINIT()

nx = ocp.nx
nu = ocp.nu
N = ocp.N
Tf = ocp.Tf

ocp_solver = ocp.ocp_solver

# Position and velocity bounds:
v_max = ocp.dthetamax
v_min = -ocp.dthetamax
q_max = ocp.thetamax
q_min = ocp.thetamin
Cmax = ocp.Cmax

q0 = np.array([(q_max + q_min) / 2, (q_max + q_min) / 2, (q_max + q_min) / 2])
v0 = np.array([0., 0., 0.])

res = ocp.compute_problem(q0, v0)

# self.get simX
simX = np.ndarray((N+1, nx))
simU = np.ndarray((N, nu))

for i in range(N):
    simX[i, :] = ocp_solver.get(i, "x")
    simU[i, :] = ocp_solver.get(i, "u")
simX[N, :] = ocp_solver.get(N, "x")

t = np.linspace(0, Tf, N+1)

plt.figure()
plt.subplot(3, 1, 1)
line, = plt.step(t, np.append([simU[0, 0]], simU[:, 0]))
plt.ylabel('$C1$')
plt.xlabel('$t$')
plt.hlines(Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
plt.hlines(-Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
plt.ylim([-1.2*Cmax, 1.2*Cmax])
plt.title('Controls')
plt.grid()
plt.subplot(3, 1, 2)
line, = plt.step(t, np.append([simU[0, 1]], simU[:, 1]))
plt.ylabel('$C2$')
plt.xlabel('$t$')
plt.hlines(Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
plt.hlines(-Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
plt.ylim([-1.2*Cmax, 1.2*Cmax])
plt.grid()
plt.subplot(3, 1, 3)
line, = plt.step(t, np.append([simU[0, 2]], simU[:, 2]))
plt.ylabel('$C3$')
plt.xlabel('$t$')
plt.hlines(Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
plt.hlines(-Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
plt.ylim([-1.2*Cmax, 1.2*Cmax])
plt.grid()

plt.figure()
plt.subplot(6, 1, 1)
line, = plt.plot(t, simX[:, 0])
plt.ylabel('$theta1$')
plt.xlabel('$t$')
plt.title('States')
plt.grid()
plt.subplot(6, 1, 2)
line, = plt.plot(t, simX[:, 1])
plt.ylabel('$theta2$')
plt.xlabel('$t$')
plt.grid()
plt.subplot(6, 1, 3)
line, = plt.plot(t, simX[:, 2])
plt.ylabel('$theta2$')
plt.xlabel('$t$')
plt.grid()
plt.subplot(6, 1, 4)
line, = plt.plot(t, simX[:, 3])
plt.ylabel('$dtheta1$')
plt.xlabel('$t$')
plt.grid()
plt.subplot(6, 1, 5)
line, = plt.plot(t, simX[:, 4])
plt.ylabel('$dtheta3$')
plt.xlabel('$t$')
plt.grid()
plt.subplot(6, 1, 6)
line, = plt.plot(t, simX[:, 5])
plt.ylabel('$dtheta3$')
plt.xlabel('$t$')
plt.grid()
plt.show()
