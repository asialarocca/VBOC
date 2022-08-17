import cProfile
import numpy as np
import matplotlib.pyplot as plt
import time
from double_pendulum_ocp_class import OCPdoublependulumINIT, OCPdoublependulumNN
import warnings

warnings.filterwarnings("ignore")


with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPdoublependulumINIT()

    nx = ocp.nx
    nu = ocp.nu
    N = ocp.N
    ocp_solver = ocp.ocp_solver
    Tf = ocp.Tf

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = -ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin
    Cmax = ocp.Cmax

    res = ocp.compute_problem([(q_max + q_min) / 2, (q_max + q_min) / 2], [0.0, 0.0])

    if res == 1:
        # get solution
        simX = np.ndarray((N + 1, nx))
        simU = np.ndarray((N, nu))

        for i in range(N):
            simX[i, :] = ocp_solver.get(i, "x")
            simU[i, :] = ocp_solver.get(i, "u")
        simX[N, :] = ocp_solver.get(N, "x")

        ocp_solver.print_statistics()

        t = np.linspace(0, Tf, N + 1)

        plt.figure()
        plt.subplot(2, 1, 1)
        (line,) = plt.step(t, np.append([simU[0, 0]], simU[:, 0]))
        plt.ylabel("$C1$")
        plt.xlabel("$t$")
        plt.hlines(Cmax, t[0], t[-1], linestyles="dashed", alpha=0.7)
        plt.hlines(-Cmax, t[0], t[-1], linestyles="dashed", alpha=0.7)
        plt.ylim([-1.2 * Cmax, 1.2 * Cmax])
        plt.title("Controls")
        plt.grid()
        plt.subplot(2, 1, 2)
        (line,) = plt.step(t, np.append([simU[0, 1]], simU[:, 1]))
        plt.ylabel("$C2$")
        plt.xlabel("$t$")
        plt.hlines(Cmax, t[0], t[-1], linestyles="dashed", alpha=0.7)
        plt.hlines(-Cmax, t[0], t[-1], linestyles="dashed", alpha=0.7)
        plt.ylim([-1.2 * Cmax, 1.2 * Cmax])
        plt.grid()

        plt.figure()
        plt.subplot(4, 1, 1)
        (line,) = plt.plot(t, simX[:, 0])
        plt.ylabel("$theta1$")
        plt.xlabel("$t$")
        plt.title("States")
        plt.grid()
        plt.subplot(4, 1, 2)
        (line,) = plt.plot(t, simX[:, 1])
        plt.ylabel("$theta2$")
        plt.xlabel("$t$")
        plt.grid()
        plt.subplot(4, 1, 3)
        (line,) = plt.plot(t, simX[:, 2])
        plt.ylabel("$dtheta1$")
        plt.xlabel("$t$")
        plt.grid()
        plt.subplot(4, 1, 4)
        (line,) = plt.plot(t, simX[:, 3])
        plt.ylabel("$dtheta2$")
        plt.xlabel("$t$")
        plt.grid()

        plt.show()
