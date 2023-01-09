import cProfile
import numpy as np
import random 
import matplotlib.pyplot as plt
from sklearn import svm
import math
from numpy.linalg import norm as norm
import time
from double_pendulum_ocp_class_reverse import OCPdoublependulumRINIT
import warnings
warnings.filterwarnings("ignore")

with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPdoublependulumRINIT()

    ocp_dim = ocp.nx

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = - ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    # Initialization of the SVM classifier:
    clf = svm.SVC(C=1e3, kernel='rbf')

    eps = 1.
    multip = 0.

    # lb = np.array([ocp.thetamin, ocp.thetamin, 0, -ocp.dthetamax])
    # ub = np.array([ocp.thetamax, ocp.thetamax, ocp.dthetamax, ocp.dthetamax])

    # for i in range(ocp.N):
    #     ocp.ocp_solver.constraints_set(i, "lbx", lb)
    #     ocp.ocp_solver.constraints_set(i, "ubx", ub)

    ocp.ocp_solver.reset()
    
    q2_fin = q_min + random.random() * (q_max-q_min)
    
    xe = np.array([q_max, q2_fin, 0., 0.])

    ocp.ocp_solver.constraints_set(ocp.N, "lbx", xe)
    ocp.ocp_solver.constraints_set(ocp.N, "ubx", xe)

    if q2_fin > (ocp.thetamax + ocp.thetamin)/2:
        q2_guess = ocp.thetamax
        q2dot_guess = ocp.dthetamax
    else:
        q2_guess = ocp.thetamin
        q2dot_guess = -ocp.dthetamax

    x_guess = np.array([ocp.thetamin, q2_guess, ocp.dthetamax, q2dot_guess])

    # x_guess = np.array([ocp.thetamin, q2_fin, ocp.dthetamax, 0.])

    ran1 = random.random() * multip
    ran2 = random.random() * multip
    normal = math.sqrt(ran1 **2 + ran2 **2 + 1)

    W = np.diag([0., -ran1, -1., -ran2, 0., 0.]) 

    for i in range(ocp.N):
        ocp.ocp_solver.set(i, "x", x_guess)
        ocp.ocp_solver.cost_set(i, "W", W)

    ocp.ocp_solver.set(ocp.N, "x", xe)

    status = ocp.ocp_solver.solve()

    if status == 0:
        # for f in range(0, ocp.N+1):
        #     current_val = ocp.ocp_solver.get(f, "x")
        #     X_save = np.append(X_save, [np.append(current_val, 1)], axis=0)
        #     current_val[1] = current_val[1] + eps * math.sqrt(ran1)/normal
        #     current_val[2] = current_val[2] + eps * 1/normal
        #     current_val[3] = current_val[3] + eps * math.sqrt(ran2)/normal
        #     X_save = np.append(X_save, [np.append(current_val, 0)], axis=0)

        # get solution
        simX = np.ndarray((ocp.N+1, ocp.nx))
        simU = np.ndarray((ocp.N, ocp.nu))

        for i in range(ocp.N):
            simX[i, :] = ocp.ocp_solver.get(i, "x")
            simU[i, :] = ocp.ocp_solver.get(i, "u")
        simX[ocp.N, :] = ocp.ocp_solver.get(ocp.N, "x")

        ocp.ocp_solver.print_statistics()

        t = np.linspace(0, ocp.Tf, ocp.N+1)

        plt.figure()
        plt.subplot(2, 1, 1)
        line, = plt.step(t, np.append([simU[0, 0]], simU[:, 0]))
        plt.ylabel('$C1$')
        plt.xlabel('$t$')
        plt.hlines(ocp.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
        plt.hlines(-ocp.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
        plt.ylim([-1.2*ocp.Cmax, 1.2*ocp.Cmax])
        plt.title('Controls')
        plt.grid()
        plt.subplot(2, 1, 2)
        line, = plt.step(t, np.append([simU[0, 1]], simU[:, 1]))
        plt.ylabel('$C2$')
        plt.xlabel('$t$')
        plt.hlines(ocp.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
        plt.hlines(-ocp.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
        plt.ylim([-1.2*ocp.Cmax, 1.2*ocp.Cmax])
        plt.grid()

        plt.figure()
        plt.subplot(4, 1, 1)
        line, = plt.plot(t, simX[:, 0])
        plt.ylabel('$theta1$')
        plt.xlabel('$t$')
        plt.title('States')
        plt.grid()
        plt.subplot(4, 1, 2)
        line, = plt.plot(t, simX[:, 1])
        plt.ylabel('$theta2$')
        plt.xlabel('$t$')
        plt.grid()
        plt.subplot(4, 1, 3)
        line, = plt.plot(t, simX[:, 2])
        plt.ylabel('$dtheta1$')
        plt.xlabel('$t$')
        plt.grid()
        plt.subplot(4, 1, 4)
        line, = plt.plot(t, simX[:, 3])
        plt.ylabel('$dtheta2$')
        plt.xlabel('$t$')
        plt.grid()

        for i in range(ocp.N):

            ocp.ocp_solver.reset()

            ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([ocp.thetamin, ocp.thetamin, 0., 0.]))
            ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([ocp.thetamax, ocp.thetamax, 0., 0.]))

            x0 = simX[i, :]
            x0[1] = x0[1] + eps * math.sqrt(ran1)/normal
            x0[2] = x0[2] + eps * 1/normal
            x0[3] = x0[3] + eps * math.sqrt(ran2)/normal

            print(x0)

            ocp.ocp_solver.constraints_set(0, "lbx", x0)
            ocp.ocp_solver.constraints_set(0, "ubx", x0)

            x_guess = np.array([x0[0], x0[1], 0.0, 0.0])

            W = np.diag([0., 0., 1., 1., 0., 0.]) 

            for i in range(ocp.N):
                ocp.ocp_solver.set(i, "x", x_guess)
                ocp.ocp_solver.cost_set(i, "W", W)

            ocp.ocp_solver.set(ocp.N, "x", x_guess)
            ocp.ocp_solver.cost_set(ocp.N, "W", 2 * np.diag([0., 0., 1., 1.]))

            status = ocp.ocp_solver.solve()

            print(status)

            if status != 0:
                ocp.ocp_solver.print_statistics()

    else:
        ocp.ocp_solver.print_statistics()
        raise Exception(f'acados returned status {status}.')

    print("Execution time: %s seconds" % (time.time() - start_time))

# pr.print_stats(sort='cumtime')

plt.show()
