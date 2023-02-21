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

    eps = 1.
    multip = 0.1

    ocp.ocp_solver.reset()
        
    q_fin = q_min + random.random() * (q_max-q_min)
    
    xe = np.array([q_max, q_fin, 0., 0.])

    ocp.ocp_solver.constraints_set(ocp.N, "lbx", xe)
    ocp.ocp_solver.constraints_set(ocp.N, "ubx", xe)

    ocp.ocp_solver.constraints_set(0, "lbx", np.array([q_min, q_min, v_min, v_min]))
    ocp.ocp_solver.constraints_set(0, "ubx", np.array([q_min, q_max, v_max, v_max]))

    # if q_fin > (ocp.thetamax + ocp.thetamin)/2:
    #     q2_guess = ocp.thetamax
    #     q2dot_guess = ocp.dthetamax
    # else:
    #     q2_guess = ocp.thetamin
    #     q2dot_guess = -ocp.dthetamax

    # x_guess = np.array([ocp.thetamin, q2_guess, ocp.dthetamax, q2dot_guess])

    ls = np.linspace(q_min, q_max, ocp.N, endpoint=False)
    val = np.full(ocp.N, q_fin)
    x_guess = np.append([ls], [val], axis=0)
    val = np.full(ocp.N, v_max)
    x_guess = np.append(x_guess, [val], axis=0)
    val = np.zeros(ocp.N)
    x_guess = np.append(x_guess, [val], axis=0).T

    # x_guess = np.array([ocp.thetamin, q_fin, ocp.dthetamax, 0.])

    ran = random.random() * multip
    q_ref = random.choice([q_min, q_max])
    normal = math.sqrt(ran**2 + 1)

    W_0 = 2 * np.diag([0., 0., 1., 0., 0., 0.]) 
    W = 2 * np.diag([1., ran, 0., 0., 0., 0.]) 

    y_ref_0 = np.array([0., 0., x_guess[0,2], 0., 0., 0.])
    y_ref = np.array([xe[0], q_ref, 0., 0., 0., 0.])

    # lb = np.array([ocp.thetamin, ocp.thetamin, 0, -ocp.dthetamax])
    # ub = np.array([ocp.thetamax, ocp.thetamax, ocp.dthetamax, ocp.dthetamax])

    ocp.ocp_solver.set(0, "x", x_guess[0])
    ocp.ocp_solver.cost_set(0, "W", W_0)
    ocp.ocp_solver.set(0, "yref", y_ref_0)

    for i in range(1,ocp.N):
        ocp.ocp_solver.set(i, "x", x_guess[i])
        ocp.ocp_solver.cost_set(i, "W", W)
        ocp.ocp_solver.set(i, "yref", y_ref)
        # ocp.ocp_solver.constraints_set(i, "lbx", lb)
        # ocp.ocp_solver.constraints_set(i, "ubx", ub)  

    ocp.ocp_solver.set(ocp.N, "x", xe)
    # ocp.ocp_solver.cost_set(ocp.N, "W", np.diag([0., 0., 0., 0.]))

    status = ocp.ocp_solver.solve()
    ocp.ocp_solver.print_statistics()

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

        # for i in range(ocp.N):

        #     ocp.ocp_solver.reset()

        #     ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([ocp.thetamin, ocp.thetamin, 0., 0.]))
        #     ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([ocp.thetamax, ocp.thetamax, 0., 0.]))

        #     x0 = simX[i, :]
        #     x0[1] = x0[1] + eps * math.sqrt(ran1)/normal
        #     x0[2] = x0[2] + eps * 1/normal
        #     x0[3] = x0[3] + eps * math.sqrt(ran2)/normal

        #     print(x0)

        #     ocp.ocp_solver.constraints_set(0, "lbx", x0)
        #     ocp.ocp_solver.constraints_set(0, "ubx", x0)

        #     x_guess = np.array([x0[0], x0[1], 0.0, 0.0])

        #     W = np.diag([0., 0., 1., 1., 0., 0.]) 

        #     for i in range(ocp.N):
        #         ocp.ocp_solver.set(i, "x", x_guess)
        #         ocp.ocp_solver.cost_set(i, "W", W)

        #     ocp.ocp_solver.set(ocp.N, "x", x_guess)
        #     ocp.ocp_solver.cost_set(ocp.N, "W", 2 * np.diag([0., 0., 1., 1.]))

        #     status = ocp.ocp_solver.solve()

        #     print(status)

        #     if status != 0:
        #         ocp.ocp_solver.print_statistics()

    else:
        ocp.ocp_solver.print_statistics()
        raise Exception(f'acados returned status {status}.')

    print("Execution time: %s seconds" % (time.time() - start_time))

# pr.print_stats(sort='cumtime')

plt.show()
