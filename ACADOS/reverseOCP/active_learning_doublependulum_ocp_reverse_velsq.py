import cProfile
import numpy as np
import random 
import matplotlib.pyplot as plt
from sklearn import svm
import math
from numpy.linalg import norm as norm
import time
from double_pendulum_ocp_class_reverse_velsq import OCPdoublependulumRINIT
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
    clf = svm.SVC(C=1e4, kernel='rbf')

    eps = 0.1
    multip = 0.
    num = 10
    
    X_save = np.array([[(q_min+q_max)/2,(q_min+q_max)/2,0.,0., 1]])
    
    for i in range(num):
        ocp.ocp_solver.reset()
        
        q_fin = q_min + random.random() * (q_max-q_min)
        
        xe = np.array([q_max, q_fin, 0., 0.])

        W = 2 * np.diag([0., 0., -1., 0., 0., 0.]) 

        lb = np.array([q_min, q_min, 0., v_min])
        ub = np.array([q_max, q_max, v_max, v_max])

        ocp.ocp_solver.constraints_set(0, "lbx", np.array([q_min, q_min, 0., v_min]))
        ocp.ocp_solver.constraints_set(0, "ubx", np.array([q_min, q_max, v_max, v_max]))

        ocp.ocp_solver.set(0, 'x', np.array([q_min, q_fin, v_max, 0.]))

        ocp.ocp_solver.cost_set(0, "W", W)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N-1, endpoint=False)):
            x_guess = np.array([(1-tau)*q_min + tau*q_max, q_fin, v_max, 0.])
            ocp.ocp_solver.set(i+1, 'x', x_guess)

            ocp.ocp_solver.cost_set(i+1, "W", W)

            ocp.ocp_solver.constraints_set(i+1, "lbx", lb)
            ocp.ocp_solver.constraints_set(i+1, "ubx", ub)

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", xe)
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", xe)

        ocp.ocp_solver.set(ocp.N, "x", xe)

        status = ocp.ocp_solver.solve()
        ocp.ocp_solver.print_statistics()

        if status == 0:
            for f in range(ocp.N+1):
                current_val = ocp.ocp_solver.get(f, "x")
                X_save = np.append(X_save, [np.append(current_val, 1)], axis=0)
                x0 = current_val
                x0[2] = x0[2] + eps
                X_save = np.append(X_save, [np.append(x0, 0)], axis=0)

    print('check')
    
    for i in range(num):
        ocp.ocp_solver.reset()
        
        q_fin = q_min + random.random() * (q_max-q_min)
        
        xe = np.array([q_min, q_fin, 0., 0.])

        W = 2 * np.diag([0., 0., -1., 0., 0., 0.]) 

        lb = np.array([q_min, q_min, v_min, v_min])
        ub = np.array([q_max, q_max, 0., v_max])

        ocp.ocp_solver.constraints_set(0, "lbx", np.array([q_max, q_min, v_min, v_min]))
        ocp.ocp_solver.constraints_set(0, "ubx", np.array([q_max, q_max, 0., v_max]))

        ocp.ocp_solver.set(0, 'x', np.array([q_max, q_fin, v_min, 0.]))

        ocp.ocp_solver.cost_set(0, "W", W)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N-1, endpoint=False)):
            x_guess = np.array([tau*q_min + (1-tau)*q_max, q_fin, v_min, 0.])
            ocp.ocp_solver.set(i+1, 'x', x_guess)

            ocp.ocp_solver.cost_set(i+1, "W", W)

            ocp.ocp_solver.constraints_set(i+1, "lbx", lb)
            ocp.ocp_solver.constraints_set(i+1, "ubx", ub)

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", xe)
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", xe)

        ocp.ocp_solver.set(ocp.N, "x", xe)

        status = ocp.ocp_solver.solve()
        # ocp.ocp_solver.print_statistics()

        if status == 0:
            for f in range(ocp.N+1):
                current_val = ocp.ocp_solver.get(f, "x")
                X_save = np.append(X_save, [np.append(current_val, 1)], axis=0)
                x0 = current_val
                x0[2] = x0[2] - eps
                X_save = np.append(X_save, [np.append(x0, 0)], axis=0)

    print('check')
    
    for i in range(num):
        ocp.ocp_solver.reset()
        
        q_fin = q_min + random.random() * (q_max-q_min)
        
        xe = np.array([q_fin, q_max, 0., 0.])

        W = 2 * np.diag([0., 0., 0., -1., 0., 0.]) 

        lb = np.array([q_min, q_min, v_min, 0.])
        ub = np.array([q_max, q_max, v_max, v_max])

        ocp.ocp_solver.constraints_set(0, "lbx", np.array([q_min, q_min, v_min, 0.]))
        ocp.ocp_solver.constraints_set(0, "ubx", np.array([q_max, q_min, v_max, v_max]))

        ocp.ocp_solver.set(0, 'x', np.array([q_fin, q_min, 0., v_max]))

        ocp.ocp_solver.cost_set(0, "W", W)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N-1, endpoint=False)):
            x_guess = np.array([q_fin, (1-tau)*q_min + tau*q_max, 0., v_max])
            ocp.ocp_solver.set(i+1, 'x', x_guess)

            ocp.ocp_solver.cost_set(i+1, "W", W)

            ocp.ocp_solver.constraints_set(i+1, "lbx", lb)
            ocp.ocp_solver.constraints_set(i+1, "ubx", ub)

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", xe)
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", xe)

        ocp.ocp_solver.set(ocp.N, "x", xe)

        status = ocp.ocp_solver.solve()
        # ocp.ocp_solver.print_statistics()

        if status == 0:
            for f in range(ocp.N+1):
                current_val = ocp.ocp_solver.get(f, "x")
                X_save = np.append(X_save, [np.append(current_val, 1)], axis=0)
                x0 = current_val
                x0[3] = x0[3] + eps
                X_save = np.append(X_save, [np.append(x0, 0)], axis=0)

    print('check')
    
    for i in range(num):
        ocp.ocp_solver.reset()
        
        q_fin = q_min + random.random() * (q_max-q_min)
        
        xe = np.array([q_fin, q_min, 0., 0.])

        W = 2 * np.diag([0., 0., 0., -1., 0., 0.]) 

        lb = np.array([q_min, q_min, v_min, v_min])
        ub = np.array([q_max, q_max, v_max, 0.])

        ocp.ocp_solver.constraints_set(0, "lbx", np.array([q_min, q_max, v_min, v_min]))
        ocp.ocp_solver.constraints_set(0, "ubx", np.array([q_max, q_max, v_max, 0.]))

        ocp.ocp_solver.set(0, 'x', np.array([q_fin, q_max, 0., v_min]))

        ocp.ocp_solver.cost_set(0, "W", W)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N-1, endpoint=False)):
            x_guess = np.array([q_fin, tau*q_min + (1-tau)*q_max, 0., v_min])
            ocp.ocp_solver.set(i+1, 'x', x_guess)

            ocp.ocp_solver.cost_set(i+1, "W", W)

            ocp.ocp_solver.constraints_set(i+1, "lbx", lb)
            ocp.ocp_solver.constraints_set(i+1, "ubx", ub)

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", xe)
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", xe)

        ocp.ocp_solver.set(ocp.N, "x", xe)

        status = ocp.ocp_solver.solve()
        # ocp.ocp_solver.print_statistics()

        if status == 0:
            for f in range(ocp.N+1):
                current_val = ocp.ocp_solver.get(f, "x")
                X_save = np.append(X_save, [np.append(current_val, 1)], axis=0)
                x0 = current_val
                x0[3] = x0[3] - eps
                X_save = np.append(X_save, [np.append(x0, 0)], axis=0)

    print('check')

    clf.fit(X_save[:,:4], X_save[:,4])

    # Plot the results:
    plt.figure()
    h = 0.02
    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()
    Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
                    np.zeros(yrav.shape[0]), yrav])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][0] - (q_min + q_max) / 2) < 0.1
            and norm(X_save[i][2]) < 1.
        ):
            xit.append(X_save[i][1])
            yit.append(X_save[i][3])
            if X_save[i][4] < 0.5:
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
    Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
                    np.zeros(yrav.shape[0])])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][1] - (q_min + q_max) / 2) < 0.1
            and norm(X_save[i][3]) < 1.
        ):
            xit.append(X_save[i][0])
            yit.append(X_save[i][2])
            if X_save[i][4] < 0.5:
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

    # Plot the results:
    plt.figure()
    h = 0.02
    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()
    Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
                    np.zeros(yrav.shape[0]), yrav])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][0] - (q_min + q_max) / 2) < 0.2
            and norm(X_save[i][2]) < 2.
        ):
            xit.append(X_save[i][1])
            yit.append(X_save[i][3])
            if X_save[i][4] < 0.5:
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
    Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
                    np.zeros(yrav.shape[0])])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][1] - (q_min + q_max) / 2) < 0.2
            and norm(X_save[i][3]) < 2.
        ):
            xit.append(X_save[i][0])
            yit.append(X_save[i][2])
            if X_save[i][4] < 0.5:
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

    # Plot the results:
    plt.figure()
    h = 0.02
    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()
    Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
                    np.zeros(yrav.shape[0]), yrav])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][0] - (q_min + q_max) / 2) < 0.3
            and norm(X_save[i][2]) < 3.
        ):
            xit.append(X_save[i][1])
            yit.append(X_save[i][3])
            if X_save[i][4] < 0.5:
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
    Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
                    np.zeros(yrav.shape[0])])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][1] - (q_min + q_max) / 2) < 0.3
            and norm(X_save[i][3]) < 3.
        ):
            xit.append(X_save[i][0])
            yit.append(X_save[i][2])
            if X_save[i][4] < 0.5:
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

    # Plot the results:
    plt.figure()
    h = 0.02
    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()
    Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
                    np.zeros(yrav.shape[0]), yrav])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][0] - (q_min + q_max) / 2) < 0.4
            and norm(X_save[i][2]) < 4.
        ):
            xit.append(X_save[i][1])
            yit.append(X_save[i][3])
            if X_save[i][4] < 0.5:
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
    Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
                    np.zeros(yrav.shape[0])])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][1] - (q_min + q_max) / 2) < 0.4
            and norm(X_save[i][3]) < 4.
        ):
            xit.append(X_save[i][0])
            yit.append(X_save[i][2])
            if X_save[i][4] < 0.5:
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

    # Plot the results:
    h = 0.02
    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()

    plt.figure()
    Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
                    np.zeros(yrav.shape[0]), yrav])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][0] - (q_min + q_max) / 2) < 0.5
            and norm(X_save[i][2]) < 5.
        ):
            xit.append(X_save[i][1])
            yit.append(X_save[i][3])
            if X_save[i][4] < 0.5:
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
    Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
                    np.zeros(yrav.shape[0])])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][1] - (q_min + q_max) / 2) < 0.5
            and norm(X_save[i][3]) < 5.
        ):
            xit.append(X_save[i][0])
            yit.append(X_save[i][2])
            if X_save[i][4] < 0.5:
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

# pr.print_stats(sort='cumtime')

plt.show()
