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
    clf = svm.SVC(C=1e4, kernel='rbf')

    eps = 0.1
    multip = 0.1
    num = 100

    X_save = np.array([[(q_min+q_max)/2,(q_min+q_max)/2,0.,0., 1]])

    X_test = np.empty((0,4))
    
    for i in range(num):
        ocp.ocp_solver.reset()
        
        q2_fin = q_min + random.random() * (q_max-q_min)
        
        xe = np.array([q_max, q2_fin, 0., 0.])

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", xe)
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", xe)

        ocp.ocp_solver.constraints_set(0, "lbx", np.array([ocp.thetamin, ocp.thetamin, -ocp.dthetamax, -ocp.dthetamax]))
        ocp.ocp_solver.constraints_set(0, "ubx", np.array([ocp.thetamax, ocp.thetamax, ocp.dthetamax, ocp.dthetamax]))

        if q2_fin > (ocp.thetamax + ocp.thetamin)/2:
            q2_guess = ocp.thetamax
            q2dot_guess = ocp.dthetamax
        else:
            q2_guess = ocp.thetamin
            q2dot_guess = -ocp.dthetamax

        x_guess = np.array([ocp.thetamin, q2_guess, ocp.dthetamax, q2dot_guess])

        ran1 = random.random() * multip
        ran2 = random.random() * multip
        normal = math.sqrt(ran1 **2 + ran2 **2 + 1)

        W = 2 * np.diag([0., -ran1, -1., -ran2, 0., 0.]) 

        lb = np.array([ocp.thetamin, ocp.thetamin, 0, -ocp.dthetamax])
        ub = np.array([ocp.thetamax, ocp.thetamax, ocp.dthetamax, ocp.dthetamax])

        for i in range(ocp.N):
            ocp.ocp_solver.set(i, "x", x_guess)
            ocp.ocp_solver.cost_set(i, "W", W)
            ocp.ocp_solver.constraints_set(i, "lbx", lb)
            ocp.ocp_solver.constraints_set(i, "ubx", ub)  

        ocp.ocp_solver.set(ocp.N, "x", xe)
        ocp.ocp_solver.cost_set(ocp.N, "W", np.diag([0., 0., 0., 0.]))

        status = ocp.ocp_solver.solve()

        if status == 0:
            for f in range(ocp.N+1):
                current_val = ocp.ocp_solver.get(f, "x")
                X_save = np.append(X_save, [np.append(current_val, 1)], axis=0)
                x0 = current_val
                x0[1] = x0[1] + eps * math.sqrt(ran1)/normal
                x0[2] = x0[2] + eps * 1/normal
                x0[3] = x0[3] + eps * math.sqrt(ran2)/normal
                X_test = np.append(X_test, [x0], axis=0)

    ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([ocp.thetamin, ocp.thetamin, 0., 0.]))
    ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([ocp.thetamax, ocp.thetamax, 0., 0.]))

    lb = np.array([ocp.thetamin, ocp.thetamin, -ocp.dthetamax, -ocp.dthetamax])
    ub = np.array([ocp.thetamax, ocp.thetamax, ocp.dthetamax, ocp.dthetamax])
    W = np.diag([0., 0., 1., 1., 0., 0.]) 

    for i in range(ocp.N):
        ocp.ocp_solver.cost_set(i, "W", W)
        ocp.ocp_solver.constraints_set(i, "lbx", lb)
        ocp.ocp_solver.constraints_set(i, "ubx", ub)

    ocp.ocp_solver.cost_set(ocp.N, "W", 2 * np.diag([0., 0., 1., 1.]))

    for i in range(len(X_test)):

        for f in range(ocp.N+1):
            ocp.ocp_solver.reset()

            x0 = X_test[f + i*ocp.N]
            ocp.ocp_solver.constraints_set(0, "lbx", x0)
            ocp.ocp_solver.constraints_set(0, "ubx", x0)

            x_guess = np.array([x0[0], x0[1], 0.0, 0.0])
            for i in range(ocp.N+1):
                ocp.ocp_solver.set(i, "x", x_guess)
            
            status = ocp.ocp_solver.solve()

            if status == 4:
                X_save = np.append(X_save, [np.append(x0, 0)], axis=0)

    X_test = np.empty((0,4))
    
    for i in range(num):
        ocp.ocp_solver.reset()
        
        q2_fin = q_min + random.random() * (q_max-q_min)
        
        xe = np.array([q_min, q2_fin, 0., 0.])

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", xe)
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", xe)

        ocp.ocp_solver.constraints_set(0, "lbx", np.array([ocp.thetamin, ocp.thetamin, -ocp.dthetamax, -ocp.dthetamax]))
        ocp.ocp_solver.constraints_set(0, "ubx", np.array([ocp.thetamax, ocp.thetamax, ocp.dthetamax, ocp.dthetamax]))

        # x_guess = np.array([ocp.thetamax, q2_fin, -ocp.dthetamax, 0.])

        if q2_fin > (ocp.thetamax + ocp.thetamin)/2:
            q2_guess = ocp.thetamax
            q2dot_guess = ocp.dthetamax
        else:
            q2_guess = ocp.thetamin
            q2dot_guess = -ocp.dthetamax

        x_guess = np.array([ocp.thetamax, q2_guess, -ocp.dthetamax, q2dot_guess])

        ran1 = random.random() * multip
        ran2 = random.random() * multip
        normal = math.sqrt(ran1 **2 + ran2 **2 + 1)

        W = 2 * np.diag([0., -ran1, -1., -ran2, 0., 0.])

        lb = np.array([ocp.thetamin, ocp.thetamin, -ocp.dthetamax, -ocp.dthetamax])
        ub = np.array([ocp.thetamax, ocp.thetamax, 0., ocp.dthetamax])

        for i in range(ocp.N):
            ocp.ocp_solver.set(i, "x", x_guess)
            ocp.ocp_solver.cost_set(i, "W", W)
            ocp.ocp_solver.constraints_set(i, "lbx", lb)
            ocp.ocp_solver.constraints_set(i, "ubx", ub)  

        ocp.ocp_solver.set(ocp.N, "x", xe)
        ocp.ocp_solver.cost_set(ocp.N, "W", np.diag([0., 0., 0., 0.]))

        status = ocp.ocp_solver.solve()

        if status == 0:
            for f in range(0, ocp.N+1):
                current_val = ocp.ocp_solver.get(f, "x")
                X_save = np.append(X_save, [np.append(current_val, 1)], axis=0)
                x0 = current_val
                x0[1] = x0[1] - eps * math.sqrt(ran1)/normal
                x0[2] = x0[2] - eps * 1/normal
                x0[3] = x0[3] - eps * math.sqrt(ran2)/normal
                X_test = np.append(X_test, [x0], axis=0)

    ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([ocp.thetamin, ocp.thetamin, 0., 0.]))
    ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([ocp.thetamax, ocp.thetamax, 0., 0.]))

    lb = np.array([ocp.thetamin, ocp.thetamin, -ocp.dthetamax, -ocp.dthetamax])
    ub = np.array([ocp.thetamax, ocp.thetamax, ocp.dthetamax, ocp.dthetamax])
    W = np.diag([0., 0., 1., 1., 0., 0.]) 

    for i in range(ocp.N):
        ocp.ocp_solver.cost_set(i, "W", W)
        ocp.ocp_solver.constraints_set(i, "lbx", lb)
        ocp.ocp_solver.constraints_set(i, "ubx", ub)

    ocp.ocp_solver.cost_set(ocp.N, "W", 2 * np.diag([0., 0., 1., 1.]))

    for i in range(len(X_test)):

        for f in range(ocp.N+1):
            ocp.ocp_solver.reset()

            x0 = X_test[f + i*ocp.N]
            ocp.ocp_solver.constraints_set(0, "lbx", x0)
            ocp.ocp_solver.constraints_set(0, "ubx", x0)

            x_guess = np.array([x0[0], x0[1], 0.0, 0.0])
            for i in range(ocp.N+1):
                ocp.ocp_solver.set(i, "x", x_guess)
            
            status = ocp.ocp_solver.solve()

            if status == 4:
                X_save = np.append(X_save, [np.append(x0, 0)], axis=0)

    X_test = np.empty((0,4))
    
    for i in range(num):
        ocp.ocp_solver.reset()
        
        q2_fin = q_min + random.random() * (q_max-q_min)
        
        xe = np.array([q2_fin, q_max, 0., 0.])

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", xe)
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", xe)

        ocp.ocp_solver.constraints_set(0, "lbx", np.array([ocp.thetamin, ocp.thetamin, -ocp.dthetamax, -ocp.dthetamax]))
        ocp.ocp_solver.constraints_set(0, "ubx", np.array([ocp.thetamax, ocp.thetamax, ocp.dthetamax, ocp.dthetamax]))

        # x_guess = np.array([q2_fin, ocp.thetamin, 0., ocp.dthetamax])

        if q2_fin > (ocp.thetamax + ocp.thetamin)/2:
            q2_guess = ocp.thetamax
            q2dot_guess = ocp.dthetamax
        else:
            q2_guess = ocp.thetamin
            q2dot_guess = -ocp.dthetamax

        x_guess = np.array([q2_guess, ocp.thetamin, q2dot_guess , ocp.dthetamax])

        ran1 = random.random() * multip
        ran2 = random.random() * multip
        normal = math.sqrt(ran1 **2 + ran2 **2 + 1)

        W = 2 * np.diag([-ran1, 0., -ran2, -1., 0., 0.]) 
        
        lb = np.array([ocp.thetamin, ocp.thetamin, -ocp.dthetamax, 0])
        ub = np.array([ocp.thetamax, ocp.thetamax, ocp.dthetamax, ocp.dthetamax])
        
        for i in range(ocp.N):
            ocp.ocp_solver.set(i, "x", x_guess)
            ocp.ocp_solver.cost_set(i, "W", W)
            ocp.ocp_solver.constraints_set(i, "lbx", lb)
            ocp.ocp_solver.constraints_set(i, "ubx", ub)  

        ocp.ocp_solver.set(ocp.N, "x", xe)
        ocp.ocp_solver.cost_set(ocp.N, "W", np.diag([0., 0., 0., 0.]))

        status = ocp.ocp_solver.solve()

        if status == 0:
            for f in range(0, ocp.N+1):
                current_val = ocp.ocp_solver.get(f, "x")
                X_save = np.append(X_save, [np.append(current_val, 1)], axis=0)
                x0 = current_val
                x0[0] = x0[0] + eps * math.sqrt(ran1)/normal
                x0[2] = x0[2] + eps * 1/normal
                x0[3] = x0[3] + eps * math.sqrt(ran2)/normal
                X_test = np.append(X_test, [x0], axis=0)

    ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([ocp.thetamin, ocp.thetamin, 0., 0.]))
    ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([ocp.thetamax, ocp.thetamax, 0., 0.]))

    lb = np.array([ocp.thetamin, ocp.thetamin, -ocp.dthetamax, -ocp.dthetamax])
    ub = np.array([ocp.thetamax, ocp.thetamax, ocp.dthetamax, ocp.dthetamax])
    W = np.diag([0., 0., 1., 1., 0., 0.]) 

    for i in range(ocp.N):
        ocp.ocp_solver.cost_set(i, "W", W)
        ocp.ocp_solver.constraints_set(i, "lbx", lb)
        ocp.ocp_solver.constraints_set(i, "ubx", ub)

    ocp.ocp_solver.cost_set(ocp.N, "W", 2 * np.diag([0., 0., 1., 1.]))

    for i in range(len(X_test)):

        for f in range(ocp.N+1):
            ocp.ocp_solver.reset()

            x0 = X_test[f + i*ocp.N]
            ocp.ocp_solver.constraints_set(0, "lbx", x0)
            ocp.ocp_solver.constraints_set(0, "ubx", x0)

            x_guess = np.array([x0[0], x0[1], 0.0, 0.0])
            for i in range(ocp.N+1):
                ocp.ocp_solver.set(i, "x", x_guess)
            
            status = ocp.ocp_solver.solve()

            if status == 4:
                X_save = np.append(X_save, [np.append(x0, 0)], axis=0)

    X_test = np.empty((0,4))
    
    for i in range(num):
        ocp.ocp_solver.reset()
        
        q2_fin = q_min + random.random() * (q_max-q_min)
        
        xe = np.array([q2_fin, q_min, 0., 0.])

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", xe)
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", xe)

        ocp.ocp_solver.constraints_set(0, "lbx", np.array([ocp.thetamin, ocp.thetamin, -ocp.dthetamax, -ocp.dthetamax]))
        ocp.ocp_solver.constraints_set(0, "ubx", np.array([ocp.thetamax, ocp.thetamax, ocp.dthetamax, ocp.dthetamax]))

        # x_guess = np.array([q1_fin, ocp.thetamax, 0., -ocp.dthetamax])

        if q2_fin > (ocp.thetamax + ocp.thetamin)/2:
            q2_guess = ocp.thetamax
            q2dot_guess = ocp.dthetamax
        else:
            q2_guess = ocp.thetamin
            q2dot_guess = -ocp.dthetamax

        x_guess = np.array([q2_guess, ocp.thetamax, q2dot_guess , -ocp.dthetamax])

        ran1 = random.random() * multip
        ran2 = random.random() * multip
        normal = math.sqrt(ran1 **2 + ran2 **2 + 1)

        W = 2 * np.diag([-ran1, 0., -ran2, -1., 0., 0.])

        lb = np.array([ocp.thetamin, ocp.thetamin, -ocp.dthetamax, -ocp.dthetamax])
        ub = np.array([ocp.thetamax, ocp.thetamax, ocp.dthetamax, 0.])
        
        for i in range(ocp.N):
            ocp.ocp_solver.set(i, "x", x_guess)
            ocp.ocp_solver.cost_set(i, "W", W)
            ocp.ocp_solver.constraints_set(i, "lbx", lb)
            ocp.ocp_solver.constraints_set(i, "ubx", ub)  

        ocp.ocp_solver.set(ocp.N, "x", xe)
        ocp.ocp_solver.cost_set(ocp.N, "W", np.diag([0., 0., 0., 0.]))

        status = ocp.ocp_solver.solve()

        if status == 0:
            for f in range(0, ocp.N+1):
                current_val = ocp.ocp_solver.get(f, "x")
                X_save = np.append(X_save, [np.append(current_val, 1)], axis=0)
                x0 = current_val
                x0[0] = x0[0] - eps * math.sqrt(ran1)/normal
                x0[2] = x0[2] - eps * 1/normal
                x0[3] = x0[3] - eps * math.sqrt(ran2)/normal
                X_test = np.append(X_test, [x0], axis=0)

    ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([ocp.thetamin, ocp.thetamin, 0., 0.]))
    ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([ocp.thetamax, ocp.thetamax, 0., 0.]))

    lb = np.array([ocp.thetamin, ocp.thetamin, -ocp.dthetamax, -ocp.dthetamax])
    ub = np.array([ocp.thetamax, ocp.thetamax, ocp.dthetamax, ocp.dthetamax])
    W = np.diag([0., 0., 1., 1., 0., 0.]) 

    for i in range(ocp.N):
        ocp.ocp_solver.cost_set(i, "W", W)
        ocp.ocp_solver.constraints_set(i, "lbx", lb)
        ocp.ocp_solver.constraints_set(i, "ubx", ub)

    ocp.ocp_solver.cost_set(ocp.N, "W", 2 * np.diag([0., 0., 1., 1.]))

    for i in range(len(X_test)):

        for f in range(ocp.N+1):
            ocp.ocp_solver.reset()

            x0 = X_test[f + i*ocp.N]
            ocp.ocp_solver.constraints_set(0, "lbx", x0)
            ocp.ocp_solver.constraints_set(0, "ubx", x0)

            x_guess = np.array([x0[0], x0[1], 0.0, 0.0])
            for i in range(ocp.N+1):
                ocp.ocp_solver.set(i, "x", x_guess)
            
            status = ocp.ocp_solver.solve()

            if status == 4:
                X_save = np.append(X_save, [np.append(x0, 0)], axis=0)
       
    print(X_save)

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
            if norm(X_save[i][4]) < 0.5:
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
            if norm(X_save[i][4]) < 0.5:
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
            norm(X_save[i][0] - (q_min + q_max) / 2) < 0.5
            and norm(X_save[i][2]) < 5.
        ):
            xit.append(X_save[i][1])
            yit.append(X_save[i][3])
            if norm(X_save[i][4]) < 0.5:
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
            if norm(X_save[i][4]) < 0.5:
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
