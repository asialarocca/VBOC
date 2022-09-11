import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from numpy.linalg import norm as norm
import time
from pendulum_ocp_class import OCPpendulumINIT
import warnings
import math
import random
warnings.filterwarnings("ignore")

print_stats = 0
show_plots = 1
print_cprof = 0

if show_plots:
    x_min, x_max = 0.0, np.pi / 2
    y_min, y_max = -10.0, 10.0
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPpendulumINIT()

    ocp_dim = ocp.nx

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = - ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    # Initialization of the SVM classifier:
    clf = svm.SVC(C=1e5, kernel='rbf', probability=True,
                  class_weight={1: 1, 0: 10}, cache_size=1000)

    # Active learning parameters:
    N_init = pow(10, ocp_dim)  # size of initial labeled set
    B = pow(10, ocp_dim)  # batch size
    etp_stop = 0.2  # active learning stopping condition

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(20, ocp_dim))
    l_bounds = [q_min, v_min]
    u_bounds = [q_max, v_max]
    data = qmc.scale(sample, l_bounds, u_bounds)
    
    Xu_iter = data  # Unlabeled set

    if show_plots:
        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(
            Xu_iter[:, 0],
            Xu_iter[:, 1],
            marker=".",
            alpha=1.
        )
        plt.xlim([0.0, np.pi / 2])
        plt.ylim([-10.0, 10.0])
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Initial unlabeled set")
        plt.grid()

    # Generate the initial set of labeled samples:
    contour = pow(10, ocp_dim)
    X_iter = np.empty((contour, ocp_dim))
    y_iter = np.full((contour, 1), [0])
    r = np.random.random(size=(contour, ocp_dim - 1))
    k = np.random.randint(ocp_dim, size=(contour, 1))
    j = np.random.randint(2, size=(contour, 1))
    x = np.zeros((contour, ocp_dim))
    for i in np.arange(contour):
        x[i, np.arange(ocp_dim)[np.arange(ocp_dim) != k[i]]] = r[i, :]
        x[i, k[i]] = j[i]

    X_iter[:, 0] = x[:, 0] * (q_max + (q_max-q_min)/100 -
                              (q_min - (q_max-q_min)/100)) + q_min - (q_max-q_min)/100
    X_iter[:, 1] = x[:, 1] * (v_max + (v_max-v_min)/100 -
                              (v_min - (v_max-v_min)/100)) + v_min - (v_max-v_min)/100


    # Training of an initial classifier:
    for n in range(N_init):
        q0 = data[n, 0]
        v0 = data[n, 1]

        # Data testing:
        res = ocp.compute_problem(q0, v0)
        if res != 2:
            X_iter = np.append(X_iter, [[q0, v0]], axis=0)
            if res == 1:
                y_iter = np.append(y_iter, res)
            else:
                y_iter = np.append(y_iter, res)

        # Add intermediate states of succesfull initial conditions
        if res == 1:
            for f in range(1, ocp.N, int(ocp.N/3)):
                current_val = ocp.ocp_solver.get(f, "x")
                X_iter = np.append(X_iter, [current_val], axis=0)
                y_iter = np.append(y_iter, 1)

    # Delete tested data from the unlabeled set:
    Xu_iter = np.delete(Xu_iter, range(N_init), axis=0)

    # Training of the classifier:
    clf.fit(X_iter, y_iter)

    print("INITIAL CLASSIFIER TRAINED")

    # Active learning:
    k = 0  # iteration number
    etpmax = 1

    while not (etpmax < etp_stop or Xu_iter.shape[0] == 0):

        if Xu_iter.shape[0] < B:
            B = Xu_iter.shape[0]

        # Compute the shannon entropy of the unlabeled samples:
        prob_xu = clf.predict_proba(Xu_iter)
        etp = entropy(prob_xu, axis=1)
        maxindex = np.argpartition(etp, -B)[-B:]  # indexes of the uncertain samples

        etpmax = max(etp[maxindex])  # max entropy used for the stopping condition

        # Add the B most uncertain samples to the labeled set:
        for x in range(B):
            q0 = Xu_iter[maxindex[x], 0]
            v0 = Xu_iter[maxindex[x], 1]

            # Data testing:
            res = ocp.compute_problem(q0, v0)
            if res != 2:
                X_iter = np.append(X_iter, [[q0, v0]], axis=0)
                if res == 1:
                    y_iter = np.append(y_iter, res)
                else:
                    y_iter = np.append(y_iter, res)

        # Delete tested data from the unlabeled set:
        Xu_iter = np.delete(Xu_iter, maxindex, axis=0)

        # Re-fit the model with the new selected X_iter:
        clf.fit(X_iter, y_iter)

        k += 1

        print("CLASSIFIER", k, "TRAINED")

        if print_stats:
            # Print statistics:
            y_pred = clf.predict(X_iter)
            # accuracy (calculated on the training set)
            print("Accuracy:", metrics.accuracy_score(y_iter, y_pred))
            print("False positives:", metrics.confusion_matrix(y_iter, y_pred)[0, 1])
            print("Number of samples in the labeled set:", X_iter.shape[0])

    if show_plots:
        # Plot the results:
        plt.figure(figsize=(6, 5))
        out = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        out = out.reshape(xx.shape)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.contour(xx, yy, out, levels=[0], linewidths=(2,), colors=("k",))
        scatter = plt.scatter(
            X_iter[:, 0],
            X_iter[:, 1],
            c=y_iter,
            marker=".",
            alpha=0.5,
            cmap=plt.cm.Paired,
        )
        plt.xlim([0.0, np.pi / 2 - 0.01])
        plt.ylim([-10.0, 10.0])
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Classifier")
        hand = scatter.legend_elements()[0]
        plt.legend(handles=hand, labels=("Non viable", "Viable"), loc='upper right')
        plt.grid()
        
        
    print('FIRST DATA ANALYZED')

    rad_q = (q_max - q_min) / 20
    rad_v = (v_max - v_min) / 20

    n_points = pow(3, ocp_dim)
    
    # Compute the shannon entropy of the initial unlabeled samples:
    prob_x = clf.predict_proba(data)
    etp = entropy(prob_x, axis=1)
    etmax = max(etp)
    
    xu = data[etp > etmax/2]

    #dec = abs(clf.decision_function(data))
    #xu = data[dec < (max(dec) - min(dec)) / 10]

    Xu_it = np.empty((xu.shape[0], n_points, ocp_dim))

    # Generate other random samples:
    for i in range(xu.shape[0]):
        for n in range(n_points):
            # random angle
            alpha = 2 * math.pi * random.random()
            # random radius
            tmp = math.sqrt(random.random())
            r_x = rad_q * tmp
            r_y = rad_v * tmp
            # calculating coordinates
            x = r_x * math.cos(alpha) + xu[i, 0]
            y = r_y * math.sin(alpha) + xu[i, 1]
            Xu_it[i, n, :] = [x, y]

    Xu_it.shape = (xu.shape[0] * n_points, ocp_dim)
    data = np.concatenate([data, Xu_it])
    Xu_iter = Xu_it

    if show_plots:
        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(
            data[:, 0],
            data[:, 1],
            marker=".",
            alpha=1.
        )
        plt.xlim([0.0, np.pi / 2])
        plt.ylim([-10.0, 10.0])
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Initial unlabeled set")
        plt.grid()

    etpmax = 1

    while not (etpmax < etp_stop or Xu_iter.shape[0] == 0):

        if Xu_iter.shape[0] < B:
            B = Xu_iter.shape[0]

        # Compute the shannon entropy of the unlabeled samples:
        prob_xu = clf.predict_proba(Xu_iter)
        etp = entropy(prob_xu, axis=1)
        maxindex = np.argpartition(etp, -B)[-B:]  # indexes of the uncertain samples

        etpmax = max(etp[maxindex])  # max entropy used for the stopping condition

        # Add the B most uncertain samples to the labeled set:
        for x in range(B):
            q0 = Xu_iter[maxindex[x], 0]
            v0 = Xu_iter[maxindex[x], 1]

            # Data testing:
            res = ocp.compute_problem(q0, v0)
            if res != 2:
                X_iter = np.append(X_iter, [[q0, v0]], axis=0)
                if res == 1:
                    y_iter = np.append(y_iter, res)
                else:
                    y_iter = np.append(y_iter, res)
                    
            # Add intermediate states of succesfull initial conditions
            if res == 1:
                for f in range(1, ocp.N, int(ocp.N/3)):
                    current_val = ocp.ocp_solver.get(f, "x")
                    X_iter = np.append(X_iter, [current_val], axis=0)
                    y_iter = np.append(y_iter, 1)

        # Delete tested data from the unlabeled set:
        Xu_iter = np.delete(Xu_iter, maxindex, axis=0)

        # Re-fit the model with the new selected X_iter:
        clf.fit(X_iter, y_iter)

        k += 1

        print("CLASSIFIER", k, "TRAINED")

        if print_stats:
            # Print statistics:
            y_pred = clf.predict(X_iter)
            # accuracy (calculated on the training set)
            print("Accuracy:", metrics.accuracy_score(y_iter, y_pred))
            print("False positives:", metrics.confusion_matrix(y_iter, y_pred)[0, 1])

    if show_plots:
        # Plot the results:
        plt.figure(figsize=(6, 5))
        out = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        out = out.reshape(xx.shape)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.contour(xx, yy, out, levels=[0], linewidths=(2,), colors=("k",))
        scatter = plt.scatter(
            X_iter[:, 0],
            X_iter[:, 1],
            c=y_iter,
            marker=".",
            alpha=0.5,
            cmap=plt.cm.Paired,
        )
        plt.xlim([0.0, np.pi / 2 - 0.01])
        plt.ylim([-10.0, 10.0])
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Classifier")
        hand = scatter.legend_elements()[0]
        plt.legend(handles=hand, labels=("Non viable", "Viable"), loc='upper right')
        plt.grid()

    print("Execution time: %s seconds" % (time.time() - start_time))

if print_cprof:
    pr.print_stats(sort='cumtime')

if show_plots:
    plt.show()
