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
warnings.filterwarnings("ignore")

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
    clf = svm.SVC(C=1000, kernel='rbf', probability=True, class_weight='balanced')

    # Active learning parameters:
    N_init = pow(5, ocp_dim)  # size of initial labeled set
    B = pow(10, ocp_dim)  # batch size
    etp_stop = 0.2  # active learning stopping condition

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=2, scramble=False)
    sample = sampler.random(n=pow(100, ocp_dim))
    l_bounds = [q_min, v_min]
    u_bounds = [q_max, v_max]
    data = qmc.scale(sample, l_bounds, u_bounds)

    # Generate the initial set of labeled samples:
    X_iter = [[(q_max+q_min)/2, 0.]]
    y_iter = [ocp.compute_problem((q_max+q_min)/2, 0.)]
    y_weight = [1]

    Xu_iter = data  # Unlabeled set

    # Training of an initial classifier:
    for n in range(N_init):
        q0 = data[n, 0]
        v0 = data[n, 1]

        # Data testing:
        X_iter = np.append(X_iter, [[q0, v0]], axis=0)
        res = ocp.compute_problem(q0, v0)
        y_iter = np.append(y_iter, res)
        y_weight = np.append(y_weight, 1)

        # Add intermediate states of succesfull initial conditions
        if res == 1:
            for f in range(1, ocp.N, int(ocp.N/3)):
                current_val = ocp.ocp_solver.get(f, "x")
                if norm(current_val[1]) > 0.01:
                    X_iter = np.append(X_iter, [current_val], axis=0)
                    y_iter = np.append(y_iter, 1)
                    y_weight = np.append(y_weight, 1)
                else:
                    break

    # Delete tested data from the unlabeled set:
    Xu_iter = np.delete(Xu_iter, range(N_init), axis=0)

    # Check boundaries:
    for p in range(10):
        v_test = v_min + p*(v_max - v_min)/10
        X_iter = np.append(X_iter, [[q_min, v_test]], axis=0)
        res = ocp.compute_problem(q_min, v_test)
        y_iter = np.append(y_iter, res)
        if res == 1:
            y_weight = np.append(y_weight, 1)
        else:
            y_weight = np.append(y_weight, 10)
        X_iter = np.append(X_iter, [[q_max, v_test]], axis=0)
        res = ocp.compute_problem(q_max, v_test)
        y_iter = np.append(y_iter, res)
        if res == 1:
            y_weight = np.append(y_weight, 1)
        else:
            y_weight = np.append(y_weight, 10)
    for p in range(10):
        q_test = q_min + p*(q_max - q_min)/10
        X_iter = np.append(X_iter, [[q_test, v_min]], axis=0)
        res = ocp.compute_problem(q_test, v_min)
        y_iter = np.append(y_iter, res)
        if res == 1:
            y_weight = np.append(y_weight, 1)
        else:
            y_weight = np.append(y_weight, 10)
        X_iter = np.append(X_iter, [[q_test, v_max]], axis=0)
        res = ocp.compute_problem(q_test, v_max)
        y_iter = np.append(y_iter, res)
        if res == 1:
            y_weight = np.append(y_weight, 1)
        else:
            y_weight = np.append(y_weight, 10)

    # Training of the classifier:
    clf.fit(X_iter, y_iter, sample_weight=y_weight)

    print("INITIAL CLASSIFIER TRAINED")

    # Print statistics:
    y_pred = clf.predict(X_iter)
    # accuracy (calculated on the training set)
    print("Accuracy:", metrics.accuracy_score(y_iter, y_pred))
    print("False positives:", metrics.confusion_matrix(y_iter, y_pred)[0, 1])

    # Plot the results:
    plt.figure()
    x_min, x_max = 0., np.pi/2
    y_min, y_max = -10., 10.
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_iter[:, 0], X_iter[:, 1], c=y_iter, marker=".", alpha=0.5, cmap=plt.cm.Paired)
    plt.xlim([0., np.pi/2 - 0.02])
    plt.ylim([-10., 10.])

    # Active learning:
    k = 0  # iteration number
    etpmax = 1

    while True:
        # Stopping condition:
        xu_shape = Xu_iter.shape[0]
        if etpmax < etp_stop or xu_shape == 0:
            break

        if xu_shape < B:
            B = xu_shape

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
            X_iter = np.append(X_iter, [[q0, v0]], axis=0)
            res = ocp.compute_problem(q0, v0)
            y_iter = np.append(y_iter, res)
            if res == 1:
                y_weight = np.append(y_weight, 1)
            else:
                y_weight = np.append(y_weight, 3)

            # Add intermediate states of succesfull initial conditions:
            if res == 1:
                for f in range(1, ocp.N, int(ocp.N/3)):
                    current_val = ocp.ocp_solver.get(f, "x")
                    prob_sample = clf.predict_proba([current_val])
                    etp_sample = entropy(prob_sample, axis=1)
                    if etp_sample > etp_stop/10:
                        X_iter = np.append(X_iter, [current_val], axis=0)
                        y_iter = np.append(y_iter, 1)
                        y_weight = np.append(y_weight, 1)
                    else:
                        break

        # Delete tested data from the unlabeled set:
        Xu_iter = np.delete(Xu_iter, maxindex, axis=0)
        # etp = np.delete(etp, maxindex)
        # Xu_iter = Xu_iter[etp > etpmax * etp_stop/10]

        # Re-fit the model with the new selected X_iter:
        clf.fit(X_iter, y_iter, sample_weight=y_weight)

        k += 1

        print("CLASSIFIER", k, "TRAINED")

        # Print statistics:
        y_pred = clf.predict(X_iter)
        # accuracy (calculated on the training set)
        print("Accuracy:", metrics.accuracy_score(y_iter, y_pred))
        print("False positives:", metrics.confusion_matrix(y_iter, y_pred)[0, 1])

        # Plot the results:
        plt.figure()
        x_min, x_max = 0., np.pi/2
        y_min, y_max = -10., 10.
        h = .01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(X_iter[:, 0], X_iter[:, 1], c=y_iter, marker=".", alpha=0.5, cmap=plt.cm.Paired)
        plt.xlim([0., np.pi/2 - 0.01])
        plt.ylim([-10., 10.])
        plt.grid()

    print("Execution time: %s seconds" % (time.time() - start_time))

pr.print_stats(sort='cumtime')

plt.show()
