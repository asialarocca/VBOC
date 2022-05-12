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
    clf = svm.SVC(C=1000, kernel='rbf', probability=True,
                  class_weight={1: 1, 0: 100}, cache_size=1000)

    # Active learning parameters:
    N_init = pow(10, ocp_dim)  # size of initial labeled set
    B = pow(5, ocp_dim)  # batch size
    etp_stop = 0.2  # active learning stopping condition
    etp_ref_x = 1e-4
    etp_ref_xu = 1e-4

    data_q = np.random.uniform(q_min, q_max, size=pow(50, ocp_dim))
    data_v = np.random.uniform(v_min, v_max, size=pow(50, ocp_dim))

    data = np.transpose(np.array([data_q[:], data_v[:]]))

    Xu_iter = data

    # Generate the initial set of labeled samples:
    X_iter = np.empty((ocp_dim * 2 * 10, ocp_dim))
    y_iter = np.zeros(ocp_dim * 2 * 10)

    q_test = np.linspace(q_min, q_max, num=10)
    v_test = np.linspace(v_min, v_max, num=10)

    for p in range(10):
        X_iter[p, :] = [q_min - 0.1, v_test[p]]
        X_iter[p + 10, :] = [q_max + 0.1, v_test[p]]
        X_iter[p + 20, :] = [q_test[p], v_min - 0.1]
        X_iter[p + 30, :] = [q_test[p], v_max + 0.1]

    # Training of an initial classifier:
    for n in range(N_init):
        q0 = data[n, 0]
        v0 = data[n, 1]

        # Data testing:
        X_iter = np.append(X_iter, [[q0, v0]], axis=0)
        res = ocp.compute_problem(q0, v0)
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

    # # Print statistics:
    # y_pred = clf.predict(X_iter)
    # # accuracy (calculated on the training set)
    # print("Accuracy:", metrics.accuracy_score(y_iter, y_pred))
    # print("False positives:", metrics.confusion_matrix(y_iter, y_pred)[0, 1])

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
    plt.xlim([0., np.pi/2 - 0.01])
    plt.ylim([-10., 10.])

    # Active learning:
    k = 0  # iteration number
    etpmax = 1

    et = entropy(clf.predict_proba(X_iter), axis=1)
    X_iter = X_iter[et > etp_ref_x]
    y_iter = y_iter[et > etp_ref_x]

    # plt.figure()
    # plt.scatter(X_iter[:, 0], X_iter[:, 1], c=y_iter, marker=".", alpha=0.5, cmap=plt.cm.Paired)
    while True:
        # Stopping condition:
        xu_shape = Xu_iter.shape[0]
        if etpmax < etp_stop or xu_shape == 0:
            break

        if xu_shape < B:
            B = xu_shape

        # Compute the shannon entropy of the unlabeled samples:
        # index = np.random.randint(Xu_iter.shape[1], size=Bu)
        prob_xu = clf.predict_proba(Xu_iter)
        etp = entropy(prob_xu, axis=1)
        maxindex = np.argpartition(etp, -B)[-B:]  # indexes of the uncertain samples

        # plt.figure()
        # plt.scatter(Xu_iter[:, 0], Xu_iter[:, 1], c=etp)

        etpmax = max(etp[maxindex])  # max entropy used for the stopping condition
        print(etpmax)

        # Add the B most uncertain samples to the labeled set:
        for x in range(B):
            q0 = Xu_iter[maxindex[x], 0]
            v0 = Xu_iter[maxindex[x], 1]

            # Data testing:
            X_iter = np.append(X_iter, [[q0, v0]], axis=0)
            res = ocp.compute_problem(q0, v0)
            y_iter = np.append(y_iter, res)

            # Add intermediate states of succesfull initial conditions:
            if res == 1:
                for f in range(1, ocp.N, int(ocp.N/3)):
                    current_val = ocp.ocp_solver.get(f, "x")
                    prob_sample = clf.predict_proba([current_val])
                    etp_sample = entropy(prob_sample, axis=1)
                    if etp_sample > etp_ref_x:
                        X_iter = np.append(X_iter, [current_val], axis=0)
                        y_iter = np.append(y_iter, 1)
                    else:
                        break

        # Delete tested data from the unlabeled set:
        Xu_iter = np.delete(Xu_iter, maxindex, axis=0)
        # etp = np.delete(etp, maxindex)
        # Xu_iter = Xu_iter[etp > etp_ref_xu]

        # Re-fit the model with the new selected X_iter:
        clf.fit(X_iter, y_iter)

        k += 1

        print("CLASSIFIER", k, "TRAINED")

        # # Print statistics:
        # y_pred = clf.predict(X_iter)
        # # accuracy (calculated on the training set)
        # print("Accuracy:", metrics.accuracy_score(y_iter, y_pred))
        # print("False positives:", metrics.confusion_matrix(y_iter, y_pred)[0, 1])

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
        plt.xlim([0., np.pi/2 - 0.01])
        plt.ylim([-10., 10.])
        plt.grid()

        et = entropy(clf.predict_proba(X_iter), axis=1)
        X_iter = X_iter[et > etp_ref_x]
        y_iter = y_iter[et > etp_ref_x]

        # plt.figure()
        # plt.scatter(X_iter[:, 0], X_iter[:, 1], c=y_iter, marker=".", alpha=0.5, cmap=plt.cm.Paired)

    print("Execution time: %s seconds" % (time.time() - start_time))

pr.print_stats(sort='cumtime')

plt.show()
