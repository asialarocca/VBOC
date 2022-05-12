import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from numpy.linalg import norm as norm
import time
from pendulum_ocp_class import OCPpendulumINIT, OCPpendulumSVM
import warnings
warnings.filterwarnings("ignore")

with cProfile.Profile() as pr:

    start_time = time.time()

    # ---------------PROBLEM INITIALIZATION-----------------
    # ------------------------------------------------------

    # Ocp initialization:
    ocp = OCPpendulumINIT()

    ocp_dim = ocp.nx

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = - ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    # Initialization of the SVM classifier:
    clf = svm.SVC(C=1000, kernel='rbf', probability=True, class_weight={1: 1, 0: 100})

    # Active learning parameters:
    N_init = pow(5, ocp_dim)  # size of initial labeled set
    B = pow(5, ocp_dim)  # batch size
    etp_stop = 0.2  # active learning stopping condition
    etp_ref = 1e-4

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=2, scramble=False)
    sample = sampler.random(n=pow(80, ocp_dim))
    l_bounds = [q_min, v_min]
    u_bounds = [q_max, v_max]
    data = qmc.scale(sample, l_bounds, u_bounds)

    # Generate the initial set of labeled samples:
    X_iter = [[(q_max+q_min)/2, 0.]]
    y_iter = [ocp.compute_problem((q_max+q_min)/2, 0.)]
    y_weight = [1]

    # # Generate the initial set of labeled samples:
    # X_iter = np.empty((ocp_dim * 2 * 10, ocp_dim))
    # y_iter = np.empty(ocp_dim * 2 * 10)

    # for p in range(10):
    #     v_test = v_min + p*(v_max - v_min)/10
    #     q_test = q_min + p*(q_max - q_min)/10

    #     X_iter[p, :] = [q_min - 0.1, v_test]
    #     y_iter[p] = 0

    #     X_iter[p + 10, :] = [q_max + 0.1, v_test]
    #     y_iter[p + 10] = 0

    #     X_iter[p + 20, :] = [q_test, v_min - 0.1]
    #     y_iter[p + 20] = 0

    #     X_iter[p + 30, :] = [q_test, v_max + 0.1]
    #     y_iter[p + 30] = 0

    # Generate the initial set of unlabeled samples:
    Xu_iter = data

    r = 0  # iterative number of trained classifier

    # ------------------------------------------------------

    # --------------TRAIN INITIAL CLASSIFIER----------------
    # ------------------------------------------------------

    # Training of an initial classifier:
    for n in range(N_init):
        q0 = data[n, 0]
        v0 = data[n, 1]

        # Data testing:
        X_iter = np.append(X_iter, [[q0, v0]], axis=0)
        res = ocp.compute_problem(q0, v0)
        y_iter = np.append(y_iter, res)

        # Add intermediate states of succesfull initial conditions:
        if res == 1:
            for f in range(1, ocp.N, int(ocp.N/3)):
                current_val = ocp.ocp_solver.get(f, "x")
                if norm(current_val[1]) > 0.01:
                    X_iter = np.append(X_iter, [current_val], axis=0)
                    y_iter = np.append(y_iter, 1)
                else:
                    break

    # Delete tested data from the unlabeled set:
    Xu_iter = np.delete(Xu_iter, range(N_init), axis=0)

    # Training of the classifier:
    clf.fit(X_iter, y_iter)

    etpmax = 1
    ind_set = np.array((0))  # indexes of values to delete from the unlabeled set
    ind_test = np.arange(Xu_iter.shape[0])  # indexes of values of the unlabeled set to consider

    # Active learning:
    while True:
        # Stopping condition:
        xu_shape = ind_test.shape[0]
        # xu_shape = Xu_iter.shape[0]
        if etpmax < etp_stop or xu_shape == 0:
            break

        if xu_shape < B:
            B = xu_shape

        # Compute the shannon entropy of the unlabeled samples:
        prob_xu = clf.predict_proba(Xu_iter[ind_test])
        # prob_xu = clf.predict_proba(Xu_iter)
        etp = entropy(prob_xu, axis=1)
        maxindex = np.argpartition(etp, -B)[-B:]  # indexes of the uncertain samples

        etpmax = max(etp[maxindex])  # max entropy used for the stopping condition

        # Add the B most uncertain samples to the labeled set:
        for x in range(B):
            q0 = Xu_iter[ind_test[maxindex[x]], 0]
            v0 = Xu_iter[ind_test[maxindex[x]], 1]
            # q0 = Xu_iter[maxindex[x], 0]
            # v0 = Xu_iter[maxindex[x], 1]

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
                    if etp_sample > etp_ref:
                        X_iter = np.append(X_iter, [current_val], axis=0)
                        y_iter = np.append(y_iter, 1)
                    else:
                        break

        # Delete tested data from the unlabeled set:
        # Xu_iter = np.delete(Xu_iter, maxindex, axis=0)
        ind_set = np.append(ind_set, ind_test[maxindex])
        ind_test = np.delete(ind_test, maxindex)
        etp = np.delete(etp, maxindex)
        ind_test = ind_test[etp > etpmax*etp_ref]

        # Re-fit the model with the new selected X_iter:
        clf.fit(X_iter, y_iter)

    Xu_iter = np.delete(Xu_iter, ind_set, axis=0)

    r += 1

    print("----------- CLASSIFIER", r, "TRAINED ------------")

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

    n_sum = sum(clf.predict(data[1:1000, :]))  # used for the stopping criterion

    # ------------------------------------------------------

    # ------------TRAIN SUCCESSIVE CLASSIFIERS--------------
    # ------------------------------------------------------

    while True:
        # Reinitialize the ocp with the current classifier as the final constraint:
        del ocp
        ocp = OCPpendulumSVM(clf, X_iter)

        # The unlabeled set is redefined with the data that resulted unfeasible from testing or classification:
        data_xinit = X_iter[y_iter == 0]
        data_xinit = np.delete(data_xinit, range(40), axis=0)
        data_xuinit = Xu_iter[clf.predict(Xu_iter) == 0]
        Xu_iter = np.concatenate([data_xinit, data_xuinit])

        # The labeled set is redefined with the data previously tested as feasible:
        x_temp_b = X_iter[:40, :]
        y_temp_b = y_iter[:40]
        x_temp_f = X_iter[y_iter == 1]
        y_temp_f = y_iter[y_iter == 1]
        X_iter = np.concatenate([x_temp_b, x_temp_f])
        y_iter = np.concatenate([y_temp_b, y_temp_f])

        # plt.figure()
        # plt.scatter(Xu_iter[:, 0], Xu_iter[:, 1])

        # Build a new initial classifier by adding some newly tested data to the labeled set:
        for n in range(N_init):  # - int(N_init * n_sum / (1000))
            q0 = Xu_iter[n, 0]
            v0 = Xu_iter[n, 1]

            # Data testing:
            X_iter = np.append(X_iter, [[q0, v0]], axis=0)
            res = ocp.compute_problem(q0, v0)
            y_iter = np.append(y_iter, res)

            # Add intermediate states of succesfull initial conditions
            if res == 1:
                for f in range(1, ocp.N, int(ocp.N/3)):
                    current_val = ocp.ocp_solver.get(f, "x")
                    if norm(current_val[1]) > 0.01:
                        X_iter = np.append(X_iter, [current_val], axis=0)
                        y_iter = np.append(y_iter, 1)
                    else:
                        break

        # Delete tested data from the unlabeled set:
        Xu_iter = np.delete(Xu_iter, range(N_init), axis=0)

        # Training of the classifier:
        clf.fit(X_iter, y_iter)

        etpmax = 1
        ind_set = np.array((0))  # indexes of values to delete from the unlabeled set
        ind_test = np.arange(Xu_iter.shape[0])  # indexes of values of the unlabeled set to consider

        # Active learning:
        while True:
            # Stopping condition:
            xu_shape = ind_test.shape[0]
            # xu_shape = Xu_iter.shape[0]
            if etpmax < etp_stop or xu_shape == 0:
                break

            if xu_shape < B:
                B = xu_shape

            # Compute the shannon entropy of the unlabeled samples:
            prob_xu = clf.predict_proba(Xu_iter[ind_test])
            # prob_xu = clf.predict_proba(Xu_iter)
            etp = entropy(prob_xu, axis=1)
            maxindex = np.argpartition(etp, -B)[-B:]  # indexes of the uncertain samples

            # plt.figure()
            # plt.scatter(Xu_iter[ind_test, 0], Xu_iter[ind_test, 1], c=etp)

            etpmax = max(etp[maxindex])  # max entropy used for the stopping condition

            # Add the B most uncertain samples to the labeled set:
            for x in range(B):
                q0 = Xu_iter[ind_test[maxindex[x]], 0]
                v0 = Xu_iter[ind_test[maxindex[x]], 1]
                # q0 = Xu_iter[maxindex[x], 0]
                # v0 = Xu_iter[maxindex[x], 1]

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
                        if etp_sample > etp_ref:
                            X_iter = np.append(X_iter, [current_val], axis=0)
                            y_iter = np.append(y_iter, 1)
                        else:
                            break

            # Delete tested data from the unlabeled set:
            # Xu_iter = np.delete(Xu_iter, maxindex, axis=0)
            ind_set = np.append(ind_set, ind_test[maxindex])
            ind_test = np.delete(ind_test, maxindex)
            etp = np.delete(etp, maxindex)
            ind_test = ind_test[etp > etpmax * etp_ref]

            # Re-fit the model with the new selected X_iter:
            clf.fit(X_iter, y_iter)

        Xu_iter = np.delete(Xu_iter, ind_set, axis=0)

        r += 1

        print("----------- CLASSIFIER", r, "TRAINED ------------")

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

        # Stopping condition: compare two consecutive classifiers by the number of positively classified
        # data from a predefined pool of samples:
        # number of positively classified data
        n_temp = sum(clf.predict(data[1: pow(10, ocp_dim), :]))
        if n_temp - n_sum == 0:
            break
        n_sum = n_temp

    # ------------------------------------------------------

    print("Execution time: %s seconds" % (time.time() - start_time))

pr.print_stats(sort='cumtime')

plt.show()
