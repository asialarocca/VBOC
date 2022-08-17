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
    v_min = -ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    # Initialization of the SVM classifier:
    clf = svm.SVC(
        C=1e5, kernel="rbf", probability=True, class_weight="balanced", cache_size=1000
    )

    # Active learning parameters:
    N_init = pow(10, ocp_dim)  # size of initial labeled set
    B = pow(10, ocp_dim)  # batch size
    etp_stop = 0.2  # active learning stopping condition

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(100, ocp_dim))
    l_bounds = [q_min, v_min]
    u_bounds = [q_max, v_max]
    data = qmc.scale(sample, l_bounds, u_bounds)

    # # Generate random samples:
    # data_q = np.random.uniform(q_min, q_max, size=pow(100, ocp_dim))
    # data_v = np.random.uniform(v_min, v_max, size=pow(100, ocp_dim))
    # data = np.transpose(np.array([data_q[:], data_v[:]]))

    Xu_iter = data  # Unlabeled set

    # Generate the initial set of labeled samples:
    res = ocp.compute_problem((q_max + q_min) / 2, 0.0)
    if res != 2:
        X_iter = np.double([[(q_max + q_min) / 2, 0.0]])
        y_iter = np.byte([res])
    else:
        raise Exception("Max iteration reached")

    # # Generate the initial set of labeled samples:
    # X_iter = np.empty((10*2*ocp_dim, ocp_dim))
    # y_iter = np.zeros((10*2*ocp_dim))
    # q_test = np.linspace(q_min, q_max, num=10)
    # v_test = np.linspace(v_min, v_max, num=10)

    # for p in range(10):
    #     X_iter[p, :] = [q_min - 0.01, v_test[p]]
    #     X_iter[p + 10, :] = [q_max + 0.01, v_test[p]]
    #     X_iter[p + 20, :] = [q_test[p], v_min - 0.1]
    #     X_iter[p + 30, :] = [q_test[p], v_max + 0.1]

    # Training of an initial classifier:
    for n in range(N_init):
        q0 = data[n, 0]
        v0 = data[n, 1]

        # Data testing:
        res = ocp.compute_problem(q0, v0)
        if res != 2:
            X_iter = np.append(X_iter, [[q0, v0]], axis=0)
            y_iter = np.append(y_iter, res)
        else:
            raise Exception("Max iteration reached")

        # Add intermediate states of succesfull initial conditions
        if res == 1:
            for f in range(1, ocp.N, int(ocp.N / 3)):
                current_val = ocp.ocp_solver.get(f, "x")
                X_iter = np.append(X_iter, [current_val], axis=0)
                y_iter = np.append(y_iter, 1)

    # Delete tested data from the unlabeled set:
    Xu_iter = np.delete(Xu_iter, range(N_init), axis=0)

    # Training of the classifier:
    clf.fit(X_iter, y_iter)

    print("INITIAL CLASSIFIER TRAINED")

    if print_stats:
        # Print statistics:
        y_pred = clf.predict(X_iter)
        # accuracy (calculated on the training set)
        print("Accuracy:", metrics.accuracy_score(y_iter, y_pred))
        print("False positives:", metrics.confusion_matrix(y_iter, y_pred)[0, 1])

    if show_plots:
        # Plot the results:
        plt.figure()
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
        plt.legend(handles=hand, labels=("Non viable", "Viable"))

    if show_plots:
        # Plot of the entropy:
        plt.figure()
        out = entropy(clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]), axis=1)
        out = out.reshape(xx.shape)
        levels = np.linspace(out.min(), out.max(), 10)
        plt.contourf(xx, yy, out, levels=levels)
        this = plt.contour(xx, yy, out, levels=levels, colors=("k",), linewidths=(1,))
        plt.clabel(this, fmt="%2.1f", colors="w", fontsize=11)
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Decision function")

    performance_history = [1]

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
                y_iter = np.append(y_iter, res)
            else:
                raise Exception("Max iteration reached")

            # Add intermediate states of succesfull initial conditions:
            if res == 1:
                for f in range(1, ocp.N, int(ocp.N / 3)):
                    current_val = ocp.ocp_solver.get(f, "x")
                    prob_sample = clf.predict_proba([current_val])
                    etp_sample = entropy(prob_sample, axis=1)

        # Delete tested data from the unlabeled set:
        Xu_iter = np.delete(Xu_iter, maxindex, axis=0)

        # Re-fit the model with the new selected X_iter:
        clf.fit(X_iter, y_iter)

        k += 1

        print("CLASSIFIER", k, "TRAINED")

        performance_history.append(etpmax)

        if print_stats:
            # Print statistics:
            y_pred = clf.predict(X_iter)
            # accuracy (calculated on the training set)
            print("Accuracy:", metrics.accuracy_score(y_iter, y_pred))
            print("False positives:", metrics.confusion_matrix(y_iter, y_pred)[0, 1])

    if show_plots:
        # Plot the results:
        plt.figure()
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
        plt.legend(handles=hand, labels=("Non viable", "Viable"))

    if show_plots:
        # Plot of the entropy:
        plt.figure()
        out = entropy(clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]), axis=1)
        out = out.reshape(xx.shape)
        levels = np.linspace(out.min(), out.max(), 10)
        plt.contourf(xx, yy, out, levels=levels)
        this = plt.contour(xx, yy, out, levels=levels, colors=("k",), linewidths=(1,))
        plt.clabel(this, fmt="%2.1f", colors="w", fontsize=11)
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Decision function")

print("Execution time: %s seconds" % (time.time() - start_time))

if print_cprof:
    pr.print_stats(sort="cumtime")

if show_plots:
    plt.figure()
    plt.plot(performance_history[1:])
    plt.scatter(range(len(performance_history[1:])), performance_history[1:])
    plt.xlabel("Iteration number")
    plt.ylabel("Maximum entropy")
    plt.title("Maximum entropy evolution")

    plt.show()