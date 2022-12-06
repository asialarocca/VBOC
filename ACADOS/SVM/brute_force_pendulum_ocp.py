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
from scipy.optimize import fsolve

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
    
    delta_t = 0.01

    def func_min_val(x):
        thetamin_new, q_dotdot_max = x
        q_dot = - delta_t * q_dotdot_max
        M = ocp.d**2 * ocp.m
        h = ocp.b * q_dot - ocp.d * math.sin(thetamin_new) * ocp.m * ocp.g
        return (q_dotdot_max - (ocp.Fmax - h)/M,
                - thetamin_new + q_min + 1/2 * delta_t**2 * q_dotdot_max)

    def func_max_val(x):
        thetamax_new, q_dotdot_min = x
        q_dot = - delta_t * q_dotdot_min
        M = ocp.d**2 * ocp.m
        h = ocp.b * q_dot - ocp.d * math.sin(thetamax_new) * ocp.m * ocp.g
        return (q_dotdot_min - (-ocp.Fmax - h)/M,
                - thetamax_new + q_max + 1/2 * delta_t**2 * q_dotdot_min)

    guess_min = (q_min, ocp.Fmax/ocp.d**2 * ocp.m)
    guess_max = (q_max, -ocp.Fmax/ocp.d**2 * ocp.m)

    thetamin_new, q_dotdot_max = fsolve(func_min_val, guess_min)
    thetamax_new, q_dotdot_min = fsolve(func_max_val, guess_max)
    
    print(thetamin_new-q_min,thetamax_new-q_max)

    # thetamin_new = q_min - delta__t ** 2 * \
    #     (ocp.g * ocp.d * ocp.m + ocp.Fmax) / (-ocp.d ** 2 * ocp.m + ocp.b * delta__t) / 2
    # thetamax_new = q_max - delta__t ** 2 * \
    #     (ocp.g * ocp.d * ocp.m - ocp.Fmax) / (-ocp.d ** 2 * ocp.m + ocp.b * delta__t) / 2

    # print(q_min, thetamin_new, q_max, thetamax_new)

    ocp.set_bounds(thetamin_new, thetamax_new)

    # Initialization of the SVM classifier:
    clf = svm.SVC(
        C=1e5, kernel="rbf", probability=True, class_weight={1: 1, 0: 10}, cache_size=1000
    )

    # Active learning parameters:
    N_init = pow(10, ocp_dim)  # size of initial labeled set
    B = pow(10, ocp_dim)  # batch size
    etp_stop = 0.2  # active learning stopping condition

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(50, ocp_dim))
    l_bounds = [q_min, v_min]
    u_bounds = [q_max, v_max]
    data = qmc.scale(sample, l_bounds, u_bounds)

    # # Generate random samples:
    # data_q = np.random.uniform(q_min, q_max, size=pow(100, ocp_dim))
    # data_v = np.random.uniform(v_min, v_max, size=pow(100, ocp_dim))
    # data = np.transpose(np.array([data_q[:], data_v[:]]))

    Xu_iter = data  # Unlabeled set

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

    X_iter[:, 0] = x[:, 0] * (q_max + (q_max-q_min)/50 -
                              (q_min - (q_max-q_min)/50)) + q_min - (q_max-q_min)/50
    X_iter[:, 1] = x[:, 1] * (v_max + (v_max-v_min)/50 -
                              (v_min - (v_max-v_min)/50)) + v_min - (v_max-v_min)/50

    for n in range(data.shape[0]):
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

    # Training of the classifier:
    clf.fit(X_iter, y_iter)

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

        # Plot of the decision function:
        plt.figure(figsize=(6, 5))
        levels = np.linspace(out.min(), out.max(), 10)
        plt.contourf(xx, yy, out, levels=levels, cmap='plasma')
        this = plt.contour(xx, yy, out, levels=levels, colors=('k',), linewidths=(1,))
        plt.clabel(this, fmt='%2.1f', colors='w', fontsize=11)
        #df = plt.contour(xx, yy, out, levels=[0], linewidths=(2,), colors=("k",))
        #plt.clabel(df, fmt='%2.1f', colors='w', fontsize=11)
        plt.xlabel('Initial position [rad]')
        plt.ylabel('Initial velocity [rad/s]')
        plt.title('Decision function')
        plt.grid()

        # Plot of the entropy:
        plt.figure(figsize=(6, 5))
        out = entropy(clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]), axis=1)
        out = out.reshape(xx.shape)
        levels = np.linspace(out.min(), out.max(), 5)
        plt.contourf(xx, yy, out, levels=levels)
        this = plt.contour(xx, yy, out, levels=levels, colors=("k",), linewidths=(1,))
        plt.clabel(this, fmt="%2.1f", colors="w", fontsize=11)
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Entropy")
        plt.grid()

        # Plot of the support vectors:
        sup = clf.support_
        sup_X = X_iter[sup]
        sup_y = y_iter[sup]

        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(sup_X[:, 0], sup_X[:, 1], c=sup_y,
                              marker=".", alpha=1, cmap=plt.cm.Paired)
        plt.xlabel('Initial position [rad]')
        plt.ylabel('Initial velocity [rad/s]')
        plt.title('Support vectors')
        hand = scatter.legend_elements()[0]
        plt.legend(handles=hand, labels=("Non viable", "Viable"), loc='upper right')
        plt.grid()

print("Execution time: %s seconds" % (time.time() - start_time))
print("Mean entropy: " + str(np.mean(entropy(clf.predict_proba(data), axis=1))))
y_pred = clf.predict(X_iter)
print(metrics.accuracy_score(y_iter, y_pred))

if print_cprof:
    pr.print_stats(sort="cumtime")

if show_plots:


    plt.show()
