import numpy as np
from numpy import nan
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from numpy.linalg import norm as norm
import time
from pendulum_ocp_class import OCPpendulum


def create_classifier():

    # Active learning parameters:
    N_init = 15  # size of initial labeled set

    ocp = OCPpendulum()

    v_max = ocp.dthetamax
    v_min = - ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=2, scramble=False)
    sample = sampler.random(n=10000)
    l_bounds = [q_min, v_min]
    u_bounds = [q_max, v_max]
    data = qmc.scale(sample, l_bounds, u_bounds)

    # Generate the initial set of labeled samples:
    X_iter = [[0., 0.]]
    y_iter = [ocp.compute_problem(0., 0.)]

    Xu_iter = data

    for n in range(N_init):
        q0 = data[n, 0]
        v0 = data[n, 1]

        ocp = OCPpendulum()
        X_iter = np.append(X_iter, [[q0, v0]], axis=0)
        res = ocp.compute_problem(q0, v0)
        y_iter = np.append(y_iter, res)
        Xu_iter = np.delete(Xu_iter, n, axis=0)

        # Add intermediate states of succesfull initial conditions
        if res == 1:
            for l in range(1, ocp.N):
                if norm(ocp.simX[l, 1]) > 0.01:
                    X_iter = np.append(
                        X_iter, [[ocp.simX[l, 0], ocp.simX[l, 1]]], axis=0)
                    y_iter = np.append(y_iter, 1)

    clf = svm.SVC(C=100000, kernel='rbf', probability=True,
                  class_weight='balanced')
    clf.fit(X_iter, y_iter)

    print("INITIAL CLASSIFIER TRAINED")

    plt.figure()
    x_min, x_max = 0., np.pi/2
    y_min, y_max = -10., 10.
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_iter[:, 0], X_iter[:, 1], c=y_iter,
                marker=".", alpha=0.5, cmap=plt.cm.Paired)

    plt.xlim([0., np.pi/2 - 0.02])
    plt.ylim([-10., 10.])

    return clf, X_iter, data
