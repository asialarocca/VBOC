from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearnex import patch_sklearn
import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from numpy.linalg import norm as norm
import time
from double_pendulum_ocp_class import OCPdoublependulumINIT
import warnings
import math
import random
warnings.filterwarnings("ignore")

patch_sklearn()

with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPdoublependulumINIT()

    ocp_dim = ocp.nx

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = - ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    # Initialization of the SVM classifier:
    clf = Pipeline([('scaler', StandardScaler()), ('svc', svm.SVC(
        C=1e4, kernel='rbf', probability=True, class_weight={1: 1, 0: 100}, cache_size=1000))])

    # Active learning parameters:
    N_init = pow(5, ocp_dim)  # size of initial labeled set
    B = pow(5, ocp_dim)  # batch size
    etp_stop = 0.2  # active learning stopping condition
    etp_ref = 1e-4

    # # Generate low-discrepancy unlabeled samples:
    # sampler = qmc.Halton(d=ocp_dim, scramble=False)
    # sample = sampler.random(n=pow(50, ocp_dim))
    # l_bounds = [q_min, q_min, v_min, v_min]
    # u_bounds = [q_max, q_max, v_max, v_max]
    # data = qmc.scale(sample, l_bounds, u_bounds)

    # Generate random samples:
    data_q1 = np.random.uniform(q_min, q_max, size=pow(20, ocp_dim))
    data_q2 = np.random.uniform(q_min, q_max, size=pow(20, ocp_dim))
    data_v1 = np.random.uniform(v_min, v_max, size=pow(20, ocp_dim))
    data_v2 = np.random.uniform(v_min, v_max, size=pow(20, ocp_dim))
    data = np.transpose(np.array([data_q1[:], data_q2[:], data_v1[:], data_v2[:]]))

    # Generate the initial set of labeled samples:
    X_iter = [[(q_max+q_min)/2, (q_max+q_min)/2, 0., 0.]]
    res = ocp.compute_problem([(q_max+q_min)/2, (q_max+q_min)/2], [0., 0.])
    y_iter = [res]

    Xu_iter = data  # Unlabeled set

    # Training of an initial classifier:
    for n in range(N_init):
        q0 = data[n, : 2]
        v0 = data[n, 2:]

        # Data testing:
        X_iter = np.append(X_iter, [[q0[0], q0[1], v0[0], v0[1]]], axis=0)
        res = ocp.compute_problem(q0, v0)
        y_iter = np.append(y_iter, res)

        # Add intermediate states of succesfull initial conditions:
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

    circle_x = (q_max-q_min)/10
    circle_y = (v_max-v_min)/10

    dec = abs(clf.decision_function(data))
    xu = data[dec < (max(dec)-min(dec))/10]

    Xu_it = np.empty((xu.shape[0],pow(2, ocp_dim), ocp_dim))

    # Generate other random samples:
    for i in range(xu.shape[0]):
        for n in range(pow(2, ocp_dim)):

            # random angle
            alpha = math.pi * random.random()
            beta = math.pi * random.random()
            gamma = 2 * math.pi * random.random()
            # random radius
            tmp = math.sqrt(random.random())
            r_x = circle_x * tmp
            r_y = circle_y * tmp
            # calculating coordinates
            x1 = r_x * math.cos(alpha) + xu[i, 0]
            x2 = r_x * math.sin(alpha) * math.cos(beta) + xu[i, 1]
            x3 = r_y * math.sin(alpha) * math.sin(beta) * math.cos(gamma) + xu[i, 2]
            x4 = r_y * math.sin(alpha) * math.sin(beta) * math.sin(gamma) + xu[i, 3]

            Xu_it[i,n, :] = [x1, x2, x3, x4]
            # Xu_iter = np.append(Xu_iter, [[x1, x2, x3, x4]], axis=0)

    Xu_it.shape = (xu.shape[0]*pow(2, ocp_dim), ocp_dim)
    Xu_iter = np.concatenate([Xu_iter,Xu_it])

    plt.figure()
    plt.scatter(Xu_iter[:, 0], Xu_iter[:, 1], marker=".", alpha=0.5)

    plt.figure()
    plt.scatter(Xu_iter[:, 0], Xu_iter[:, 2], marker=".", alpha=0.5)

    plt.figure()
    plt.scatter(Xu_iter[:, 1], Xu_iter[:, 3], marker=".", alpha=0.5)

    print('done')
    print(Xu_iter.shape[0])

print("Execution time: %s seconds" % (time.time() - start_time))

pr.print_stats(sort='cumtime')

plt.show()
