import numpy as np
from numpy import nan
from numpy.linalg import norm
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import time
from PID_pendulum_bangbang import PIDpendulum
import config_file as conf
import warnings
import math

start_time = time.time()
warnings.filterwarnings("ignore")

# Active learning parameters:
N_init = 100  # size of initial labeled set
# N_iter = 0		#number of active learning iteration
B = 10  # batch size

pid = PIDpendulum(conf)

v_max = pid.v_max
v_min = pid.v_min
q_max = pid.q_max
q_min = pid.q_min

# Generate low-discrepancy unlabeled samples:
sampler = qmc.Halton(d=2, scramble=False)
sample = sampler.random(n=10000)
l_bounds = [q_min, v_min]
u_bounds = [q_max, v_max]
data = qmc.scale(sample, l_bounds, u_bounds)

# plt.figure()
#plt.scatter(data[:,0], data[:,1], marker=".", alpha=0.5);

# Generate the initial set of labeled samples:
X_iter = [[0, 0]]
y_iter = [pid.compute_problem(np.zeros(1), np.zeros(1))]

Xu_iter = data

# Counters of positive and negative samples:
#n_pos = 0
#n_neg = 0

for n in range(N_init):
    q0 = data[n, 0]
    v0 = data[n, 1]

    X_iter = np.append(X_iter, [[q0, v0]], axis=0)
    res = pid.compute_problem(np.array([q0]), np.array([v0]))
    y_iter = np.append(y_iter, res)
    Xu_iter = np.delete(Xu_iter, n, axis=0)

    # if res == 1:
    #	n_pos += 1
    # else:
    #	n_neg += 1

    # Add intermediate states of succesfull initial conditions
    if res == 1:
        q_traj = pid.q[:pid.niter]
        v_traj = pid.v[:pid.niter]
        if int(q_traj.shape[0]/5) != 0:
            for l in range(1, q_traj.shape[0], int(q_traj.shape[0]/5)):
                X_iter = np.append(X_iter, [[q_traj[l], v_traj[l]]], axis=0)
                y_iter = np.append(y_iter, 1)

# Create and train the initial svm classifier:
#param_grid = {'C': [100000], 'kernel': ['rbf'], 'probability': [True], 'class_weight': ['balanced']}
#clf = GridSearchCV(svm.SVC(), param_grid, refit = True)

clf = svm.SVC(C=100000, kernel='rbf', probability=True,
              class_weight='balanced')
clf.fit(X_iter, y_iter)

print("INITIAL CLASSIFIER TRAINED")

# Model Accuracy (calculated on the training set):
print("Accuracy:", metrics.accuracy_score(y_iter, clf.predict(X_iter)))

plt.figure()
#x_min, x_max = X_iter[:,0].min(), X_iter[:,0].max()
#y_min, y_max = X_iter[:,1].min(), X_iter[:,1].max()
x_min, x_max = 0., np.pi/2
y_min, y_max = -10., 10.
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_iter[:, 0], X_iter[:, 1], c=y_iter,
            marker=".", alpha=0.5, cmap=plt.cm.Paired)

plt.xlim([0., np.pi/2 - 0.02])
plt.ylim([-10., 10.])

sup = clf.support_
sup_X = X_iter[sup]
sup_y = y_iter[sup]

plt.figure()
plt.scatter(sup_X[:, 0], sup_X[:, 1], c=sup_y,
            marker=".", alpha=0.5, cmap=plt.cm.Paired)

out = clf.decision_function(data)
plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=out,
            marker=".", alpha=0.5, cmap=plt.cm.Paired)

# y_score = clf.decision_function(data)
# y_score = np.where(abs(y_score) <= 0.1, 1, 0)
# plt.figure()
# plt.scatter(data[:, 0], data[:, 1], c=y_score,
#             marker=".", alpha=0.5, cmap=plt.cm.Paired)

# DECISION FUNCTION
# dual_coef = clf.dual_coef_
# sup_vec = clf.support_vectors_
# const = clf.intercept_

# x = X_iter[:1, :]
# output = 0
# for i in range(sup.shape[0]):
#     output += dual_coef[0, i] * \
#         math.exp(- (norm(input - sup_vec[i])**2)/(2*X_iter.var()))
# output += const

# # Update the labeled set with active learning and re-train the classifier:
# for k in range(N_iter):

k = 0
while True:
    # Compute the shannon entropy of the unlabeled samples:
    prob_xu = clf.predict_proba(Xu_iter)
    etp = np.empty(prob_xu.shape[0])*nan
    for p in range(prob_xu.shape[0]):
        etp[p] = entropy(prob_xu[p])

    # Add the B most uncertain samples to the labeled set:
    # indexes of the uncertain samples
    maxindex = np.argpartition(etp, -B)[-B:]

    if sum(etp[maxindex])/B < 0.1:
        break

    for x in range(B):
        q0 = Xu_iter[maxindex[x], 0]
        v0 = Xu_iter[maxindex[x], 1]

        X_iter = np.append(X_iter, [[q0, v0]], axis=0)
        res = pid.compute_problem(np.array([q0]), np.array([v0]))
        y_iter = np.append(y_iter, res)

        # Add intermediate states of succesfull initial conditions
        if res == 1:
            q_traj = pid.q[:pid.niter]
            v_traj = pid.v[:pid.niter]
            if int(q_traj.shape[0]/5) != 0:
                for l in range(1, q_traj.shape[0], int(q_traj.shape[0]/5)):
                    X_iter = np.append(
                        X_iter, [[q_traj[l], v_traj[l]]], axis=0)
                    y_iter = np.append(y_iter, 1)

    Xu_iter = np.delete(Xu_iter, maxindex, axis=0)

    # Re-fit the model with the new selected X_iter:
    clf.fit(X_iter, y_iter)
    print("CLASSIFIER", k+1, "TRAINED")

    # Model Accuracy (calculated on the training set):
    print("Accuracy:", metrics.accuracy_score(y_iter, clf.predict(X_iter)))

    k += 1

plt.figure()
#x_min, x_max = X_iter[:,0].min(), X_iter[:,0].max()
#y_min, y_max = X_iter[:,1].min(), X_iter[:,1].max()
x_min, x_max = 0., np.pi/2
y_min, y_max = -10., 10.
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_iter[:, 0], X_iter[:, 1], c=y_iter,
            marker=".", alpha=0.5, cmap=plt.cm.Paired)

plt.xlim([0., np.pi/2 - 0.02])
plt.ylim([-10., 10.])

sup = clf.support_
sup_X = X_iter[sup]
sup_y = y_iter[sup]

plt.figure()
plt.scatter(sup_X[:, 0], sup_X[:, 1], c=sup_y,
            marker=".", alpha=0.5, cmap=plt.cm.Paired)

sup = clf.support_
sup_X = X_iter[sup]
sup_y = y_iter[sup]

plt.figure()
plt.scatter(sup_X[:, 0], sup_X[:, 1], c=sup_y,
            marker=".", alpha=0.5, cmap=plt.cm.Paired)

out = clf.decision_function(data)
plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=out,
            marker=".", alpha=0.5, cmap=plt.cm.Paired)

plt.show()

print("Execution time: %s seconds" % (time.time() - start_time))
