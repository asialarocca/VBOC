import numpy as np
from numpy import nan
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from sklearn import svm
#from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from numpy.linalg import norm as norm
import time
#from pendulum_ocp_class import OCPpendulum
from pendulum_ocp_class_prova import OCPpendulum
import warnings

start_time = time.time()
warnings.filterwarnings("ignore")

# Active learning parameters:
N_init = 100  # size of initial labeled set
B = 10  # batch size

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

    X_iter = np.append(X_iter, [[q0, v0]], axis=0)
    res = ocp.compute_problem(q0, v0)
    y_iter = np.append(y_iter, res)
    Xu_iter = np.delete(Xu_iter, n, axis=0)

    # Add intermediate states of succesfull initial conditions
    if res == 1:
        for f in range(1, ocp.N):
            if norm(ocp.simX[f, 1]) > 0.001:
                X_iter = np.append(
                    X_iter, [[ocp.simX[f, 0], ocp.simX[f, 1]]], axis=0)
                y_iter = np.append(y_iter, 1)

clf = svm.SVC(C=100000, kernel='rbf', probability=True,
              class_weight='balanced')
clf.fit(X_iter, y_iter)

print("INITIAL CLASSIFIER TRAINED")

# Model Accuracy (calculated on the training set):
print("Accuracy:", metrics.accuracy_score(y_iter, clf.predict(X_iter)))

plt.figure()
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
        res = ocp.compute_problem(q0, v0)
        y_iter = np.append(y_iter, res)

    # Add intermediate states of succesfull initial conditions
    if res == 1:
        for f in range(1, ocp.N):
            if norm(ocp.simX[f, 1]) > 0.001:
                X_iter = np.append(
                    X_iter, [[ocp.simX[f, 0], ocp.simX[f, 1]]], axis=0)
                y_iter = np.append(y_iter, 1)

    Xu_iter = np.delete(Xu_iter, maxindex, axis=0)

    # Re-fit the model with the new selected X_iter:
    clf.fit(X_iter, y_iter)
    print("CLASSIFIER", k+1, "TRAINED")

    # Model Accuracy (calculated on the training set):
    print("Accuracy:", metrics.accuracy_score(y_iter, clf.predict(X_iter)))

    k += 1

plt.figure()
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

plt.show()

print("Execution time: %s seconds" % (time.time() - start_time))
