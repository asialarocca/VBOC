import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from numpy.linalg import norm as norm
import time
from pendulum_ocp_class import OCPpendulum
import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

# Ocp initialization:
ocp = OCPpendulum()

ocp_dim = ocp.nx # number of simulated time steps

# Position and velocity bounds:
v_max = ocp.dthetamax
v_min = - ocp.dthetamax
q_max = ocp.thetamax
q_min = ocp.thetamin

# Initialization of the SVM classifier:
clf = svm.SVC(C=1000, kernel='rbf', probability=True, class_weight={0: 1000, 1: 1})

# Active learning parameters:
N_init = pow(5, ocp_dim)  # size of initial labeled set
B = pow(5, ocp_dim)  # batch size

# Generate low-discrepancy unlabeled samples:
sampler = qmc.Halton(d=2, scramble=False)
sample = sampler.random(n=pow(100, ocp_dim))
l_bounds = [q_min, v_min]
u_bounds = [q_max, v_max]
data = qmc.scale(sample, l_bounds, u_bounds)

# Generate the initial set of labeled samples:
X_iter = [[(q_max+q_min)/2, 0.]]
y_iter = [1]

Xu_iter = data # Unlabeled seti

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
        for f in range(1, ocp.N, 2):
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

print("INITIAL CLASSIFIER TRAINED")

# Print statistics: 
y_pred = clf.predict(X_iter)
print("Accuracy:", metrics.accuracy_score(y_iter, y_pred)) # accuracy (calculated on the training set)
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
k = 0 # iteration number
while True:
    # Compute the shannon entropy of the unlabeled samples:
    prob_xu = clf.predict_proba(Xu_iter)
    etp = entropy(prob_xu, axis=1)

    maxindex = np.argpartition(etp, -B)[-B:]  # indexes of the uncertain samples

    # Stopping condition:
    if max(etp[maxindex]) < 0.1:
        break

    # Add the B most uncertain samples to the labeled set:
    for x in range(B):
        q0 = Xu_iter[maxindex[x], 0]
        v0 = Xu_iter[maxindex[x], 1]

        # Data testing:
        X_iter = np.append(X_iter, [[q0, v0]], axis=0)
        res = ocp.compute_problem(q0, v0)
        y_iter = np.append(y_iter, res)

    # Add intermediate states of succesfull initial conditions
    if res == 1:
        for f in range(1, ocp.N, 2):
            current_val = ocp.ocp_solver.get(f, "x")
            if norm(current_val[1]) > 0.01:
                X_iter = np.append(X_iter, [current_val], axis=0)
                y_iter = np.append(y_iter, 1)
            else:
                break

    # Delete tested data from the unlabeled set:
    Xu_iter = np.delete(Xu_iter, maxindex, axis=0)

    # Re-fit the model with the new selected X_iter:
    clf.fit(X_iter, y_iter)
    
    print("CLASSIFIER", k+1, "TRAINED")

    # Print statistics: 
    y_pred = clf.predict(X_iter)
    print("Accuracy:", metrics.accuracy_score(y_iter, y_pred)) # accuracy (calculated on the training set)
    print("False positives:", metrics.confusion_matrix(y_iter, y_pred)[0, 1])

    k += 1

# Plot the results:
plt.figure()
x_min, x_max = 0., np.pi/2
y_min, y_max = -10., 10.
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_iter[:, 0], X_iter[:, 1], c=y_iter, marker=".", alpha=0.5, cmap=plt.cm.Paired)
plt.xlim([0., np.pi/2 - 0.02])
plt.ylim([-10., 10.])
plt.grid()

print("Execution time: %s seconds" % (time.time() - start_time))

plt.show()
