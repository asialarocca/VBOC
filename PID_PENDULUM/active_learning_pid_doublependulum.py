import numpy as np
from numpy import nan
from scipy.stats import entropy, qmc
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import time
from PID_doublependulum import PIDdoublependulum
import config_file_doublep as conf
import warnings

start_time = time.time()
warnings.filterwarnings("ignore")

# Active learning parameters:
N_init = 100  # size of initial labeled set
B = 10

pid = PIDdoublependulum(conf)

v_max = pid.v_max
v_min = pid.v_min
q_max = pid.q_max
q_min = pid.q_min

# Generate low-discrepancy unlabeled samples:
sampler = qmc.Halton(d=4, scramble=False)
sample = sampler.random(n=100000000)
l_bounds = [q_min[0], q_min[1], v_min[0], v_min[1]]
u_bounds = [q_max[0], q_max[1], v_max[0], v_max[1]]
data = qmc.scale(sample, l_bounds, u_bounds)

# plt.figure()
#ax = plt.axes(projection='3d')
#ax.scatter3D(data[:,0], data[:,1], data[:,2], marker=".", alpha=0.5);

# Generate the initial set of labeled samples:
X_iter = [[0., 0., 0., 0.]]
y_iter = [pid.compute_problem(np.zeros(2), np.zeros(2))]

Xu_iter = data

for n in range(N_init):
    q0 = [data[n, 0], data[n, 1]]
    v0 = [data[n, 2], data[n, 3]]
    X_iter = np.append(
        X_iter, [[data[n, 0], data[n, 1], data[n, 2], data[n, 3]]], axis=0)
    res = pid.compute_problem(np.array(q0), np.array(v0))
    y_iter = np.append(y_iter, res)
    Xu_iter = np.delete(Xu_iter, n, axis=0)

# Create and train the initial svm classifier:
clf = svm.SVC(C=100000, kernel='rbf', probability=True,
              class_weight='balanced')
clf.fit(X_iter, y_iter)

print("INITIAL CLASSIFIER TRAINED")

# Model Accuracy (calculated on the training set):
print("Accuracy:", metrics.accuracy_score(y_iter, clf.predict(X_iter)))

# Update the labeled set with active learning and re-train the classifier:
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

    if sum(etp[maxindex])/B < 0.1 or k == 50:
        break

    for x in range(B):
        q0 = [Xu_iter[maxindex[x], 0], Xu_iter[maxindex[x], 1]]
        v0 = [Xu_iter[maxindex[x], 2], Xu_iter[maxindex[x], 3]]

        X_iter = np.append(X_iter, [[Xu_iter[maxindex[x], 0], Xu_iter[maxindex[x], 1],
                                     Xu_iter[maxindex[x], 2], Xu_iter[maxindex[x], 3]]], axis=0)
        res = pid.compute_problem(np.array(q0), np.array(v0))
        y_iter = np.append(y_iter, res)

    Xu_iter = np.delete(Xu_iter, maxindex, axis=0)

    # Re-fit the model with the new selected X_iter:
    clf.fit(X_iter, y_iter)
    print("CLASSIFIER", k+1, "TRAINED")

    # Model Accuracy (calculated on the training set):
    print("Accuracy:", metrics.accuracy_score(y_iter, clf.predict(X_iter)))

    k += 1

print("Execution time: %s seconds" % (time.time() - start_time))
