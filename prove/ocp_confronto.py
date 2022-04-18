import numpy as np
from numpy import nan
from utils import plot_pendulum
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
# from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from numpy.linalg import norm as norm
import time
from pendulum_ocp_class import OCPpendulum
import warnings
warnings.filterwarnings("ignore")

ocp = OCPpendulum()

res = ocp.compute_problem(1.2, 2.3)

ocp.ocp_solver.print_statistics()

# get solution
simX = np.ndarray((11, 2))
simU = np.ndarray((10, 1))

for i in range(10):
    simX[i, :] = ocp.ocp_solver.get(i, "x")
    simU[i, :] = ocp.ocp_solver.get(i, "u")
simX[10, :] = ocp.ocp_solver.get(10, "x")

print(simX[9, 1])

plot_pendulum(np.linspace(0, 0.1, 11), 10, simU, simX, latexify=False)
