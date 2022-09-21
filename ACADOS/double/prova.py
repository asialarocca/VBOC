import cProfile
import numpy as np
import matplotlib.pyplot as plt
import time
from double_pendulum_ocp_class import OCPdoublependulumINIT, OCPdoublependulumNN
import warnings
from scipy.stats import entropy, qmc

warnings.filterwarnings("ignore")


with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPdoublependulumINIT()

    nx = ocp.nx
    nu = ocp.nu
    N = ocp.N
    ocp_solver = ocp.ocp_solver
    Tf = ocp.Tf

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = -ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin
    Cmax = ocp.Cmax
    
    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=2, scramble=False)
    sample = sampler.random(n=pow(100, 2))
    l_bounds = [q_min, q_min]
    u_bounds = [q_max, q_max]
    Xu_iter = qmc.scale(sample, l_bounds, u_bounds).tolist()
    
    U0 = 0
    U1=0
    
    # Training of an initial classifier:
    for n in range(len(Xu_iter)):
        q0 = Xu_iter[n]
        v0 = [0.0, 0.0]
        # Data testing:
        res = ocp.compute_problem(q0, v0)
        U = ocp_solver.get(0, "u")
        Ux = abs(U[0])
        if Ux>U0:
            U0 = Ux
        Ux = abs(U[1])
        if Ux>U1:
            U1 = Ux
    print(U0,U1)
