import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
import time
from doublependulum_class_fixedveldir import OCPdoublependulumINIT, SYMdoublependulumINIT
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from my_nn import NeuralNetRegression
import math
from multiprocessing import Pool
from scipy.stats import qmc
import random

def simulate(p):
    x0 = np.array([data[p,0], data[p,1], 1e-10, 1e-10])

    simX = np.ndarray((tot_steps + 1, ocp.ocp.dims.nx))
    simU = np.ndarray((tot_steps, ocp.ocp.dims.nu))
    simX[0] = np.copy(x0)

    failed_iter = -1

    tot_time = 0
    calls = 0

    # Guess:
    x_sol_guess = np.full((ocp.N+1, ocp.ocp.dims.nx), x0)
    u_sol_guess = np.full((ocp.N, ocp.ocp.dims.nu), np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x0[0]),ocp.g*ocp.l2*ocp.m2*math.sin(x0[1])]))

    for f in range(tot_steps):
       
        temp = time.time()
        status = ocp.OCP_solve(simX[f], q_ref, x_sol_guess, u_sol_guess)
        tot_time += time.time() - temp
        calls += 1

        if status != 0:

            if failed_iter >= ocp.N-1 or failed_iter < 0:
                break

            failed_iter += 1

            simU[f] = u_sol_guess[0]

            for i in range(ocp.N-1):
                x_sol_guess[i] = np.copy(x_sol_guess[i+1])
                u_sol_guess[i] = np.copy(u_sol_guess[i+1])

            x_sol_guess[ocp.N-1] = np.copy(x_sol_guess[ocp.N])
            u_sol_guess[ocp.N-1] = [ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x_sol_guess[ocp.N-1,0]),ocp.g*ocp.l2*ocp.m2*math.sin(x_sol_guess[ocp.N-1,1])]

        else:
            failed_iter = 0

            simU[f] = ocp.ocp_solver.get(0, "u")

            for i in range(ocp.N-1):
                x_sol_guess[i] = ocp.ocp_solver.get(i+1, "x")
                u_sol_guess[i] = ocp.ocp_solver.get(i+1, "u")

            x_sol_guess[ocp.N-1] = ocp.ocp_solver.get(ocp.N, "x")
            x_sol_guess[ocp.N] = np.copy(x_sol_guess[ocp.N-1])
            u_sol_guess[ocp.N-1] = [ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x_sol_guess[ocp.N-1,0]),ocp.g*ocp.l2*ocp.m2*math.sin(x_sol_guess[ocp.N-1,1])]

        noise_intensity = 1e-3
        noise = np.array([(ocp.thetamax-ocp.thetamin)*noise_intensity*random.uniform(-1, 1), (ocp.thetamax-ocp.thetamin)*2*noise_intensity*random.uniform(-1, 1), ocp.dthetamax*2*noise_intensity*random.uniform(-1, 1), ocp.dthetamax*2*noise_intensity*random.uniform(-1, 1)])

        sim.acados_integrator.set("u", simU[f])
        sim.acados_integrator.set("x", simX[f])
        status = sim.acados_integrator.solve()
        simX[f+1] = sim.acados_integrator.get("x") + noise
        simU[f] = u_sol_guess[0]

    tm = tot_time/calls

    return f, tm

start_time = time.time()

# Pytorch params:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pytorch device

model_dir = NeuralNetRegression(4, 300, 1).to(device)
model_dir.load_state_dict(torch.load('../../model_2dof_vboc_10_300_0.5_2.4007833'))
mean_dir = torch.load('../../mean_2dof_vboc_10_300')
std_dir = torch.load('../../std_2dof_vboc_10_300')
safety_margin = 2.4007833

ocp = OCPdoublependulumINIT(True, model_dir.parameters(), mean_dir, std_dir, safety_margin)
sim = SYMdoublependulumINIT(True)

# Generate low-discrepancy unlabeled samples:
sampler = qmc.Halton(d=2, scramble=False)
sample = sampler.random(n=pow(10, 2))
q_max = ocp.thetamax
q_min = ocp.thetamin
l_bounds = [q_min, q_min]
u_bounds = [q_max, q_max]
data = qmc.scale(sample, l_bounds, u_bounds)

tot_steps = 100
q_ref = np.array([(ocp.thetamax+ocp.thetamin)/2, ocp.thetamax - 0.05])

# MPC controller without terminal constraints:
cpu_num = 30
with Pool(cpu_num) as p:
    res = np.array(p.map(simulate, range(data.shape[0])))

res_steps, stats = zip(*res)

print('Mean solve time: ' + str(np.mean(stats)))

print(np.array(res_steps).astype(int))

np.save('res_steps_noconstr.npy', np.array(res_steps).astype(int))