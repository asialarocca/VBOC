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

def simulate(p):
    x0 = np.array([data[p,0], data[p,1], 1e-10, 1e-10])

    simX = np.ndarray((tot_steps + 1, ocp.ocp.dims.nx))
    simU = np.ndarray((tot_steps, ocp.ocp.dims.nu))
    simX[0] = np.copy(x0)

    failed_iter = False

    # Guess:
    x_sol_guess = np.full((ocp.N+1, ocp.ocp.dims.nx), x0)
    u_sol_guess = np.full((ocp.N, ocp.ocp.dims.nu), np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x0[0]),ocp.g*ocp.l2*ocp.m2*math.sin(x0[1])]))

    for f in range(tot_steps):
       
        status = ocp.OCP_solve(simX[f], q_ref, x_sol_guess, u_sol_guess)

        if status != 0:
            failed_iter += 1

            print(status)
            print(ocp.ocp_solver.get_stats('residuals'))

            if failed_iter >= ocp.N:
                break

            simU[f] = u_sol_guess[0]

            for i in range(ocp.N-1):
                x_sol_guess[i] = x_sol_guess[i+1]
                u_sol_guess[i] = u_sol_guess[i+1]

            x_sol_guess[ocp.N-1] = x_sol_guess[ocp.N]
            x_sol_guess[ocp.N] = [q_ref[0], q_ref[1], 1e-10, 1e-10]
            u_sol_guess[ocp.N-1] = [ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x_sol_guess[ocp.N-1,0]),ocp.g*ocp.l2*ocp.m2*math.sin(x_sol_guess[ocp.N-1,1])]

        else:
            failed_iter = 0

            for i in range(ocp.N-1):
                x_sol_guess[i] = ocp.ocp_solver.get(i+1, "x")
                u_sol_guess[i] = ocp.ocp_solver.get(i+1, "u")

            x_sol_guess[ocp.N-1] = ocp.ocp_solver.get(ocp.N, "x")
            x_sol_guess[ocp.N] = [q_ref[0], q_ref[1], 1e-10, 1e-10]
            u_sol_guess[ocp.N-1] = [ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x_sol_guess[ocp.N-1,0]),ocp.g*ocp.l2*ocp.m2*math.sin(x_sol_guess[ocp.N-1,1])]

            simU[f] = ocp.ocp_solver.get(0, "u")

        sim.acados_integrator.set("u", simU[f])
        sim.acados_integrator.set("x", simX[f])
        status = sim.acados_integrator.solve()
        simX[f+1] = sim.acados_integrator.get("x")
        simU[f] = u_sol_guess[0]

    return f

start_time = time.time()

# Pytorch params:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pytorch device

model_dir = NeuralNetRegression(4, 300, 1).to(device)
model_dir.load_state_dict(torch.load('../../model_2dof_vboc_10_300_0.5_2.4007833'))
mean_dir = torch.load('../../mean_2dof_vboc_10_300')
std_dir = torch.load('../../std_2dof_vboc_10_300')

ocp = OCPdoublependulumINIT(True, list(model_dir.parameters()), mean_dir, std_dir)
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
    res_steps_term = np.array(p.map(simulate, range(data.shape[0])))

res_steps = np.load('../no_constraints/res_steps_noconstr.npy')

better = 0
equal = 0
worse = 0

plt.figure()
for i in range(res_steps.shape[0]):
    if res_steps_term[i]-res_steps[i]>0:
        plt.plot(data[i,0], data[i,1],color='darkblue',marker="*",markersize=9,zorder=1,linestyle="None")
        better += 1
    elif res_steps_term[i]-res_steps[i]==0:
        plt.plot(data[i,0], data[i,1],color='darkgrey',marker=".",markersize=7,zorder=1,linestyle="None")
        equal += 1
    else:
        plt.plot(data[i,0], data[i,1],color='darkred',marker="X",markersize=7,zorder=1,linestyle="None")
        worse += 1
plt.title('MPC standard vs MPC with hard term constraints')

print('MPC standard vs MPC with hard term constraints')
print('Percentage of initial states in which the MPC+VBOC behaves better: ' + str(better))
print('Percentage of initial states in which the MPC+VBOC behaves equal: ' + str(equal))
print('Percentage of initial states in which the MPC+VBOC behaves worse: ' + str(worse))

print(res_steps_term)

np.save('res_steps_hardtermconstr.npy', res_steps_term)

plt.show()