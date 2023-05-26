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

def plots_results(sol_steps):
    t = np.linspace(0, 1e-2*sol_steps, sol_steps + 1)
    th = np.linspace(0, 1e-2*(tot_steps-1), tot_steps)

    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)
    (line,) = plt.step(t, np.append([simU[0, 0]], simU[:sol_steps, 0]), label='Control trajectory')
    plt.ylabel("$u_1$ [Nm]")
    plt.hlines(ocp.Cmax, th[0], th[-1], linestyles="dashed", alpha=0.7, label='Control limits')
    plt.hlines(-ocp.Cmax, th[0], th[-1], linestyles="dashed", alpha=0.7)
    plt.ylim([-1.2 * ocp.Cmax, 1.2 * ocp.Cmax])
    plt.title("Controls")
    plt.legend(loc="best")
    plt.grid()
    plt.subplot(2, 1, 2)
    (line,) = plt.step(t, np.append([simU[0, 1]], simU[:sol_steps, 1]), label='Control trajectory')
    plt.ylabel("$u_2$ [Nm]")
    plt.xlabel("t [s]")
    plt.hlines(ocp.Cmax, th[0], th[-1], linestyles="dashed", alpha=0.7, label='Control limits')
    plt.hlines(-ocp.Cmax, th[0], th[-1], linestyles="dashed", alpha=0.7)
    plt.ylim([-1.2 * ocp.Cmax, 1.2 * ocp.Cmax])
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('fig_control.pdf')

    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)
    (line,) = plt.plot(t, simX[:sol_steps+1, 0], label='Position trajectory')
    plt.hlines(ocp.thetamax, th[0], th[-1], linestyles="dashed", alpha=0.7, label='Position limits')
    plt.hlines(ocp.thetamin, th[0], th[-1], linestyles="dashed", alpha=0.7)
    plt.hlines(q_ref[0], th[0], th[-1], linestyles="dashed",
            alpha=0.7, color='green', label='Position reference')
    #plt.ylim([1.2 * mpc.xmin[0], 1.2 * mpc.xmax[0]])
    plt.ylabel("$x_1$ [rad]")
    plt.title("Positions")
    plt.legend(loc="best")
    plt.grid()
    plt.subplot(2, 1, 2)
    (line,) = plt.plot(t, simX[:sol_steps+1, 1], label='Position trajectory')
    plt.hlines(ocp.thetamax, th[0], th[-1], linestyles="dashed", alpha=0.7, label='Position limits')
    plt.hlines(ocp.thetamin, th[0], th[-1], linestyles="dashed", alpha=0.7)
    plt.hlines(q_ref[1], th[0], th[-1], linestyles="dashed",
            alpha=0.7, color='green', label='Position reference')
    #plt.ylim([1.2 * mpc.xmin[1], 1.2 * mpc.xmax[1]])
    plt.ylabel("$x_2$ [rad]")
    plt.xlabel("t [s]")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('fig_positions.pdf')

    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)
    (line,) = plt.plot(t, simX[:sol_steps+1, 2], label='Velocity trajectory')
    inp = np.empty((sol_steps+1,4))
    for i in range(inp.shape[0]):
        vel_norm = norm(simX[i][2:4])
        for l in range(2):
            inp[i][l] = (simX[i][l] - mean_dir) / std_dir
            inp[i][l+2] = simX[i][l+2] / vel_norm
    max_vel = np.multiply(np.reshape(model_dir(torch.from_numpy(inp.astype(np.float32)).to(device)).detach().cpu().numpy()*(100-safety_margin)/100,(sol_steps+1,)),inp[:,2])
    (line,) = plt.plot(t, max_vel, label='Max safe velocity')
    plt.hlines(ocp.dthetamax, th[0], th[-1], linestyles="dashed", alpha=0.7, label='Velocity limits')
    plt.hlines(-ocp.dthetamax, th[0], th[-1], linestyles="dashed", alpha=0.7)
    #plt.hlines(mpc.ocp.cost.yref[2], th[0], th[-1], linestyles="dashed", alpha=0.7, color='green', label='Velocity reference')
    #plt.ylim([1.2 * mpc.xmin[2], 1.2 * mpc.xmax[2]])
    plt.ylabel("$\dot{x}_1$ [rad/s]")
    plt.title("Velocities")
    plt.legend(loc="best")
    plt.grid()
    plt.subplot(2, 1, 2)
    (line,) = plt.plot(t, simX[:sol_steps+1, 3], label='Velocity trajectory')
    inp = np.empty((sol_steps+1,4))
    for i in range(inp.shape[0]):
        vel_norm = norm(simX[i][2:4])
        for l in range(2):
            inp[i][l] = (simX[i][l] - mean_dir) / std_dir
            inp[i][l+2] = simX[i][l+2] / vel_norm
    max_vel = np.multiply(np.reshape(model_dir(torch.from_numpy(inp.astype(np.float32)).to(device)).detach().cpu().numpy()*(100-safety_margin)/100,(sol_steps+1,)),inp[:,3])
    (line,) = plt.plot(t, max_vel, label='Max safe velocity')
    plt.hlines(ocp.dthetamax, th[0], th[-1], linestyles="dashed", alpha=0.7, label='Velocity limits')
    plt.hlines(-ocp.dthetamax, th[0], th[-1], linestyles="dashed", alpha=0.7)
    #plt.hlines(mpc.ocp.cost.yref[3], th[0], th[-1], linestyles="dashed", alpha=0.7, color='green', label='Velocity reference')
    #plt.ylim([1.2 * mpc.xmin[3], 1.2 * mpc.xmax[3]])
    plt.ylabel("$\dot{x}_2$ [rad/s]")
    plt.xlabel("t [s]")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('fig_velocities.pdf')

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
        # print("iteration " + str(f))
        
        status = ocp.OCP_solve(simX[f], q_ref, x_sol_guess, u_sol_guess)

        if status != 0:
            print("acados returned status " + str(status) + " at iteration " + str(f))
            # print(ocp.ocp_solver.get_residuals())

            failed_iter += 1

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
        
        # if all(ocp.nn_decisionfunction(model_dir.parameters(), mean_dir, std_dir, safety_margin, ocp.ocp_solver.get(i, 'x')) < 0 for i in range(ocp.N+1)):
        #     print('THERE ARE NO MORE GUARANTEES OF RECURSIVE FEASIBILITY FROM ITERATION ' + str(f))

        sim.acados_integrator.set("u", simU[f])
        sim.acados_integrator.set("x", simX[f])
        status = sim.acados_integrator.solve()
        simX[f+1] = sim.acados_integrator.get("x")
        simU[f] = u_sol_guess[0]

        # plots_results(f)

    return f

start_time = time.time()

# Pytorch params:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pytorch device

model_dir = NeuralNetRegression(4, 300, 1).to(device)
model_dir.load_state_dict(torch.load('../model_2dof_vboc_10_300_0.5_2.4007833'))
mean_dir = torch.load('../mean_2dof_vboc_10_300')
std_dir = torch.load('../std_2dof_vboc_10_300')
safety_margin = 2.4007833

ocp = OCPdoublependulumINIT(True, model_dir.parameters(), mean_dir, std_dir, safety_margin)
sim = SYMdoublependulumINIT(True)

# Generate low-discrepancy unlabeled samples:
from scipy.stats import entropy, qmc
sampler = qmc.Halton(d=2, scramble=False)
sample = sampler.random(n=pow(10, 2))
q_max = ocp.thetamax
q_min = ocp.thetamin
l_bounds = [q_min, q_min]
u_bounds = [q_max, q_max]
data = qmc.scale(sample, l_bounds, u_bounds)

tot_steps = 100
q_ref = np.array([(ocp.thetamax+ocp.thetamin)/2, ocp.thetamax - 0.05])

cpu_num = 30
with Pool(cpu_num) as p:
    res_steps = np.array(p.map(simulate, range(data.shape[0])))

for i in range(ocp.N):
    ocp.ocp_solver.cost_set(i, "Zl", np.ones((1,)))
ocp.ocp_solver.cost_set(ocp.N, "Zl", 1e2*np.ones((1,)))

cpu_num = 30
with Pool(cpu_num) as p:
    res_steps_term = np.array(p.map(simulate, range(data.shape[0])))

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

print(better)
print(equal)
print(worse)
print(res_steps)
print(res_steps_term)

plt.show()