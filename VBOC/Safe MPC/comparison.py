import numpy as np
import matplotlib.pyplot as plt

# Load data
data_dir = "data/"
data_no = np.load(data_dir + "results_no_constraint.npz")

use_data = 'receidinghard' # soft_term, soft_traj, hard_term, receidinghard, receidingsoft

if use_data == 'soft_term':
    data_vboc = np.load(data_dir + "results_softterm.npz")
if use_data == 'soft_traj':
    data_vboc = np.load(data_dir + "results_softtraj.npz")
if use_data == 'hard_term':
    data_vboc = np.load(data_dir + "results_hardterm.npz")
if use_data == 'receidinghard':
    data_vboc = np.load(data_dir + "results_receidinghard.npz")
if use_data == 'receidingsoft':
    data_vboc = np.load(data_dir + "results_receidingsoft.npz")

dt = data_no["dt"]
tot_time = data_no["tot_time"]
times_no = data_no["times"]
res_steps_no = data_no["res_steps"]

times_hard = data_vboc["times"]
res_steps_hard = data_vboc["res_steps_term"]
better = data_vboc["better"]
worse = data_vboc["worse"]
equal = data_vboc["equal"]

# Plot timing
plt.figure()
plt.plot(np.linspace(0, len(times_no), len(times_no)), times_no, label="naive MPC", color='red')
plt.plot(np.linspace(0, len(times_hard), len(times_hard)), times_hard, label="MPC + VBOC", color='green')
plt.plot(np.linspace(0, len(times_no), len(times_no)), np.ones(len(times_no)) * np.quantile(times_no, 0.9),
         label="90% quantile naive MPC", color='fuchsia', linestyle='--')
plt.plot(np.linspace(0, len(times_hard), len(times_hard)), np.ones(len(times_hard)) * np.quantile(times_hard, 0.9),
         label="90% quantile MPC + VBOC", color='DarkBlue', linestyle='--')
plt.xlabel('MPC Iteration')
plt.ylabel('Solve time [s]')
plt.legend()
plt.title("Solve time comparison")
plt.savefig(data_dir + 'times_' + use_data + '.png')

# Barchart that states when the hard terminal constraint is better, equal or worse than the naive MPC
plt.figure()
plt.bar(["Better", "Equal", "Worse"], [better, equal, worse], color=["green", "blue", "red"])
plt.ylabel("Number")
plt.title("Comparison between MPC + VBOC and naive MPC")
plt.savefig(data_dir + 'numbers_' + use_data + '.png')

# Directly compare the number of iteration taken by the two MPCs before the first infeasible solution
plt.figure()
total = np.linspace(1, 100, 100)
plt.plot(total, res_steps_no, label="naive MPC", color='red')
plt.plot(total, np.ones(len(total)) * np.mean(res_steps_no), label="mean naive MPC", color='fuchsia', linestyle='--')
plt.plot(total, res_steps_hard, label="MPC + VBOC", color='green')
plt.plot(total, np.ones(len(total)) * np.mean(res_steps_hard), label="mean MPC + VBOC", color='DarkBlue', linestyle='--')
plt.title("Comparison between MPC + VBOC and naive MPC")
plt.xlabel("Number of problems")
plt.ylabel("Number of iterations")
plt.legend()
plt.savefig(data_dir + 'iterations_' + use_data + '.png')
