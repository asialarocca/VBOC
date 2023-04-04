import numpy as np
import random 
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from numpy.linalg import norm as norm
import time
from doublependulum_class_fixedveldir import OCPdoublependulumRINIT, SYMdoublependulumINIT
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from my_nn import NeuralNet, NeuralNetRegression
import math
from multiprocessing import Pool

def testing(v):
    # Reset the number of steps used in the OCP:
    N = ocp.N
    ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
    ocp.ocp_solver.update_qp_solver_cond_N(N)

    # Time step duration:
    dt_sym = 1e-2

    # Initial velocity optimization direction (the cost is in the form p[0] * dtheta1 + p[1] * dtheta2 + p[2] * dt):
    ran1 = random.choice([-1, 1]) * random.random()
    ran2 = random.choice([-1, 1]) * random.random() 
    norm_weights = norm(np.array([ran1, ran2]))         
    p = np.array([ran1/norm_weights, ran2/norm_weights, 0.])

    # Bounds on the initial state:
    q_init_1 = q_min + random.random() * (q_max-q_min)
    q_init_2 = q_min + random.random() * (q_max-q_min)
    q_init_lb = np.array([q_init_1, q_init_2, v_min, v_min, dt_sym])
    q_init_ub = np.array([q_init_1, q_init_2, v_max, v_max, dt_sym])

    # Bounds on the final state:
    q_fin_lb = np.array([q_min, q_min, 0., 0., dt_sym])
    q_fin_ub = np.array([q_max, q_max, 0., 0., dt_sym])

    # Guess:
    x_sol_guess = np.full((N, 5), np.array([q_init_1, q_init_2, 0., 0., dt_sym]))
    u_sol_guess = np.full((N, 2), np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(q_init_1),ocp.g*ocp.l2*ocp.m2*math.sin(q_init_2)]))

    cost = 1e6

    # Iteratively solve the OCP with an increased number of time steps until the the solution does not change:
    while True:
        # Reset current iterate:
        ocp.ocp_solver.reset()

        # Set parameters, guesses and constraints:
        for i in range(N):
            ocp.ocp_solver.set(i, 'x', x_sol_guess[i])
            ocp.ocp_solver.set(i, 'u', u_sol_guess[i])
            ocp.ocp_solver.set(i, 'p', p)
            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 
            ocp.ocp_solver.constraints_set(i, 'lbu', np.array([-tau_max, -tau_max]))
            ocp.ocp_solver.constraints_set(i, 'ubu', np.array([tau_max, tau_max]))
            ocp.ocp_solver.constraints_set(i, 'C', np.array([[0., 0., 0., 0., 0.]]))
            ocp.ocp_solver.constraints_set(i, 'D', np.array([[0., 0.]]))
            ocp.ocp_solver.constraints_set(i, 'lg', np.array([0.]))
            ocp.ocp_solver.constraints_set(i, 'ug', np.array([0.]))

        ocp.ocp_solver.constraints_set(0, "lbx", q_init_lb)
        ocp.ocp_solver.constraints_set(0, "ubx", q_init_ub)
        ocp.ocp_solver.constraints_set(0, "C", np.array([[0., 0., p[1], -p[0], 0.]])) 

        ocp.ocp_solver.constraints_set(N, "lbx", q_fin_lb)
        ocp.ocp_solver.constraints_set(N, "ubx", q_fin_ub)
        ocp.ocp_solver.set(N, 'x', x_sol_guess[-1])
        ocp.ocp_solver.set(N, 'p', p)

        # Solve the OCP:
        status = ocp.ocp_solver.solve()

        if status == 0: # the solver has found a solution
            # Compare the current cost with the previous one:
            cost_new = ocp.ocp_solver.get_cost()
            if cost_new > float(f'{cost:.4f}') - 1e-4:
                break
            cost = cost_new

            # Update the guess with the current solution:
            x_sol_guess = np.empty((N+1,5))
            u_sol_guess = np.empty((N+1,2))
            for i in range(N):
                x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
                u_sol_guess[i] = ocp.ocp_solver.get(i, "u")
            x_sol_guess[N] = ocp.ocp_solver.get(N, "x")
            u_sol_guess[N] = np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x_sol_guess[N][0]),ocp.g*ocp.l2*ocp.m2*math.sin(x_sol_guess[N][1])])

            # Increase the number of time steps:
            N = N + 1
            ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
            ocp.ocp_solver.update_qp_solver_cond_N(N)
        else:
            # Reset the number of steps used in the OCP:
            N = ocp.N
            ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
            ocp.ocp_solver.update_qp_solver_cond_N(N)

            # Time step duration:
            dt_sym = 1e-2

            # Initial velocity optimization direction (the cost is in the form p[0] * dtheta1 + p[1] * dtheta2 + p[2] * dt):
            ran1 = ran1 + random.random() * random.choice([-1, 1]) * 0.01
            ran2 = ran2 + random.random() * random.choice([-1, 1]) * 0.01
            norm_weights = norm(np.array([ran1, ran2]))         
            p = np.array([ran1/norm_weights, ran2/norm_weights, 0.])
            
            # Bounds on the initial state:
            q_init_1 = q_init_1 + random.random() * random.choice([-1, 1]) * 0.01
            q_init_2 = q_init_2 + random.random() * random.choice([-1, 1]) * 0.01
            q_init_lb = np.array([q_init_1, q_init_2, v_min, v_min, dt_sym])
            q_init_ub = np.array([q_init_1, q_init_2, v_max, v_max, dt_sym])

            # Guess:
            x_sol_guess = np.full((N, 5), np.array([q_init_1, q_init_2, 0., 0., dt_sym]))
            u_sol_guess = np.full((N, 2), np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(q_init_1),ocp.g*ocp.l2*ocp.m2*math.sin(q_init_2)]))

            cost = 1e6

    return ocp.ocp_solver.get(0, "x")[:4]

start_time = time.time()

# Ocp initialization:
ocp = OCPdoublependulumRINIT()
sim = SYMdoublependulumINIT()

# Position, velocity and torque bounds:
v_max = ocp.dthetamax
v_min = - ocp.dthetamax
q_max = ocp.thetamax
q_min = ocp.thetamin
tau_max = ocp.Cmax

# Pytorch device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Unviable data generation parameter:
eps = 1e-3

min_time = 1 # set to 1 to also solve a minimum time problem to improve the solution

cpu_num = 30

num_prob = 30000

# Data generation:
with Pool(cpu_num) as p:
    temp = p.map(testing, range(num_prob))

print("Execution time: %s seconds" % (time.time() - start_time))

X_save = np.array(temp)

np.save('data_reverse_5_x0.npy', np.asarray(X_save))
# X_save = np.load('data_reverse_5_x0.npy')

# for k in range(4):
#     solved_tot = [i[1:] for i in X_solved if round(i[0]) == k]
#     failed_tot  = [i[1:] for i in X_failed if round(i[0]) == k]
#     if len(solved_tot)+len(failed_tot) != 0:
#         ratio = round(len(solved_tot)/(len(solved_tot)+len(failed_tot)),3)
#     else:
#         ratio = None
#     if k == 0:
#         print('first joint max velocity')
#     if k == 1:
#         print('second joint max velocity')
#     if k == 2:
#         print('first joint min velocity')
#     if k == 3:
#         print('second joint min velocity')
#     print('solved/total:', ratio)

#     num_setpoints = 11

#     pos_setpoints = np.linspace(q_min,q_max,num=num_setpoints)
#     ang_setpoints = np.linspace(-np.pi/2,np.pi/2,num=num_setpoints)
#     solved = np.empty((num_setpoints-1,num_setpoints-1))
#     failed = np.empty((num_setpoints-1,num_setpoints-1))
#     ratios = np.empty((num_setpoints-1,num_setpoints-1))
#     for j in range(num_setpoints-1):
#         for l in range(num_setpoints-1):
#             solved[j,l] = sum([1 for i in solved_tot if i[2] < pos_setpoints[j+1] and i[2] >= pos_setpoints[j] and math.atan(i[1]/i[0]) < ang_setpoints[l+1] and math.atan(i[1]/i[0]) > ang_setpoints[l]])
#             failed[j,l] = sum([1 for i in failed_tot if i[2] < pos_setpoints[j+1] and i[2] >= pos_setpoints[j] and math.atan(i[1]/i[0]) < ang_setpoints[l+1] and math.atan(i[1]/i[0]) > ang_setpoints[l]])
#             if solved[j,l]+failed[j,l] != 0:
#                 ratios[j,l] = round(solved[j,l]/(solved[j,l]+failed[j,l]),3)
#             else:
#                 ratios[j,l] = None
#         print(solved[j], failed[j], ratios[j])

#     print('')

plt.figure()
plt.scatter(X_save[:,0],X_save[:,1],s=0.1)
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("OCP dataset positions")

plt.figure()
plt.scatter(X_save[:,2],X_save[:,3],s=0.1)
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("OCP dataset velocities")

model_dir = NeuralNetRegression(4, 400, 1).to(device)
criterion_dir = nn.MSELoss()
optimizer_dir = torch.optim.Adam(model_dir.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_dir, gamma=0.98)

# X_save = np.array([X_save[i] for i in range(len(X_save)) if abs(X_save[i][2]) > 1e-3 and abs(X_save[i][3]) > 1e-3])

mean_dir, std_dir = torch.mean(torch.tensor(X_save[:,:2].tolist())).to(device).item(), torch.std(torch.tensor(X_save[:,:2].tolist())).to(device).item()

X_save_dir = np.empty((X_save.shape[0],5))

for i in range(X_save_dir.shape[0]):
    X_save_dir[i][0] = (X_save[i][0] - mean_dir) / std_dir
    X_save_dir[i][1] = (X_save[i][1] - mean_dir) / std_dir
    vel_norm = norm([X_save[i][2], X_save[i][3]])
    if vel_norm != 0:
        X_save_dir[i][2] = X_save[i][2] / vel_norm
        X_save_dir[i][3] = X_save[i][3] / vel_norm
    X_save_dir[i][4] = vel_norm 

# std_out_dir = torch.std(torch.tensor(X_save_dir[:,4:].tolist())).to(device).item()
# for i in range(X_save_dir.shape[0]):
#     X_save_dir[i][4] = X_save_dir[i][4] / std_out_dir

# summ = sum([X_save_dir[i][4] for i in range(len(X_save_dir))])
# X_prob = [X_save_dir[i][4]/summ for i in range(len(X_save_dir))]

# ind = np.random.choice(range(len(X_save_dir)), size=int(len(X_save_dir)*0.8), p=X_prob)
# X_save_dir = np.array([X_save_dir[i] for i in ind])
X_train_dir = np.copy(X_save_dir)

# ind = random.sample(range(len(X_save_dir)), int(len(X_save_dir)*0.7))
# X_train_dir = np.array([X_save_dir[i] for i in ind])
# X_test_dir = np.array([X_save_dir[i] for i in range(len(X_save_dir)) if i not in ind])

# model_dir.load_state_dict(torch.load('model_2pendulum_dir_20'))

it = 1
val = max(X_save_dir[:,4])

beta = 0.95
n_minibatch = 512
B = int(X_save.shape[0]*100/n_minibatch) # number of iterations for 100 epoch
it_max = B * 100

training_evol = []

# Train the model
while val > 1e-4 and it < it_max:
    ind = random.sample(range(len(X_train_dir)), n_minibatch)

    X_iter_tensor = torch.Tensor([X_train_dir[i][:4] for i in ind]).to(device)
    y_iter_tensor = torch.Tensor([[X_train_dir[i][4]] for i in ind]).to(device)

    # Forward pass
    outputs = model_dir(X_iter_tensor)
    loss = criterion_dir(outputs, y_iter_tensor)

    # Backward and optimize
    loss.backward()
    optimizer_dir.step()
    optimizer_dir.zero_grad()

    val = beta * val + (1 - beta) * loss.item()
    it += 1

    if it % B == 0: 
        print(val)
        training_evol.append(val)

        # if it > it_max:
        #     current_mean = sum(training_evol[-50:]) / 50
        #     previous_mean = sum(training_evol[-100:-50]) / 50
        #     if current_mean > previous_mean - 1e-6:

        current_mean = sum(training_evol[-10:]) / 10
        previous_mean = sum(training_evol[-20:-10]) / 10
        if current_mean > previous_mean - 1e-6:
            scheduler.step()

plt.figure()
plt.plot(training_evol)

torch.save(model_dir.state_dict(), 'model_2pendulum_dir_5_x0')

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_train_dir[:,:4]).to(device)
    y_iter_tensor = torch.Tensor(X_train_dir[:,4:]).to(device)
    outputs = model_dir(X_iter_tensor)
    print('RMSE train data: ', torch.sqrt(criterion_dir(outputs, y_iter_tensor))) 

# with torch.no_grad():
#     X_iter_tensor = torch.Tensor(X_test_dir[:,:4]).to(device)
#     y_iter_tensor = torch.Tensor(X_test_dir[:,4:]).to(device)
#     outputs = model_dir(X_iter_tensor)
#     print('RMSE test data: ', torch.sqrt(criterion_dir(outputs, y_iter_tensor))) 

with torch.no_grad():
    # Plots:
    h = 0.02
    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max+h, h))
    xrav = xx.ravel()
    yrav = yy.ravel()

    # Plot the results:
    plt.figure()
    inp = np.c_[
                (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                xrav,
                np.zeros(yrav.shape[0]),
                yrav,
                np.empty(yrav.shape[0]),
            ]
    for i in range(inp.shape[0]):
        vel_norm = norm([inp[i][2],inp[i][3]])
        inp[i][0] = (inp[i][0] - mean_dir) / std_dir
        inp[i][1] = (inp[i][1] - mean_dir) / std_dir
        if vel_norm != 0:
            inp[i][2] = inp[i][2] / vel_norm
            inp[i][3] = inp[i][3] / vel_norm
        inp[i][4] = vel_norm
    out = (model_dir(torch.from_numpy(inp[:,:4].astype(np.float32)).to(device))).cpu().numpy() 
    y_pred = np.empty(out.shape)
    for i in range(len(out)):
        if inp[i][4] > out[i]:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    for i in range(X_save.shape[0]):
        if (
            norm(X_save[i][0] - (q_min + q_max) / 2) < 0.01
            and norm(X_save[i][2]) < 0.1
        ):
            xit.append(X_save[i][1])
            yit.append(X_save[i][3])
    plt.plot(
        xit,
        yit,
        "ko",
        markersize=2
    )
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.grid()
    plt.title("Second actuator")

    plt.figure()
    inp = np.c_[
                xrav,
                (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                yrav,
                np.zeros(yrav.shape[0]),
                np.empty(yrav.shape[0]),
            ]
    for i in range(inp.shape[0]):
        vel_norm = norm([inp[i][2],inp[i][3]])
        inp[i][0] = (inp[i][0] - mean_dir) / std_dir
        inp[i][1] = (inp[i][1] - mean_dir) / std_dir
        if vel_norm != 0:
            inp[i][2] = inp[i][2] / vel_norm
            inp[i][3] = inp[i][3] / vel_norm
        inp[i][4] = vel_norm
    out = (model_dir(torch.from_numpy(inp[:,:4].astype(np.float32)).to(device))).cpu().numpy() 
    y_pred = np.empty(out.shape)
    for i in range(len(out)):
        if inp[i][4] > out[i]:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    for i in range(X_save.shape[0]):
        if (
            norm(X_save[i][1] - (q_min + q_max) / 2) < 0.01
            and norm(X_save[i][3]) < 0.1
        ):
            xit.append(X_save[i][0])
            yit.append(X_save[i][2])
    plt.plot(
        xit,
        yit,
        "ko",
        markersize=2
    )
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.grid()
    plt.title("First actuator")

    # Plots:
    h = 0.05
    xx, yy = np.meshgrid(np.arange(v_min, v_max+h, h), np.arange(v_min, v_max+h, h))
    xrav = xx.ravel()
    yrav = yy.ravel()

    # for l in range(10):
    #     q1ran = q_min + random.random() * (q_max-q_min)
    #     q2ran = q_min + random.random() * (q_max-q_min)

    #     # Plot the results:
    #     plt.figure()
    #     inp = np.float32(
    #             np.c_[
    #                 q1ran * np.ones(xrav.shape[0]),
    #                 q2ran * np.ones(xrav.shape[0]),
    #                 xrav,
    #                 yrav,
    #                 np.empty(yrav.shape[0]),
    #             ]
    #         )
    #     for i in range(inp.shape[0]):
    #         vel_norm = norm([inp[i][2],inp[i][3]])
    #         inp[i][0] = (inp[i][0] - mean_dir) / std_dir
    #         inp[i][1] = (inp[i][1] - mean_dir) / std_dir
    #         if vel_norm != 0:
    #             inp[i][2] = inp[i][2] / vel_norm
    #             inp[i][3] = inp[i][3] / vel_norm
    #         inp[i][4] = vel_norm
    #     out = (model_dir(torch.Tensor(inp[:,:4]).to(device))).cpu().numpy() 
    #     y_pred = np.empty(out.shape)
    #     for i in range(len(out)):
    #         if inp[i][4] > out[i]:
    #             y_pred[i] = 0
    #         else:
    #             y_pred[i] = 1
    #     Z = y_pred.reshape(yy.shape)
    #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #     xit = []
    #     yit = []
    #     for i in range(X_save.shape[0]):
    #         if (
    #             norm(X_save[i][0] - q1ran) < 0.01
    #             and norm(X_save[i][1] - q2ran) < 0.01
    #         ):
    #             xit.append(X_save[i][2])
    #             yit.append(X_save[i][3])
    #     plt.plot(
    #         xit,
    #         yit,
    #         "ko",
    #         markersize=2
    #     )
    #     plt.xlim([v_min, v_max])
    #     plt.ylim([v_min, v_max])
    #     plt.grid()
    #     plt.title("q1="+str(q1ran)+" q2="+str(q2ran))

print("Execution time: %s seconds" % (time.time() - start_time))

plt.show()