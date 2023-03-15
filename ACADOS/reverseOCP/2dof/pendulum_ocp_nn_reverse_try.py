import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
import time
from pendulum_ocp_class_reverse_try import OCPpendulumRINIT
import warnings
import random
import torch
import torch.nn as nn
from my_nn_try import NeuralNet
import math
from sklearn import svm
from numpy.linalg import norm as norm

warnings.filterwarnings("ignore")

with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocpr = OCPpendulumRINIT()
    
    # Position and velocity bounds:
    v_max = ocpr.dthetamax
    v_min = -ocpr.dthetamax
    q_max = ocpr.thetamax
    q_min = ocpr.thetamin
    
    ocp_dim = ocpr.nx  # state space dimension

    device = torch.device("cpu") 
    
    model_dir = NeuralNet(2, 16, 1).to(device)
    criterion_dir = nn.MSELoss()
    optimizer_dir = torch.optim.Adam(model_dir.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_dir, gamma=0.98)

    ls = np.linspace(q_max, q_min, ocpr.N, endpoint=False)
    vel = np.full(ocpr.N, v_min)
    x_guess = np.append([ls], [vel], axis=0).T
    
    res = ocpr.compute_problem(np.array([q_min, 0.]), x_guess)

    X_save = np.empty((0,2))

    if res == 1:
        for f in range(0, ocpr.N+1):
            current_val = ocpr.ocp_solver.get(f, "x")

            if abs(current_val[0] - q_min) <= 1e-3:
                break

            X_save = np.append(X_save,[current_val], axis = 0)

    ls = np.linspace(q_min, q_max, ocpr.N, endpoint=False)
    vel = np.full(ocpr.N, v_max)
    x_guess = np.append([ls], [vel], axis=0).T
         
    res = ocpr.compute_problem(np.array([q_max, 0.]), x_guess)

    if res == 1:
        for f in range(0, ocpr.N+1):
            current_val = ocpr.ocp_solver.get(f, "x")

            if abs(current_val[0] - q_max) <= 1e-3:
                break

            X_save = np.append(X_save, [current_val], axis = 0)
    X_train_dir = np.empty((X_save.shape[0],3))

    mean_dir, std_dir = torch.mean(torch.tensor(X_save[:,:1].tolist())).to(device).item(), torch.std(torch.tensor(X_save[:,:1].tolist())).to(device).item()
    std_out_dir = v_max

    for i in range(X_train_dir.shape[0]):
        X_train_dir[i][0] = (X_save[i][0] - mean_dir) / std_dir
        vel_norm = abs(X_save[i][1])
        if vel_norm != 0:
            X_train_dir[i][1] = X_save[i][1] / vel_norm
        X_train_dir[i][2] = vel_norm / std_out_dir

        print(X_save[i], X_train_dir[i])
            
    it = 1
    val = 1.

    beta = 0.95
    n_minibatch = 32
    B = int(X_save.shape[0]*100/n_minibatch) # number of iterations for 100 epoch
    it_max = B * 100

    training_evol = []

    # Train the model
    while val > 1e-8:
        ind = random.sample(range(len(X_train_dir)), n_minibatch)

        X_iter_tensor = torch.Tensor([X_train_dir[i][:2] for i in ind]).to(device)
        y_iter_tensor = torch.Tensor([[X_train_dir[i][2]] for i in ind]).to(device)

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

            scheduler.step()

            if it > it_max:
                current_mean = sum(training_evol[-20:]) / 20
                previous_mean = sum(training_evol[-40:-20]) / 20
                if current_mean > previous_mean - 1e-6:
                    break

    with torch.no_grad():
        X_iter_tensor = torch.Tensor(X_train_dir[:,:2])
        y_iter_tensor = torch.Tensor(X_train_dir[:,2:])
        outputs = model_dir(X_iter_tensor)
        print('RMSE train data: ', torch.sqrt(criterion_dir(outputs, y_iter_tensor))*std_out_dir)
            
    plt.figure()
    plt.plot(
        X_save[:,0], X_save[:,1], "ko", markersize=2
    )
    h = 0.01
    x_min, x_max = q_min-(q_max-q_min)/10, q_max+(q_max-q_min)/10
    y_min, y_max = v_min-(v_max-v_min)/10, v_max+(v_max-v_min)/10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    inp = np.c_[xx.ravel(), yy.ravel(), yy.ravel()]
    for i in range(inp.shape[0]):
        inp[i][0] = (inp[i][0] - mean_dir) / std_dir
        vel_norm = abs(inp[i][1])
        if vel_norm != 0:
            inp[i][1] = inp[i][1] / vel_norm
        inp[i][2] = vel_norm / std_out_dir
    out = model_dir(torch.from_numpy(inp[:,:2].astype(np.float32)))
    y_pred = np.empty(out.shape)
    for i in range(len(out)):
        if inp[i][2] > out[i]:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.xlabel("Initial position [rad]")
    plt.ylabel("Initial velocity [rad/s]")
    plt.grid(True)

    print("Execution time: %s seconds" % (time.time() - start_time))

#pr.print_stats(sort="cumtime")

plt.show()
