import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
import time
from pendulum_class_fixedveldir import OCPpendulumINIT
import warnings
import random
import torch
import torch.nn as nn
from my_nn import NeuralNetRegression
import math
from sklearn import svm
from numpy.linalg import norm as norm

warnings.filterwarnings("ignore")

with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPpendulumINIT()
    
    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = - ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_dir = NeuralNetRegression(2, 100, 1).to(device)
    criterion_dir = nn.MSELoss()
    optimizer_dir = torch.optim.Adam(model_dir.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_dir, gamma=0.9)

    ls = np.linspace(q_max, q_min, ocp.N, endpoint=False)
    vel = np.full(ocp.N, v_min)
    x_guess = np.append([ls], [vel], axis=0).T
    
    res = ocp.compute_problem(np.array([q_min, 0.]), x_guess)

    X_save = np.empty((0,2))

    if res == 1:
        for f in range(0, ocp.N+1):
            current_val = ocp.ocp_solver.get(f, "x")[:2]

            if abs(current_val[0] - q_min) <= 1e-3:
                break

            X_save = np.append(X_save,[current_val], axis = 0)

    ls = np.linspace(q_min, q_max, ocp.N, endpoint=False)
    vel = np.full(ocp.N, v_max)
    x_guess = np.append([ls], [vel], axis=0).T
         
    res = ocp.compute_problem(np.array([q_max, 0.]), x_guess)

    if res == 1:
        for f in range(0, ocp.N+1):
            current_val = ocp.ocp_solver.get(f, "x")[:2]

            if abs(current_val[0] - q_max) <= 1e-3:
                break

            X_save = np.append(X_save, [current_val], axis = 0)

    mean_dir, std_dir = torch.mean(torch.tensor(X_save[:,:1].tolist())).to(device).item(), torch.std(torch.tensor(X_save[:,:1].tolist())).to(device).item()

    X_save_dir = np.empty((X_save.shape[0],3))

    for i in range(X_save_dir.shape[0]):
        X_save_dir[i][0] = (X_save[i][0] - mean_dir) / std_dir
        vel_norm = abs(X_save[i][1])
        if vel_norm != 0:
            X_save_dir[i][1] = X_save[i][1] / vel_norm
        X_save_dir[i][2] = vel_norm

    X_train_dir = np.copy(X_save_dir)

    # np.save('data1_vbocp_10.npy', np.asarray(X_save))
            
    it = 1
    val = max(X_save_dir[:,2])

    beta = 0.95
    n_minibatch = 64
    B = int(X_save.shape[0]*100/n_minibatch) # number of iterations for 100 epoch
    it_max = B * 50

    training_evol = []

    # Train the model
    while val > 1e-3 and it < it_max:
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

            current_mean = sum(training_evol[-10:]) / 10
            previous_mean = sum(training_evol[-20:-10]) / 10
            if current_mean > previous_mean - 1e-4:
                scheduler.step()

    print("Execution time: %s seconds" % (time.time() - start_time))

    plt.figure()
    plt.plot(training_evol)

    torch.save(model_dir.state_dict(), 'model_1pendulum_dir_10')

    with torch.no_grad():
        X_iter_tensor = torch.Tensor(X_train_dir[:,:2]).to(device)
        y_iter_tensor = torch.Tensor(X_train_dir[:,2:]).to(device)
        outputs = model_dir(X_iter_tensor)
        print('RMSE train data: ', torch.sqrt(criterion_dir(outputs, y_iter_tensor))) 
            
        plt.figure(figsize=(6, 4))
        plt.plot(
            X_save[:,0], X_save[:,1], "ko", markersize=2
        )
        h = 0.01
        xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))
        inp = np.c_[xx.ravel(), yy.ravel(), yy.ravel()]
        for i in range(inp.shape[0]):
            inp[i][0] = (inp[i][0] - mean_dir) / std_dir
            vel_norm = abs(inp[i][1])
            if vel_norm != 0:
                inp[i][1] = inp[i][1] / vel_norm
            inp[i][2] = vel_norm
        out = (model_dir(torch.from_numpy(inp[:,:2].astype(np.float32)).to(device))).cpu().numpy() 
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
        plt.ylabel('$\dot{q}$')
        plt.xlabel('$q$')
        plt.subplots_adjust(left=0.14)
        plt.grid()
        plt.title("Classifier")

    print("Execution time: %s seconds" % (time.time() - start_time))

# X_test = np.load('data1_test_10.npy')
# X_test_dir = np.empty((X_test.shape[0],3))
# for i in range(X_test_dir.shape[0]):
#     X_test_dir[i][0] = (X_test[i][0] - mean_dir) / std_dir
#     vel_norm = abs(X_test[i][1])
#     if vel_norm != 0:
#         X_test_dir[i][1] = X_test[i][1] / vel_norm
#     X_test_dir[i][2] = vel_norm 

# with torch.no_grad():
#     X_iter_tensor = torch.Tensor(X_test_dir[:,:2]).to(device)
#     y_iter_tensor = torch.Tensor(X_test_dir[:,2:]).to(device)
#     outputs = model_dir(X_iter_tensor).cpu().numpy()
#     print('RRMSE test data wrt VBOCP NN in %: ', math.sqrt(np.sum([((outputs[i] - X_test_dir[i,2])/X_test_dir[i,2])**2 for i in range(len(X_test_dir))])/len(X_test_dir))*100)
#     print('RMSE test data wrt VBOCP NN: ', torch.sqrt(criterion_dir(model_dir(X_iter_tensor), y_iter_tensor)))

plt.show()
