import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
import time
from pendulum_ocp_class_reverse import OCPpendulumRINIT
import warnings
import random
import torch
import torch.nn as nn
from my_nn import NeuralNet
import math
from sklearn import svm

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
    
    # Initialization of the SVM classifier:
    clf = svm.SVC(C=1e6, kernel='rbf')
    
    # Hyper-parameters for nn:
    input_size = 2
    hidden_size = 100
    output_size = 2
    learning_rate = 0.001
    loss_stop = 0.001  # nn training stopping condition
    it_max = 1e2

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    eps = 1e-1
    
    X_save = np.empty((4*(ocpr.N+1),3))

    ls = np.linspace(q_max, q_min, ocpr.N, endpoint=False)
    vel = np.full(ocpr.N, v_min)
    x_guess = np.append([ls], [vel], axis=0).T
    
    res = ocpr.compute_problem(np.array([q_min, 0.]), x_guess)

    if res == 1:
        for f in range(0, ocpr.N+1):
            current_val = ocpr.ocp_solver.get(f, "x")

            if abs(current_val[0] - q_min) <= 1e-3:
                break

            X_save[f] = np.append(current_val, [1])
            X_save[2*(ocpr.N+1) + f] = [X_save[f][0], X_save[f][1] - eps, 0]

    ls = np.linspace(q_min, q_max, ocpr.N, endpoint=False)
    vel = np.full(ocpr.N, v_max)
    x_guess = np.append([ls], [vel], axis=0).T
         
    res = ocpr.compute_problem(np.array([q_max, 0.]), x_guess)

    if res == 1:
        for f in range(0, ocpr.N+1):
            current_val = ocpr.ocp_solver.get(f, "x")

            if abs(current_val[0] - q_max) <= 1e-3:
                break

            X_save[ocpr.N+1 + f] = np.append(current_val, [1])
            X_save[3*(ocpr.N+1) + f] = [X_save[ocpr.N+1 + f][0], X_save[ocpr.N+1 + f][1] + eps, 0]
            
    it = 0
    val = 1

    # # Train the model
    # while val > loss_stop and it <= it_max:

    #     X_iter_tensor = torch.Tensor(X_save[:,:2])
    #     y_iter_tensor = torch.Tensor(X_save[:,2:])

    #     # Forward pass
    #     outputs = model(X_iter_tensor)
    #     loss = criterion(outputs, y_iter_tensor)

    #     # Backward and optimize
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()

    #     val = loss.item()
    #     it += 1
        
    clf.fit(X_save[:,:2], X_save[:,2])
            
    plt.figure()
    plt.scatter(
        X_save[:,0], X_save[:,1], c =X_save[:,2], marker=".", cmap=plt.cm.Paired
    )
    h = 0.01
    x_min, x_max = q_min-(q_max-q_min)/100, q_max+(q_max-q_min)/100
    y_min, y_max = v_min-(v_max-v_min)/100, v_max+(v_max-v_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #inp = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    #out = model(inp)
    #y_pred = np.argmax(out.detach().numpy(), axis=1)
    #Z = y_pred.reshape(xx.shape)
    #plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    out = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    out = out.reshape(xx.shape)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.xlabel("Initial position [rad]")
    plt.ylabel("Initial velocity [rad/s]")
    plt.grid(True)

    print("Execution time: %s seconds" % (time.time() - start_time))

#pr.print_stats(sort="cumtime")

plt.show()
