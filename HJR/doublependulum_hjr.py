import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import time
from doublependulum_hjr_class import OCPdoublependulum
import warnings
import math
import torch
import torch.nn as nn
from my_nn import NeuralNetCLS
import random
from multiprocessing import Pool

warnings.filterwarnings("ignore")

def data_generation(v):
    x0 = Xu_iter[v]
    out_pred = y_pred[v]

    state = None
    output = None

    if (x0[0] >= q_min and x0[0] <= q_max and x0[2] >= v_min and x0[2] <= v_max and x0[1] >= q_min and x0[1] <= q_max and 
       x0[3] >= v_min and x0[3] <= v_max and out_pred == 1):
        res = ocp.compute_problem(x0)
        if res == 1:
            state = x0

            if ocp.ocp_solver.get_cost() < 0.:
                output = [0, 1]
            else:
                output = [1, 0]
    else:
        state = x0
        output = [1, 0]

    return state, output

start_time = time.time()

# test 2023/04/26:
# - increase hidden neurons from 100 to 300
# - increase batch size from 15^4 to 20^4
# - increase max comp time from 0.5 h to 3 h 
# test 2023/04/27: 
# - decrease hidden neurons from 300 to 100
# - increase num it for training from 100 to 500
# - final RMSE 0.6, but algorithm did not converge yet
# - most time spent on NN training
# test 2023/04/27: 
# - move NN training to GPU (6x faster now)
# - remove time spent testing, which has grown 3x now
# - converged in 1 h with RMSE 0.6
# test 2023/04/27 evening:
# - NN loss thr increased to 1e-2
# - max NN training iters increased to 2e3
# - converged in 1h with RMSE 0.8
# test 2023/04/28:
# - NN loss thr decreased to 5e-3
# - more iterations to converge but same total time and same RMSE
# test 2023/04/28
# - num samples increased to 25^4

# Load testing data:
X_test = np.load('../data2_test.npy')

# Position and velocity bounds:
v_max = 10.
v_min = -10.
q_max = np.pi / 4 + np.pi
q_min = - np.pi / 4 + np.pi

# Hyper-parameters for nn:
input_size = 4
hidden_size = 100
output_size = 2
learning_rate = 1e-3

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and optimizer:
model = NeuralNetCLS(input_size, hidden_size, output_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

B = pow(25, 4)  # batch size 
max_time = 3600 # max execution time of the algorithm

loss_stop = 5e-3  # nn training stopping condition
beta = 0.95
n_minibatch = 4096
it_max = int(2e3 * B / n_minibatch)

print("Number of grid values: ", B)
print("Max execution time", max_time)
print("Number of neurons in hidden layer", hidden_size)

# Generate unlabeled and test dataset:
Xu_iter = np.random.uniform(low=[q_min-(q_max-q_min)/10, q_min-(q_max-q_min)/10, v_min-(v_max-v_min)/10, v_min-(v_max-v_min)/10], high=[q_max+(q_max-q_min)/10, q_max+(q_max-q_min)/10, v_max+(v_max-v_min)/10, v_max+(v_max-v_min)/10], size=(B,4))
Xu_test = np.random.uniform(low=[q_min, q_min, v_min, v_min], high=[q_max, q_max, v_max, v_max], size=(B,4))

# Mean and standard:
Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32)).to(device)
mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

# Initial train dataset:
X_iter = Xu_iter
y_iter = np.array([[0, 1] if Xu_iter[n,0] >= q_min and Xu_iter[n,0] <= q_max and Xu_iter[n,2] >= v_min and Xu_iter[n,2] <= v_max and Xu_iter[n,1] >= q_min and Xu_iter[n,1] <= q_max and Xu_iter[n,3] >= v_min and Xu_iter[n,3] <= v_max else [1, 0] for n in range(Xu_iter.shape[0])])

print("Start NN training")
time_training_start = time.time()
it = 0
val = 1

# Train the model:
while val > loss_stop and it < it_max:

    ind = random.sample(range(X_iter.shape[0]), n_minibatch)

    X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32)).to(device)
    y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32)).to(device)
    X_iter_tensor = (X_iter_tensor - mean) / std

    # Forward pass
    outputs = model(X_iter_tensor)
    loss = criterion(outputs, y_iter_tensor)

    # Backward and optimize
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    val = beta * val + (1 - beta) * loss.item()

    it += 1

print("Training finished") 
print("\t%d iter, %.1f s, loss %.4f (loss thr %.4f)"%(it, time.time()-time_training_start,val, loss_stop))

# Count misclassified points on training set:
X_iter_tensor = torch.from_numpy(X_iter.astype(np.float32)).to(device)
y_iter_tensor = torch.from_numpy(y_iter.astype(np.float32)).to(device)
X_iter_tensor = (X_iter_tensor - mean) / std
outputs = model(X_iter_tensor)
num_correct = torch.count_nonzero(((torch.sign(outputs[:,0])+1)/2)==y_iter_tensor[:,0])
print("Percentage of correctly classified training points: %.2f"%(1e2*num_correct/outputs.shape[0]))

# Compute number of positively classifier test samples:
pos_old = Xu_iter.shape[0]
y_test_pred = np.argmax(model((torch.from_numpy(Xu_test.astype(np.float32)).to(device) - mean) / std).detach().cpu().numpy(), axis=1)
pos_new = np.sum([1 for i in range(Xu_test.shape[0]) if y_test_pred[i]==1])/Xu_test.shape[0]

# Compute RMSE wrt current model:
output_hjr_test = np.argmax(model((torch.Tensor(X_test).to(device) - mean) / std).cpu().detach().numpy(), axis=1)
norm_error = np.empty((len(X_test),))
for i in range(len(X_test)):
    vel_norm = norm([X_test[i][2],X_test[i][3]])
    v0 = X_test[i][2]
    v1 = X_test[i][3]

    if output_hjr_test[i] == 0:
        out = 0
        while out == 0 and norm([v0,v1]) > 1e-2:
            v0 = v0 - 1e-2 * X_test[i][2]/vel_norm
            v1 = v1 - 1e-2 * X_test[i][3]/vel_norm
            out = np.argmax(model((torch.Tensor([[X_test[i][0], X_test[i][1], v0, v1]]).to(device) - mean) / std).cpu().detach().numpy(), axis=1)
    else:
        out = 1
        while out == 1 and norm([v0,v1]) > 1e-2:
            v0 = v0 + 1e-2 * X_test[i][2]/vel_norm
            v1 = v1 + 1e-2 * X_test[i][3]/vel_norm
            out = np.argmax(model((torch.Tensor([[X_test[i][0], X_test[i][1], v0, v1]]).to(device) - mean) / std).cpu().detach().numpy(), axis=1)

    norm_error[i] = vel_norm - norm([v0,v1])

rmse = np.array([math.sqrt(np.sum(np.power(norm_error,2))/len(norm_error))])
times = np.array([time.time() - start_time])

init_times = 0
test_times = 0
iterbreak = 0

# Iteratively update the set approx:
while (pos_new <= pos_old or iterbreak > 10) and iterbreak < 150 and time.time() - start_time - init_times - test_times < max_time:
    print("Starting iteration ", iterbreak)
    iterbreak = iterbreak + 1

    time_before = time.time()
    ocp = OCPdoublependulum(mean.item(), std.item(), model.parameters())
    init_times = init_times + time.time() - time_before
    # print("OCP generated in %.1f s"%(time.time()-time_before))

    time_OCP_start = time.time()
    Xu_iter = np.random.uniform(low=[q_min, q_min, v_min-(v_max-v_min)/5, v_min-(v_max-v_min)/5], high=[q_max, q_max, v_max+(v_max-v_min)/5, v_max+(v_max-v_min)/5], size=(B,4))
    
    # compute NN prediction:
    inp = (torch.from_numpy(Xu_iter.astype(np.float32)).to(device) - mean) / std
    out = model(inp)
    y_pred = np.argmax(out.detach().cpu().numpy(), axis=1)
    # Xu_iter = np.insert(Xu_iter,4,y_pred,axis=1)
    # Xu_iter = np.reshape(Xu_iter,(B,5))

    # Data testing:
    with Pool(30) as p:
        temp = p.map(data_generation, range(Xu_iter.shape[0]))
    print("Finished solving OCPs after %.1f s"%(time.time() - time_OCP_start))

    x, y = zip(*temp)
    X_iter, y_iter = np.array([i for i in x if i is not None]), np.array([i for i in y if i is not None])

    it = 0
    val = 1

    # count misclassified points on training set before training:
    X_iter_tensor = torch.from_numpy(X_iter.astype(np.float32)).to(device)
    y_iter_tensor = torch.from_numpy(y_iter.astype(np.float32)).to(device)
    X_iter_tensor = (X_iter_tensor - mean) / std
    outputs = model(X_iter_tensor)
    num_correct = torch.count_nonzero(((torch.sign(outputs[:,0])+1)/2)==y_iter_tensor[:,0])
    print("Percentage of correctly classified training points before training: %.2f"%(1e2*num_correct/outputs.shape[0]))

    # Train the model:
    print("Start training the network")
    time_training_start = time.time()
    loss_hst = np.zeros(it_max)

    while val > loss_stop and it < it_max:

        ind = random.sample(range(X_iter.shape[0]), n_minibatch)

        X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32)).to(device)
        y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32)).to(device)
        X_iter_tensor = (X_iter_tensor - mean) / std

        # Forward pass
        outputs = model(X_iter_tensor)
        loss = criterion(outputs, y_iter_tensor)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        val = beta * val + (1 - beta) * loss.item()
        loss_hst[it] = loss.item()
        it += 1

    print("Training finished after %d iter, %.1f s, loss %.4f (loss thr %.4f)"%(it, time.time()-time_training_start,val, loss_stop))

    # Compute number of positively classified test samples (for stopping condition):
    pos_old = pos_new
    y_test_pred = np.argmax(model((torch.from_numpy(Xu_test.astype(np.float32)).to(device) - mean) / std).detach().cpu().numpy(), axis=1)
    pos_new = np.sum([1 for i in range(Xu_test.shape[0]) if y_test_pred[i]==1])/Xu_test.shape[0]

    del ocp

    times = np.append(times, [time.time() - start_time - init_times - test_times])

    start_time_testing = time.time()

    # count misclassified points on training set:
    X_iter_tensor = torch.from_numpy(X_iter.astype(np.float32)).to(device)
    y_iter_tensor = torch.from_numpy(y_iter.astype(np.float32)).to(device)
    X_iter_tensor = (X_iter_tensor - mean) / std
    outputs = model(X_iter_tensor)
    num_correct = torch.count_nonzero(((torch.sign(outputs[:,0])+1)/2)==y_iter_tensor[:,0])
    print("Percentage of correctly classified training points: %.2f"%(1e2*num_correct/outputs.shape[0]))

    # Compute RMSE:
    output_al_test = np.argmax(model((torch.Tensor(X_test).to(device) - mean) / std).cpu().detach().numpy(), axis=1)
    norm_error_al = np.empty((len(X_test),))
    for i in range(len(X_test)):
        vel_norm = norm([X_test[i][2],X_test[i][3]])
        v0 = X_test[i][2]
        v1 = X_test[i][3]

        if output_al_test[i] == 0:
            out = 0
            while out == 0 and norm([v0,v1]) > 1e-2:
                v0 = v0 - 1e-2 * X_test[i][2]/vel_norm
                v1 = v1 - 1e-2 * X_test[i][3]/vel_norm
                out = np.argmax(model((torch.Tensor([[X_test[i][0], X_test[i][1], v0, v1]]).to(device) - mean) / std).cpu().detach().numpy(), axis=1)
        else:
            out = 1
            while out == 1 and norm([v0,v1]) > 1e-2:
                v0 = v0 + 1e-2 * X_test[i][2]/vel_norm
                v1 = v1 + 1e-2 * X_test[i][3]/vel_norm
                out = np.argmax(model((torch.Tensor([[X_test[i][0], X_test[i][1], v0, v1]]).to(device) - mean) / std).cpu().detach().numpy(), axis=1)

        norm_error_al[i] = vel_norm - norm([v0,v1])

    rmse_it = math.sqrt(np.sum(np.power(norm_error_al,2))/len(norm_error_al))
    rmse = np.append(rmse, [rmse_it])
    plt.figure()
    plt.plot(times, rmse)
    plt.xlabel("Time [s]")
    plt.ylabel("RMSE")
    plt.grid()
    # plt.savefig("rmse_evol_hjb", format='png')
    # plt.close()

    time_testing = time.time() - start_time_testing
    test_times += time_testing
    print("RMSE:", rmse_it)
    print("Time spent testing", time_testing)
    print("Execution time so far:", time.time() - start_time - init_times - test_times)


print("Execution time:", time.time() - start_time - init_times - test_times, "(max time was %d)"%(max_time))
print("pos_new", pos_new, "pos_old", pos_old)
print("iterbreak", iterbreak) 

torch.save(model.state_dict(), 'model_2dof_hjr')
torch.save(mean, 'mean_2dof_hjr')
torch.save(std, 'std_2dof_hjr')

print('RMSE test data: ', math.sqrt(np.sum(np.power(norm_error_al,2))/len(norm_error_al))) 

np.save('times_2dof_hjr.npy', np.asarray(times))
np.save('rmse_2dof_hjr.npy', np.asarray(rmse))

plt.figure()
plt.plot(times, rmse)
plt.xlabel("Time [s]")
plt.ylabel("RMSE")
plt.grid()

# Show the results:
with torch.no_grad():
    # Plots:
    h = 0.02
    x_min, x_max = q_min-(q_max-q_min)/100, q_max+(q_max-q_min)/100
    y_min, y_max = v_min-(v_max-v_min)/100, v_max+(v_max-v_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max+h, h), np.arange(y_min, y_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()

    # Plot the results:
    plt.figure()
    inp = torch.from_numpy(
        np.float32(
            np.c_[
                (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                xrav,
                np.zeros(yrav.shape[0]),
                yrav,
            ]
        )
    ).to(device)
    inp = (inp - mean) / std
    out = model(inp)
    y_pred = np.argmax(out.cpu().numpy(), axis=1)
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.grid()
    plt.title("Second actuator")

    plt.figure()
    inp = torch.from_numpy(
        np.float32(
            np.c_[
                xrav,
                (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                yrav,
                np.zeros(yrav.shape[0]),
            ]
        )
    ).to(device)
    inp = (inp - mean) / std
    out = model(inp)
    y_pred = np.argmax(out.cpu().numpy(), axis=1)
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.grid()
    plt.title("First actuator")

plt.show()