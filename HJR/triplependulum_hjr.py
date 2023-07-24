import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import time
from triplependulum_hjr_class import OCPtriplependulum
import warnings
import math
import torch
import torch.nn as nn
from my_nn import NeuralNetCLS
from torch.utils.data import DataLoader
import random
from multiprocessing import Pool

warnings.filterwarnings("ignore")

def data_generation(v):
    x0 = Xu_iter[v]
    out_pred = y_pred[v]

    state = None
    output = None

    if all(x0[i] >= q_min and x0[i] <= q_max for i in range(3)) and all(x0[i+3] >= v_min and x0[i+3] <= v_max for i in range(3)) and out_pred == 1:
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

# Load testing data:
X_test = np.load('../data3_test.npy')

# Position and velocity bounds:
v_max = 10.
v_min = -10.
q_max = np.pi / 4 + np.pi
q_min = - np.pi / 4 + np.pi

# Hyper-parameters for nn:
input_size = 6
hidden_size = 100
output_size = 2
learning_rate = 1e-3

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and optimizer:
model = NeuralNetCLS(input_size, hidden_size, output_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

B = pow(12, 6)  # batch size 
max_time = 21600 # max execution time of the algorithm

loss_stop = 5e-3  # nn training stopping condition
beta = 0.95
n_minibatch = 4096
it_max = int(2e3 * B / n_minibatch)
n_minibatch_model = pow(2,15)

print("Number of grid values: ", B)
print("Max execution time", max_time)
print("Number of neurons in hidden layer", hidden_size)

# Generate unlabeled and test dataset:
Xu_iter = np.random.uniform(low=[q_min-(q_max-q_min)/10, q_min-(q_max-q_min)/10, q_min-(q_max-q_min)/10, v_min-(v_max-v_min)/10, v_min-(v_max-v_min)/10, v_min-(v_max-v_min)/10], high=[q_max+(q_max-q_min)/10, q_max+(q_max-q_min)/10, q_max+(q_max-q_min)/10, v_max+(v_max-v_min)/10, v_max+(v_max-v_min)/10, v_max+(v_max-v_min)/10], size=(B,6))
Xu_test = np.random.uniform(low=[q_min, q_min, q_min, v_min, v_min, v_min], high=[q_max, q_max, q_max, v_max, v_max, v_max], size=(B,6))

# Mean and standard:
Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32)).to(device)
mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

# Initial train dataset:
X_iter = Xu_iter
y_iter = np.array([[0, 1] if all(Xu_iter[n,i] >= q_min and Xu_iter[n,i] <= q_max for i in range(3)) and all(Xu_iter[n,i+3] >= v_min and Xu_iter[n,i+3] <= v_max for i in range(3)) else [1, 0] for n in range(Xu_iter.shape[0])])

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
print("\t%d iter, %.1f s, loss %.4f (loss thr %.4f)"%(it, time.time()-time_training_start, val, loss_stop))

# # count misclassified points on training set:
# X_iter_tensor = torch.from_numpy(X_iter.astype(np.float32)).to(device)
# y_iter_tensor = torch.from_numpy(y_iter.astype(np.float32)).to(device)
# X_iter_tensor = (X_iter_tensor - mean) / std
# outputs = model(X_iter_tensor)
# num_correct = torch.count_nonzero(((torch.sign(outputs[:,0])+1)/2)==y_iter_tensor[:,0])
# print("Percentage of correctly classified training points: %.2f"%(1e2*num_correct/outputs.shape[0]))

# Compute number of positively classifier test samples:
pos_old = Xu_iter.shape[0]

# y_test_pred = np.argmax(model((torch.from_numpy(Xu_test.astype(np.float32)).to(device) - mean) / std).detach().cpu().numpy(), axis=1)

y_test_pred = np.empty((len(Xu_test),))
with torch.no_grad():
    Xu_test_tensor = torch.from_numpy(Xu_test.astype(np.float32)).to(device)
    Xu_test_tensor = (Xu_test_tensor - mean) / std
    my_dataloader = DataLoader(Xu_test_tensor,batch_size=n_minibatch_model,shuffle=False)
    for (idx, batch) in enumerate(my_dataloader):
        out = model(batch).detach().cpu().numpy()
        if n_minibatch_model*(idx+1) > len(Xu_iter):
            y_test_pred[n_minibatch_model*idx:len(Xu_iter)] = np.argmax(out, axis=1)
        else:
            y_test_pred[n_minibatch_model*idx:n_minibatch_model*(idx+1)] = np.argmax(out, axis=1)

pos_new = np.sum([1 for i in range(Xu_test.shape[0]) if y_test_pred[i]==1])/Xu_test.shape[0]

# Compute RMSE wrt current model:
output_hjr_test = np.argmax(model((torch.Tensor(X_test).to(device) - mean) / std).cpu().detach().numpy(), axis=1)
norm_error = np.empty((len(X_test),))
for i in range(len(X_test)):
    vel_norm = norm([X_test[i][3],X_test[i][4],X_test[i][5]])
    v0 = X_test[i][3]
    v1 = X_test[i][4]
    v2 = X_test[i][5]

    if output_hjr_test[i] == 0:
        out = 0
        while out == 0 and norm([v0,v1,v2]) > 1e-2:
            v0 = v0 - 1e-2 * X_test[i][3]/vel_norm
            v1 = v1 - 1e-2 * X_test[i][4]/vel_norm
            v2 = v2 - 1e-2 * X_test[i][5]/vel_norm
            out = np.argmax(model((torch.Tensor([[X_test[i][0], X_test[i][1], X_test[i][2], v0, v1, v2]]).to(device) - mean) / std).cpu().detach().numpy(), axis=1)
    else:
        out = 1
        while out == 1 and norm([v0,v1,v2]) > 1e-2:
            v0 = v0 + 1e-2 * X_test[i][3]/vel_norm
            v1 = v1 + 1e-2 * X_test[i][4]/vel_norm
            v2 = v2 + 1e-2 * X_test[i][5]/vel_norm
            out = np.argmax(model((torch.Tensor([[X_test[i][0], X_test[i][1], X_test[i][2], v0, v1, v2]]).to(device) - mean) / std).cpu().detach().numpy(), axis=1)

    norm_error[i] = vel_norm - norm([v0,v1,v2])

rmse = np.array([math.sqrt(np.sum(np.power(norm_error,2))/len(norm_error))])
times = np.array([time.time() - start_time])

init_times = 0
test_times = 0
iterbreak = 0

torch.cuda.empty_cache()

# Iteratively update the set approx:
while (pos_new <= pos_old or iterbreak > 10) and iterbreak < 120 and time.time() - start_time - init_times - test_times < max_time:
    print("Starting iteration ", iterbreak)
    iterbreak = iterbreak + 1

    time_before = time.time()
    ocp = OCPtriplependulum(mean.item(), std.item(), model.parameters())
    init_times = init_times + time.time() - time_before
    # print("OCP generated in %.1f s"%(time.time()-time_before))

    time_OCP_start = time.time()
    Xu_iter = np.random.uniform(low=[q_min, q_min, q_min, v_min-(v_max-v_min)/5, v_min-(v_max-v_min)/5, v_min-(v_max-v_min)/5], high=[q_max, q_max, q_max, v_max+(v_max-v_min)/5, v_max+(v_max-v_min)/5, v_max+(v_max-v_min)/5], size=(B,6))
    
    # compute NN prediction:
    # inp = (torch.from_numpy(Xu_iter.astype(np.float32)).to(device) - mean) / std
    # out = model(inp)
    # y_pred = np.argmax(out.detach().cpu().numpy(), axis=1)

    y_pred = np.empty((len(Xu_iter),))
    with torch.no_grad():
        Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32)).to(device)
        Xu_iter_tensor = (Xu_iter_tensor - mean) / std
        my_dataloader = DataLoader(Xu_iter_tensor,batch_size=n_minibatch_model,shuffle=False)
        for (idx, batch) in enumerate(my_dataloader):
            out = model(batch).detach().cpu().numpy()
            if n_minibatch_model*(idx+1) > len(Xu_iter):
                y_pred[n_minibatch_model*idx:len(Xu_iter)] = np.argmax(out, axis=1)
            else:
                y_pred[n_minibatch_model*idx:n_minibatch_model*(idx+1)] = np.argmax(out, axis=1)

    # Data testing:
    with Pool(30) as p:
        temp = p.map(data_generation, range(Xu_iter.shape[0]))
    print("Finished solving OCPs after %.1f s"%(time.time() - time_OCP_start))

    x, y = zip(*temp)
    X_iter, y_iter = np.array([i for i in x if i is not None]), np.array([i for i in y if i is not None])

    it = 0
    val = 1

    # # count misclassified points on training set before training:
    # X_iter_tensor = torch.from_numpy(X_iter.astype(np.float32)).to(device)
    # y_iter_tensor = torch.from_numpy(y_iter.astype(np.float32)).to(device)
    # X_iter_tensor = (X_iter_tensor - mean) / std
    # outputs = model(X_iter_tensor)
    # num_correct = torch.count_nonzero(((torch.sign(outputs[:,0])+1)/2)==y_iter_tensor[:,0])
    # print("Percentage of correctly classified training points before training: %.2f"%(1e2*num_correct/outputs.shape[0]))

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
    # y_test_pred = np.argmax(model((torch.from_numpy(Xu_test.astype(np.float32)).to(device) - mean) / std).detach().cpu().numpy(), axis=1)
    # pos_new = np.sum([1 for i in range(Xu_test.shape[0]) if y_test_pred[i]==1])/Xu_test.shape[0]

    y_test_pred = np.empty((len(Xu_test),))
    with torch.no_grad():
        Xu_test_tensor = torch.from_numpy(Xu_test.astype(np.float32)).to(device)
        Xu_test_tensor = (Xu_test_tensor - mean) / std
        my_dataloader = DataLoader(Xu_test_tensor,batch_size=n_minibatch_model,shuffle=False)
        for (idx, batch) in enumerate(my_dataloader):
            out = model(batch).detach().cpu().numpy()
            if n_minibatch_model*(idx+1) > len(Xu_iter):
                y_test_pred[n_minibatch_model*idx:len(Xu_iter)] = np.argmax(out, axis=1)
            else:
                y_test_pred[n_minibatch_model*idx:n_minibatch_model*(idx+1)] = np.argmax(out, axis=1)

    pos_new = np.sum([1 for i in range(Xu_test.shape[0]) if y_test_pred[i]==1])/Xu_test.shape[0]

    del ocp

    times = np.append(times, [time.time() - start_time - init_times - test_times])

    start_time_testing = time.time()

    # # count misclassified points on training set:
    # X_iter_tensor = torch.from_numpy(X_iter.astype(np.float32)).to(device)
    # y_iter_tensor = torch.from_numpy(y_iter.astype(np.float32)).to(device)
    # X_iter_tensor = (X_iter_tensor - mean) / std
    # outputs = model(X_iter_tensor)
    # num_correct = torch.count_nonzero(((torch.sign(outputs[:,0])+1)/2)==y_iter_tensor[:,0])
    # print("Percentage of correctly classified training points: %.2f"%(1e2*num_correct/outputs.shape[0]))

    # Compute RMSE wrt current model:
    output_hjr_test = np.argmax(model((torch.Tensor(X_test).to(device) - mean) / std).cpu().detach().numpy(), axis=1)
    norm_error = np.empty((len(X_test),))
    for i in range(len(X_test)):
        vel_norm = norm([X_test[i][3],X_test[i][4],X_test[i][5]])
        v0 = X_test[i][3]
        v1 = X_test[i][4]
        v2 = X_test[i][5]

        if output_hjr_test[i] == 0:
            out = 0
            while out == 0 and norm([v0,v1,v2]) > 1e-2:
                v0 = v0 - 1e-2 * X_test[i][3]/vel_norm
                v1 = v1 - 1e-2 * X_test[i][4]/vel_norm
                v2 = v2 - 1e-2 * X_test[i][5]/vel_norm
                out = np.argmax(model((torch.Tensor([[X_test[i][0], X_test[i][1], X_test[i][2], v0, v1, v2]]).to(device) - mean) / std).cpu().detach().numpy(), axis=1)
        else:
            out = 1
            while out == 1 and norm([v0,v1,v2]) > 1e-2:
                v0 = v0 + 1e-2 * X_test[i][3]/vel_norm
                v1 = v1 + 1e-2 * X_test[i][4]/vel_norm
                v2 = v2 + 1e-2 * X_test[i][5]/vel_norm
                out = np.argmax(model((torch.Tensor([[X_test[i][0], X_test[i][1], X_test[i][2], v0, v1, v2]]).to(device) - mean) / std).cpu().detach().numpy(), axis=1)

        norm_error[i] = vel_norm - norm([v0,v1,v2])

    rmse_it = math.sqrt(np.sum(np.power(norm_error,2))/len(norm_error))
    rmse = np.append(rmse, [rmse_it])

    time_testing = time.time() - start_time_testing
    test_times += time_testing
    print("RMSE:", rmse_it)
    print("Time spent testing", time_testing)
    print("Execution time so far:", time.time() - start_time - init_times - test_times)

    torch.cuda.empty_cache()

print("Execution time:", time.time() - start_time - init_times - test_times, "(max time was %d)"%(max_time))
print("pos_new", pos_new, "pos_old", pos_old)
print("iterbreak", iterbreak) 

torch.save(model.state_dict(), 'model_3dof_hjr')
torch.save(mean, 'mean_3dof_hjr')
torch.save(std, 'std_3dof_hjr')

print('RMSE test data: ', math.sqrt(np.sum(np.power(norm_error,2))/len(norm_error))) 

np.save('times_3dof_hjr.npy', np.asarray(times))
np.save('rmse_3dof_hjr.npy', np.asarray(rmse))

plt.figure()
plt.plot(times, rmse)
plt.xlabel("Time [s]")
plt.ylabel("RMSE")
plt.grid()

plt.show()