import numpy as np
from scipy.stats import entropy
import time
from triple_pendulum_ocp_class import OCPtriplependulumINIT
import random
import math
import warnings
import torch
import torch.nn as nn
from my_nn import NeuralNet
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm


warnings.filterwarnings("ignore")

if __name__ == "__main__":

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPtriplependulumINIT()

    ocp_dim = ocp.nx

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = -ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    # Hyper-parameters for nn:
    input_size = ocp_dim
    hidden_size = ocp_dim * 50
    output_size = 2
    learning_rate = 0.001

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Active learning parameters:
    N_init = pow(10, ocp_dim)  # size of initial labeled set
    B = pow(10,ocp_dim)  # batch size
    etp_stop = 0.5  # active learning stopping condition
    loss_stop = 0.1  # nn training stopping condition
    beta = 0.8
    n_minibatch = pow(2,9)
    n_minibatch_model = pow(2,15)
    it_max = int(10*B / n_minibatch)
    num = 15

    # # Generate low-discrepancy unlabeled samples:
    # sampler = qmc.Halton(d=ocp_dim, scramble=False)
    # sample = sampler.random(n=pow(20, ocp_dim))
    # l_bounds = [q_min, q_min, v_min, v_min]
    # u_bounds = [q_max, q_max, v_max, v_max]
    # data = qmc.scale(sample, l_bounds, u_bounds)

    # Generate random samples:
    data_q1 = np.random.uniform(q_min, q_max, size=pow(num, ocp_dim))
    data_q2 = np.random.uniform(q_min, q_max, size=pow(num, ocp_dim))
    data_q3 = np.random.uniform(q_min, q_max, size=pow(num, ocp_dim))
    data_v1 = np.random.uniform(v_min, v_max, size=pow(num, ocp_dim))
    data_v2 = np.random.uniform(v_min, v_max, size=pow(num, ocp_dim))
    data_v3 = np.random.uniform(v_min, v_max, size=pow(num, ocp_dim))
    
    Xu_iter = np.transpose(np.array([data_q1, data_q2, data_q3, data_v1, data_v2, data_v3])).tolist()

    # Generate the initial set of labeled samples:
    n = pow(10, ocp_dim)
    X_iter = np.empty((n, ocp_dim))
    y_iter = np.full((n, 2), [1, 0])
    r = np.random.random(size=(n, ocp_dim - 1))
    k = np.random.randint(ocp_dim, size=(n, 1))
    j = np.random.randint(2, size=(n, 1))
    x = np.zeros((n, ocp_dim))
    for i in np.arange(n):
        x[i, np.arange(ocp_dim)[np.arange(ocp_dim) != k[i]]] = r[i, :]
        x[i, k[i]] = j[i]

    X_iter[:, 0] = x[:, 0] * (q_max + 0.1 - (q_min - 0.1)) + q_min - 0.1
    X_iter[:, 1] = x[:, 1] * (q_max + 0.1 - (q_min - 0.1)) + q_min - 0.1
    X_iter[:, 2] = x[:, 2] * (q_max + 0.1 - (q_min - 0.1)) + q_min - 0.1
    X_iter[:, 3] = x[:, 3] * (v_max + 0.5 - (v_min - 0.5)) + v_min - 0.5
    X_iter[:, 4] = x[:, 4] * (v_max + 0.5 - (v_min - 0.5)) + v_min - 0.5
    X_iter[:, 5] = x[:, 5] * (v_max + 0.5 - (v_min - 0.5)) + v_min - 0.5

    X_iter = X_iter.tolist()
    y_iter = y_iter.tolist()

    # # Generate the initial set of labeled samples:
    # X_iter = [[(q_max + q_min) / 2, (q_max + q_min) / 2, 0.0, 0.0]]
    # res = ocp.compute_problem([(q_max + q_min) / 2, (q_max + q_min) / 2], [0.0, 0.0])
    # if res == 1:
    #     y_iter = [[0, 1]]
    # else:
    #     y_iter = [[1, 0]]

    Xu_iter_tensor = torch.Tensor(Xu_iter).to(device)
    mean, std = torch.mean(Xu_iter_tensor).to(device), torch.std(Xu_iter_tensor).to(device)

    # Training of an initial classifier:
    for n in range(N_init):
        q0 = Xu_iter[n][:3]
        v0 = Xu_iter[n][3:]

        # Data testing:
        res = ocp.compute_problem(q0, v0)
        if res == 1:
            X_iter.append([q0[0], q0[1], q0[2], v0[0], v0[1], v0[2]])
            y_iter.append([0, 1])
            for f in range(1, ocp.N, int(ocp.N/3)):
                if v0[0] >= 0.1 and v0[1] >= 0.1 and v0[1] >= 0.1:
                    X_iter.append(ocp.ocp_solver.get(f,"x").tolist())
                    y_iter.append([0, 1])
        elif res == 0:
            X_iter.append([q0[0], q0[1], q0[2], v0[0], v0[1], v0[2]])
            y_iter.append([1, 0])
            

    # Delete tested data from the unlabeled set:
    del Xu_iter[:N_init]

    print("INITIAL CLASSIFIER IN TRAINING")

    it = 0
    val = 1

    # Train the model
    while val > loss_stop and it <= it_max:

        ind = random.sample(range(len(X_iter)), n_minibatch)

        X_iter_tensor = torch.Tensor([X_iter[i] for i in ind]).to(device)
        y_iter_tensor = torch.Tensor([y_iter[i] for i in ind]).to(device)
        X_iter_tensor = (X_iter_tensor - mean) / std

        # Forward pass
        outputs = model(X_iter_tensor).to(device)
        loss = criterion(outputs, y_iter_tensor)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        for param in model.parameters():
            param.grad = None

        val = beta * val + (1 - beta) * loss.item()

        it += 1

    print("INITIAL CLASSIFIER TRAINED")

    # Active learning:
    k = 0  # iteration number
    etpmax = 1
    performance_history = []

    sigmoid = nn.Sigmoid()

    while not (etpmax < etp_stop or len(Xu_iter) == 0):

        if len(Xu_iter) < B:
            B = len(Xu_iter)

        etp = np.empty((len(Xu_iter),))
    
        with torch.no_grad():
            Xu_iter_tensor = torch.Tensor(Xu_iter).to(device)
            Xu_iter_tensor = (Xu_iter_tensor - mean) / std
#            prob_xu = sigmoid(model(Xu_iter_tensor)).cpu()
#            etp = entropy(prob_xu, axis=1)
            my_dataloader = DataLoader(Xu_iter_tensor,batch_size=n_minibatch_model,shuffle=False)
            for (idx, batch) in enumerate(my_dataloader):
                if n_minibatch_model*(idx+1) > len(Xu_iter):
                    prob_xu = sigmoid(model(batch)).cpu()
                    etp[n_minibatch_model*idx:len(Xu_iter)] = entropy(prob_xu, axis=1)
                else:
                    prob_xu = sigmoid(model(batch)).cpu()
                    etp[n_minibatch_model*idx:n_minibatch_model*(idx+1)] = entropy(prob_xu, axis=1)
    
        maxindex = np.argpartition(etp, -B)[
            -B:
        ].tolist()  # indexes of the uncertain samples
        maxindex.sort(reverse=True)

        etpmax = max(etp[maxindex])  # max entropy used for the stopping condition
        performance_history.append(etpmax)

        k += 1

        # Add the B most uncertain samples to the labeled set:
        for x in range(B):
            x0 = Xu_iter[maxindex[x]]
            del Xu_iter[maxindex[x]]
            q0 = x0[:3]
            v0 = x0[3:]

            # Data testing:
            res = ocp.compute_problem(q0, v0)
            if res == 1:
                X_iter.append([q0[0], q0[1], q0[2], v0[0], v0[1], v0[2]])
                y_iter.append([0, 1])
            elif res == 0:
                X_iter.append([q0[0], q0[1], q0[2], v0[0], v0[1], v0[2]])
                y_iter.append([1, 0])
                
        print("CLASSIFIER", k, "IN TRAINING")

        it = 0
        val = 1

        # Train the model
        while val > loss_stop and it <= it_max:

            ind = random.sample(range(len(X_iter) - B), int(n_minibatch / 2))
            ind.extend(
                random.sample(
                    range(len(X_iter) - B, len(X_iter)),
                    int(n_minibatch / 2),
                )
            )

            X_iter_tensor = torch.Tensor([X_iter[i] for i in ind]).to(device)
            y_iter_tensor = torch.Tensor([y_iter[i] for i in ind]).to(device)
            X_iter_tensor = (X_iter_tensor - mean) / std

            # Forward pass
            outputs = model(X_iter_tensor).to(device)
            loss = criterion(outputs, y_iter_tensor)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            for param in model.parameters():
                param.grad = None

            val = beta * val + (1 - beta) * loss.item()

            it += 1

        print("CLASSIFIER", k, "TRAINED")

        print("etpmax:", etpmax)
        
        with open('etp.txt','w') as f:
            f.write('initial it'+ str(etpmax))

    torch.save(model.state_dict(), 'model_save')
    torch.save(optimizer.state_dict(), 'opt_save')

    with open('times.txt','w') as f:
        f.write(str(time.time() - start_time))

    np.save('X_iter.npy',np.asarray(X_iter))

    np.save('y_iter.npy',np.asarray(y_iter))

    np.save('Xu_iter.npy',np.asarray(Xu_iter))
            
    with open('mean.txt','w') as f:
        f.write(str(mean.item()))

    with open('std.txt','w') as f:
        f.write(str(std.item()))
        
#    # Plots:
#    h = 0.02
#    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
#    xrav = xx.ravel()
#    yrav = yy.ravel()
#
#    with torch.no_grad():
#        # Plot the results:
#        plt.figure()
#        inp = torch.from_numpy(
#            np.float32(
#                np.c_[(q_min + q_max) / 2 * np.ones(xrav.shape[0]),
#                      (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
#                      xrav,
#                      np.zeros(yrav.shape[0]),
#                      np.zeros(yrav.shape[0]),
#                      yrav,
#                      ]
#            )
#        ).to(device)
#        inp = (inp - mean) / std
#        out = model(inp).to(device)
#        y_pred = np.argmax(out.cpu().numpy(), axis=1)
#        Z = y_pred.reshape(xx.shape)
#        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
#        xit = []
#        yit = []
#        cit = []
#        for i in range(len(X_iter)):
#            if (
#                norm(X_iter[i][0] - (q_min + q_max) / 2) < 0.1
#                and norm(X_iter[i][1] - (q_min + q_max) / 2) < 0.1
#                and norm(X_iter[i][3]) < 0.1 and norm(X_iter[i][4]) < 0.1
#            ):
#                xit.append(X_iter[i][2])
#                yit.append(X_iter[i][5])
#                if y_iter[i] == [1, 0]:
#                    cit.append(0)
#                else:
#                    cit.append(1)
#        plt.scatter(
#            xit,
#            yit,
#            c=cit,
#            marker=".",
#            alpha=0.5,
#            cmap=plt.cm.Paired,
#        )
#        plt.xlim([q_min, q_max])
#        plt.ylim([v_min, v_max])
#        plt.grid()
#        plt.title("Third actuator")
#
#        plt.figure()
#        inp = torch.from_numpy(
#            np.float32(
#                np.c_[(q_min + q_max) / 2 * np.ones(xrav.shape[0]),
#                      xrav,
#                      (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
#                      np.zeros(yrav.shape[0]),
#                      yrav,
#                      np.zeros(yrav.shape[0]),
#                      ]
#            )
#        ).to(device)
#        inp = (inp - mean) / std
#        out = model(inp).to(device)
#        y_pred = np.argmax(out.cpu().numpy(), axis=1)
#        Z = y_pred.reshape(xx.shape)
#        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
#        xit = []
#        yit = []
#        cit = []
#        for i in range(len(X_iter)):
#            if (
#                norm(X_iter[i][0] - (q_min + q_max) / 2) < 0.1
#                and norm(X_iter[i][2] - (q_min + q_max) / 2) < 0.1
#                and norm(X_iter[i][3]) < 0.1 and norm(X_iter[i][5]) < 0.1
#            ):
#                xit.append(X_iter[i][1])
#                yit.append(X_iter[i][4])
#                if y_iter[i] == [1, 0]:
#                    cit.append(0)
#                else:
#                    cit.append(1)
#        plt.scatter(
#            xit,
#            yit,
#            c=cit,
#            marker=".",
#            alpha=0.5,
#            cmap=plt.cm.Paired,
#        )
#        plt.xlim([q_min, q_max])
#        plt.ylim([v_min, v_max])
#        plt.grid()
#        plt.title("Second actuator")
#
#        plt.figure()
#        inp = torch.from_numpy(
#            np.float32(
#                np.c_[xrav,
#                      (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
#                      (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
#                      yrav,
#                      np.zeros(yrav.shape[0]),
#                      np.zeros(yrav.shape[0]),
#                      ]
#            )
#        ).to(device)
#        inp = (inp - mean) / std
#        out = model(inp).to(device)
#        y_pred = np.argmax(out.cpu().numpy(), axis=1)
#        Z = y_pred.reshape(xx.shape)
#        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
#        xit = []
#        yit = []
#        cit = []
#        for i in range(len(X_iter)):
#            if (
#                norm(X_iter[i][1] - (q_min + q_max) / 2) < 0.1
#                and norm(X_iter[i][2] - (q_min + q_max) / 2) < 0.1
#                and norm(X_iter[i][4]) < 0.1 and norm(X_iter[i][5]) < 0.1
#            ):
#                xit.append(X_iter[i][0])
#                yit.append(X_iter[i][3])
#                if y_iter[i] == [1, 0]:
#                    cit.append(0)
#                else:
#                    cit.append(1)
#        plt.scatter(
#            xit,
#            yit,
#            c=cit,
#            marker=".",
#            alpha=0.5,
#            cmap=plt.cm.Paired,
#        )
#        plt.xlim([q_min, q_max])
#        plt.ylim([v_min, v_max])
#        plt.grid()
#        plt.title("First actuator")
#        plt.show()
        
    #etp_stop = 0.2  # active learning stopping condition
        
    for iteration in range(100):
        
        print('GENERATE NEW DATA')
        
        X_iter = X_iter[-100*B:]
        y_iter = y_iter[-100*B:]

        rad_q = (q_max - q_min) / num
        rad_v = (v_max - v_min) / num
    
        n_points = 1
    
        etp = np.empty((len(X_iter),))
                    
        with torch.no_grad():
            X_iter_tensor = torch.Tensor(X_iter).to(device)
            X_iter_tensor = (X_iter_tensor - mean) / std
#            prob_x = sigmoid(model(X_iter_tensor)).cpu()
#            etp=entropy(prob_x, axis=1)
            my_dataloader = DataLoader(X_iter_tensor,batch_size=n_minibatch_model,shuffle=False)
            for (idx, batch) in enumerate(my_dataloader):
                if n_minibatch_model*(idx+1) > len(X_iter):
                    prob_x = sigmoid(model(batch)).cpu()
                    etp[n_minibatch_model*idx:len(X_iter)] = entropy(prob_x, axis=1)
                else:
                    prob_x = sigmoid(model(batch)).cpu()
                    etp[n_minibatch_model*idx:n_minibatch_model*(idx+1)] = entropy(prob_x, axis=1)
    
        xu = np.array(X_iter)[etp > 0.5]
        xu = xu[-100*B:]
        
        #X_iter = np.array(X_iter)[etp > 1e-2].tolist()
        #y_iter = np.array(y_iter)[etp > 1e-2].tolist()
    
        Xu_it = np.empty((xu.shape[0], n_points, ocp_dim))
    
        # Generate other random samples:
        for i in range(xu.shape[0]):
            for n in range(n_points):
    
                # random angle
                alpha = math.pi * random.random()
                beta = math.pi * random.random()
                lamb = math.pi * random.random()
                theta = math.pi * random.random()
                gamma = 2 * math.pi * random.random()
    
                # random radius
                tmp = math.sqrt(random.random())
                r_x = rad_q * tmp
                r_y = rad_v * tmp
                # calculating coordinates
                x1 = r_x * math.cos(alpha) + xu[i, 0]
                x2 = r_x * math.sin(alpha) * math.cos(beta) + xu[i, 1]
                x3 = r_y * math.sin(alpha) * math.sin(beta) * math.cos(lamb) + xu[i, 2]
                x4 = r_y * math.sin(alpha) * math.sin(beta) * \
                    math.sin(lamb) * math.cos(theta) + xu[i, 3]
                x5 = r_y * math.sin(alpha) * math.sin(beta) * math.sin(lamb) * \
                    math.sin(theta) * math.cos(gamma) + xu[i, 4]
                x6 = r_y * math.sin(alpha) * math.sin(beta) * math.sin(lamb) * \
                    math.sin(theta) * math.sin(gamma) + xu[i, 5]
    
                Xu_it[i, n, :] = [x1, x2, x3, x4, x5, x6]
    
        Xu_it.shape = (xu.shape[0] * n_points, ocp_dim)
        Xu_iter = Xu_it.tolist()
    
        etpmax = 1
    
        while not (etpmax < etp_stop or len(Xu_iter) == 0):
    
            if len(Xu_iter) < B:
                B = len(Xu_iter)
    
            etp = np.empty((len(Xu_iter),))
    
            with torch.no_grad():
                Xu_iter_tensor = torch.Tensor(Xu_iter).to(device)
                Xu_iter_tensor = (Xu_iter_tensor - mean) / std
#                prob_xu = sigmoid(model(Xu_iter_tensor)).cpu()
#                etp = entropy(prob_xu, axis=1)
                
                my_dataloader = DataLoader(Xu_iter_tensor,batch_size=n_minibatch_model,shuffle=False)
                for (idx, batch) in enumerate(my_dataloader):
                    if n_minibatch_model*(idx+1) > len(Xu_iter):
                        prob_xu = sigmoid(model(batch)).cpu()
                        etp[n_minibatch_model*idx:len(Xu_iter)] = entropy(prob_xu, axis=1)
                    else:
                        prob_xu = sigmoid(model(batch)).cpu()
                        etp[n_minibatch_model*idx:n_minibatch_model*(idx+1)] = entropy(prob_xu, axis=1)
    
            maxindex = np.argpartition(etp, -B)[
                -B:
            ].tolist()  # indexes of the uncertain samples
            maxindex.sort(reverse=True)
    
            etpmax = max(etp[maxindex])  # max entropy used for the stopping condition
            performance_history.append(etpmax)
    
            k += 1
    
            # Add the B most uncertain samples to the labeled set:
            for x in range(B):
                x0 = Xu_iter[maxindex[x]]
                del Xu_iter[maxindex[x]]
                q0 = x0[:3]
                v0 = x0[3:]
                
                if x0[0] > ocp.thetamax or x0[0] < ocp.thetamin or x0[1] > ocp.thetamax or x0[1] < ocp.thetamin or x0[2] > ocp.thetamax or x0[2] < ocp.thetamin or abs(x0[0]) > ocp.dthetamax or abs(x0[1]) > ocp.dthetamax or abs(x0[2]) > ocp.dthetamax:
                    X_iter.append([q0[0], q0[1], q0[2], v0[0], v0[1], v0[2]])
                    y_iter.append([1, 0])
                else:
                    # Data testing:
                    res = ocp.compute_problem(q0, v0)
                    if res == 1:
                        X_iter.append([q0[0], q0[1], q0[2], v0[0], v0[1], v0[2]])
                        y_iter.append([0, 1])
                    elif res == 0:
                        X_iter.append([q0[0], q0[1], q0[2], v0[0], v0[1], v0[2]])
                        y_iter.append([1, 0])
    
            print("CLASSIFIER", k, "IN TRAINING")
    
            it = 0
            val = 1
    
            # Train the model
            while val > loss_stop and it <= it_max:
                
                ind = random.sample(range(len(X_iter) - B), int(n_minibatch / 2))
                
                ind.extend(
                    random.sample(
                        range(len(X_iter) - B, len(X_iter)),
                        int(n_minibatch / 2),
                    )
                )
    
                X_iter_tensor = torch.Tensor([X_iter[i] for i in ind]).to(device)
                y_iter_tensor = torch.Tensor([y_iter[i] for i in ind]).to(device)
                X_iter_tensor = (X_iter_tensor - mean) / std
    
                # Zero the gradients
                for param in model.parameters():
                    param.grad = None
    
                # Forward pass
                outputs = model(X_iter_tensor).to(device)
                loss = criterion(outputs, y_iter_tensor)
    
                # Backward and optimize
                loss.backward()
                optimizer.step()
    
                val = beta * val + (1 - beta) * loss.item()
    
                it += 1
    
            print("CLASSIFIER", k, "TRAINED")
    
            print("etpmax:", etpmax)
            
            with open('etp.txt','w') as f:
                f.write(str(iteration)+ ' it'+ str(etpmax))
            
    
        torch.save(model.state_dict(), 'model_save_np')
        torch.save(optimizer.state_dict(), 'opt_save_np')
    
        with open('times.txt','w') as f:
            f.write(str(time.time() - start_time))
    
        np.save('X_iter_np.npy',np.asarray(X_iter))
    
        np.save('y_iter_np.npy',np.asarray(y_iter))
    
        np.save('Xu_iter_np.npy',np.asarray(Xu_iter))
            
        with open('mean.txt','w') as f:
            f.write(str(mean.item()))
    
        with open('std.txt','w') as f:
            f.write(str(std.item()))


    print("Execution time: %s seconds" % (time.time() - start_time))
