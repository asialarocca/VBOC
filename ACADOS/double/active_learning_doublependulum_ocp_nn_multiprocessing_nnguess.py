import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
import time
from double_pendulum_ocp_class import OCPdoublependulumINIT
import random
import warnings
import torch
import torch.nn as nn
from my_nn import NeuralNet, NeuralNetGuess
from statistics import mean
from multiprocessing import Pool
from pstats import Stats
from torch.utils.data import TensorDataset, DataLoader
import math

warnings.filterwarnings("ignore")

def testing(s0):
    q0,v0 = s0[:2], s0[2:]

    # Data testing:
    res = ocp.compute_problem(q0, v0)

    simX = np.ndarray((ocp.N+1, ocp_dim))

    if res == 1:
        for i in range(ocp.N+1):
            simX[i, :] = ocp.ocp_solver.get(i, "x")
        sim = np.reshape(simX,(ocp.N+1)*ocp_dim,).tolist()
        return [q0[0], q0[1], v0[0], v0[1], 0, 1], sim

    elif res == 0:
        return [q0[0], q0[1], v0[0], v0[1], 1, 0], None

def testing_guess(s0):
    q0,v0 = s0[:2], s0[2:]

    # Data testing:
    res = ocp.compute_problem_nnguess(q0, v0, model_guess, mean, std)

    simX = np.ndarray((ocp.N+1, ocp_dim))

    if res == 1:
        for i in range(ocp.N+1):
            simX[i, :] = ocp.ocp_solver.get(i, "x")
        sim = np.reshape(simX,(ocp.N+1)*ocp_dim,).tolist()
        return [q0[0], q0[1], v0[0], v0[1], 0, 1], sim

    elif res == 0:
        return [q0[0], q0[1], v0[0], v0[1], 1, 0], None

with cProfile.Profile() as pr:

    start_time = time.time()

    cpu_num = 31

    # Ocp initialization:
    ocp = OCPdoublependulumINIT()

    ocp_dim = ocp.nx

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = -ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    # Hyper-parameters for nn:
    input_size = ocp_dim
    hidden_size = ocp_dim * 100
    output_size = 2
    learning_rate = 0.001

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model_guess = NeuralNetGuess(input_size, hidden_size, ocp_dim*ocp.N).to(device)

    # Active learning parameters:
    N_init = pow(5, ocp_dim)  # size of initial labeled set
    B = pow(10, ocp_dim)  # batch size
    etp_stop = 0.2  # active learning stopping condition
    loss_stop = 0.1  # nn training stopping condition
    beta = 0.8
    n_minibatch = 512
    n_minibatch_model = pow(50,2)
    it_max = int(100 * B / n_minibatch)
    gridp = 70

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion_g = nn.MSELoss()
    optimizer_g = torch.optim.Adam(model_guess.parameters(), lr=learning_rate)

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp_dim, scramble=False)
    sample = sampler.random(n=pow(gridp, ocp_dim))
    l_bounds = [q_min-(q_max-q_min)/20, q_min-(q_max-q_min)/20, v_min-(v_max-v_min)/20, v_min-(v_max-v_min)/20]
    u_bounds = [q_max+(q_max-q_min)/20, q_max+(q_max-q_min)/20, v_max+(v_max-v_min)/20, v_max+(v_max-v_min)/20]
    Xu_iter = qmc.scale(sample, l_bounds, u_bounds).tolist()

    Xu_iter_tensor = torch.Tensor(Xu_iter).to(device)
    mean, std = torch.mean(Xu_iter_tensor).to(device), torch.std(Xu_iter_tensor).to(device)
    
    print(mean, std)

    with Pool(cpu_num) as p:
        temp = list(p.map(testing, Xu_iter[:N_init]))

    X_iter, X_traj = zip(*temp)
    X_traj = [i for i in X_traj if i is not None]

    # Delete tested data from the unlabeled set:
    del Xu_iter[:N_init]

    print("INITIAL CLASSIFIER IN TRAINING")

    it = 0
    val = 1

    # Train the model
    while val > loss_stop and it <= it_max:

        ind = random.sample(range(len(X_iter)), n_minibatch)

        X_iter_tensor = torch.Tensor([X_iter[i][:4] for i in ind]).to(device)
        y_iter_tensor = torch.Tensor([X_iter[i][4:] for i in ind]).to(device)
        X_iter_tensor = (X_iter_tensor - mean) / std

        # Zero the gradients
        for param in model.parameters():
            param.grad = None

        # Forward pass
        outputs = model(X_iter_tensor)
        loss = criterion(outputs, y_iter_tensor)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        val = beta * val + (1 - beta) * loss.item()

        it += 1

    it = 0
    val = 1

    qt = n_minibatch 

    # Train the guess model
    while val > loss_stop and it <= it_max:
    
        if len(X_traj) < qt:
            qt = len(X_traj)
        ind = random.sample(range(len(X_traj)), qt)

        X_iter_tensor = torch.Tensor([X_traj[i][:4] for i in ind])
        y_iter_tensor = torch.Tensor([X_traj[i][4:] for i in ind])
        X_iter_tensor = (X_iter_tensor - mean) / std
        y_iter_tensor = (y_iter_tensor - mean) / std

        # Forward pass
        outputs = model_guess(X_iter_tensor)
        loss = criterion_g(outputs, y_iter_tensor)

        # Backward and optimize
        loss.backward()
        optimizer_g.step()
        optimizer_g.zero_grad()

        val = beta * val + (1 - beta) * loss.item()

        it += 1

    print("INITIAL CLASSIFIER TRAINED")

    # Active learning:
    k = 0  # iteration number
    etpmax = 1
    # performance_history = []

    sigmoid = nn.Sigmoid()

    while not (etpmax < etp_stop or len(Xu_iter) == 0):

        if len(Xu_iter) < B:
            B = len(Xu_iter)

        # with torch.no_grad():
        #     Xu_iter_tensor = torch.Tensor(Xu_iter).to(device)
        #     Xu_iter_tensor = (Xu_iter_tensor - mean) / std
        #     prob_xu = sigmoid(model(Xu_iter_tensor)).cpu()
        #     etp = entropy(prob_xu, axis=1)

        etp = np.empty((len(Xu_iter),))
    
        with torch.no_grad():
            Xu_iter_tensor = torch.Tensor(Xu_iter).to(device)
            Xu_iter_tensor = (Xu_iter_tensor - mean) / std
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
        # performance_history.append(etpmax)

        k += 1

        # Take and delete data to test from the unlabeled set:
        elems = [None] * B
        for i in range(B):
            elems[i] = Xu_iter[maxindex[i]]
            del Xu_iter[maxindex[i]]

        with Pool(cpu_num) as p:
            temp = p.map(testing_guess, elems)

        tmp1, tmp2 = zip(*temp)
        X_iter = X_iter + tmp1
        tp = [i for i in tmp2 if i is not None]
        qn = n_minibatch if len(X_traj) > n_minibatch else len(X_traj)
        ind = random.sample(range(len(X_traj)), qn)
        tkeep = [i for i in X_traj if i in ind]
        X_traj = tkeep + tp

        print("CLASSIFIER", k, "IN TRAINING")

        it = 0
        val = 1

        Bg = len(X_traj)

        # Train the model
        while val > loss_stop and it <= it_max:

            ind = random.sample(range(len(X_iter) - B), int(n_minibatch / 2))
            ind.extend(
                random.sample(
                    range(len(X_iter) - B, len(X_iter)),
                    int(n_minibatch / 2),
                )
            )

            X_iter_tensor = torch.Tensor([X_iter[i][:4] for i in ind]).to(device)
            y_iter_tensor = torch.Tensor([X_iter[i][4:] for i in ind]).to(device)
            X_iter_tensor = (X_iter_tensor - mean) / std

            # Zero the gradients
            for param in model.parameters():
                param.grad = None

            # Forward pass
            outputs = model(X_iter_tensor)
            loss = criterion(outputs, y_iter_tensor)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            val = beta * val + (1 - beta) * loss.item()

            it += 1
        
        it = 0
        val = 1

        qt = int(n_minibatch / 2)

        end_ind = len(X_traj)
        init_ind = len(X_traj) - pow(20,4)
        if init_ind < 0:
            init_ind = 0

        X_traj = X_traj[init_ind:end_ind]

        # Train the guess model
        while val > loss_stop and it <= it_max:

            # if len(X_traj) - Bg < qt:
            #     qt = len(X_traj) - Bg
            # ind = random.sample(range(len(X_traj) - Bg), qt)
            # if Bg < qt:
            #     qt = Bg
            # ind.extend(
            #     random.sample(
            #         range(len(X_traj) - Bg, len(X_traj)),
            #         qt,
            #     )
            # )

            if len(X_traj) < qt:
                qt = len(X_traj)
            ind = random.sample(range(len(X_traj)), qt)

            X_iter_tensor = torch.Tensor([X_traj[i][:4] for i in ind])
            y_iter_tensor = torch.Tensor([X_traj[i][4:] for i in ind])
            X_iter_tensor = (X_iter_tensor - mean) / std
            y_iter_tensor = (y_iter_tensor - mean) / std

            # Forward pass
            outputs = model_guess(X_iter_tensor)
            loss = criterion_g(outputs, y_iter_tensor)

            # Backward and optimize
            loss.backward()
            optimizer_g.step()
            optimizer_g.zero_grad()

            val = beta * val + (1 - beta) * loss.item()

            it += 1

        print("CLASSIFIER", k, "TRAINED")

        print("etpmax:", etpmax)
        
        torch.save(model.state_dict(), 'model_2pendulum')

    # model = NeuralNet(input_size, hidden_size, output_size).to(device)
    # model.load_state_dict(torch.load('model_2pendulum'))

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
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        xit = []
        yit = []
        cit = []
        for i in range(len(X_iter)):
            if (
                norm(X_iter[i][0] - (q_min + q_max) / 2) < 0.1
                and norm(X_iter[i][2]) < 0.1
            ):
                xit.append(X_iter[i][1])
                yit.append(X_iter[i][3])
                if X_iter[i][4] == 1:
                    cit.append(0)
                else:
                    cit.append(1)
        plt.scatter(
            xit,
            yit,
            c=cit,
            marker=".",
            alpha=0.5,
            cmap=plt.cm.Paired,
        )
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
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        xit = []
        yit = []
        cit = []
        for i in range(len(X_iter)):
            if (
                norm(X_iter[i][1] - (q_min + q_max) / 2) < 0.1
                and norm(X_iter[i][3]) < 0.1
            ):
                xit.append(X_iter[i][0])
                yit.append(X_iter[i][2])
                if X_iter[i][4] == 1:
                    cit.append(0)
                else:
                    cit.append(1)
        plt.scatter(
            xit,
            yit,
            c=cit,
            marker=".",
            alpha=0.5,
            cmap=plt.cm.Paired,
        )
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.grid()
        plt.title("First actuator")

    print("Execution time: %s seconds" % (time.time() - start_time))

# stats = Stats(pr)
# stats.sort_stats('cumtime').print_stats(20)

plt.show()
