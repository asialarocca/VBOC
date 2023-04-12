import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from numpy.linalg import norm
import time
from pendulum_ocp_class import OCPpendulum
import warnings
import math
from scipy.optimize import fsolve
import torch
import torch.nn as nn
from my_nn import NeuralNet
import random

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    start_time = time.time()

    # Position and velocity bounds:
    v_max = 10.
    v_min = -10.
    q_max = np.pi / 2
    q_min = 0.

    # Hyper-parameters for nn:
    input_size = 2
    hidden_size = 100
    output_size = 2
    learning_rate = 0.01

    # Device configuration
    device = torch.device("cpu")

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # Initialization of the SVM classifier:
    # clf = svm.SVC(
    #     C=1e4, kernel="rbf", class_weight='balanced'
    # )

    B = pow(100, 2)  # batch size

    loss_stop = 0.001  # nn training stopping condition
    beta = 0.99
    n_minibatch = 64
    it_max = 1e3 * B / n_minibatch

    Xu_iter = np.random.uniform(low=[q_min-(q_max-q_min)/10, v_min-(v_max-v_min)/10], high=[q_max+(q_max-q_min)/10, v_max+(v_max-v_min)/10], size=(B,2))

    Xu_test = np.random.uniform(low=[q_min, v_min], high=[q_max, v_max], size=(B,2))

    Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32))
    mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

    X_iter = Xu_iter
    y_iter = np.empty((0, 2))
    # y_iter = np.empty((0, 1))

    for n in range(Xu_iter.shape[0]):
        q0 = Xu_iter[n, 0]
        v0 = Xu_iter[n, 1]

        if q0>=q_min and q0<=q_max and v0>=v_min and v0<=v_max:
            y_iter = np.append(y_iter, [[0, 1]], axis=0)
            # y_iter = np.append(y_iter, 1)
        else:
            y_iter = np.append(y_iter, [[1, 0]], axis=0)
            # y_iter = np.append(y_iter, 0)

    it = 0
    val = 1

    # Train the model
    while val > loss_stop and it <= it_max:

        ind = random.sample(range(X_iter.shape[0]), n_minibatch)

        X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32))
        y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32))
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
        
    # clf.fit(X_iter, y_iter)
    
    h = 0.01
    xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))

    with torch.no_grad():
        # Plot the results:
        plt.figure()
        h = 0.01
        xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))
        inp = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
        inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        z = [
            0 if np.array_equal(y_iter[x], [1, 0]) else 1
            for x in range(y_iter.shape[0])
        ]
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        scatter = plt.scatter(
            X_iter[:, 0], X_iter[:, 1], c=z, marker=".", alpha=0.5, cmap=plt.cm.Paired
        )
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.xlabel("Initial position [rad]")
        plt.ylabel("Initial velocity [rad/s]")
        plt.title("Initialization Classifier")
        hand = scatter.legend_elements()[0]
        plt.legend(handles=hand, labels=("Non viable", "Viable"))
        plt.grid(True)

        # Plot of the decision function:
        plt.figure()
        y_plot = out.numpy()[:,0]
        y_plot = y_plot.reshape(xx.shape)
        levels = np.linspace(y_plot.min(), y_plot.max(), 20)
        plt.contourf(xx, yy, y_plot, levels=levels, cmap='plasma')
        this = plt.contour(xx, yy, y_plot, levels=levels, colors=('k',), linewidths=(1,))
        plt.clabel(this, fmt='%2.1f', colors='w', fontsize=11)
        #df = plt.contour(xx, yy, out, levels=[0], linewidths=(2,), colors=("k",))
        #plt.clabel(df, fmt='%2.1f', colors='w', fontsize=11)
        plt.xlabel('Initial position [rad]')
        plt.ylabel('Initial velocity [rad/s]')
        plt.title('Decision function')
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.grid()
        # plt.show()

    # # Plot the results:
    # h = 0.01
    # xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))
    # plt.figure(figsize=(6, 5))
    # out = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    # out = out.reshape(xx.shape)
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # plt.contour(xx, yy, out, levels=[0], linewidths=(2,), colors=("k",))
    # scatter = plt.scatter(
    #     X_iter[:, 0],
    #     X_iter[:, 1],
    #     c=y_iter,
    #     marker=".",
    #     alpha=0.5,
    #     cmap=plt.cm.Paired,
    # )
    # plt.xlim([0.0, np.pi / 2 - 0.01])
    # plt.ylim([-10.0, 10.0])
    # plt.xlabel("Initial position [rad]")
    # plt.ylabel("Initial velocity [rad/s]")
    # plt.title("Classifier")
    # hand = scatter.legend_elements()[0]
    # plt.legend(handles=hand, labels=("Non viable", "Viable"), loc='upper right')
    # plt.grid()

    # # Plot of the decision function:
    # plt.figure(figsize=(6, 5))
    # levels = np.linspace(out.min(), out.max(), 10)
    # plt.contourf(xx, yy, out, levels=levels, cmap='plasma')
    # this = plt.contour(xx, yy, out, levels=levels, colors=('k',), linewidths=(1,))
    # plt.clabel(this, fmt='%2.1f', colors='w', fontsize=11)
    # #df = plt.contour(xx, yy, out, levels=[0], linewidths=(2,), colors=("k",))
    # #plt.clabel(df, fmt='%2.1f', colors='w', fontsize=11)
    # plt.xlabel('Initial position [rad]')
    # plt.ylabel('Initial velocity [rad/s]')
    # plt.title('Decision function')
    # plt.grid()

    pos_old = Xu_iter.shape[0]
    y_test_pred = np.argmax(model((torch.from_numpy(Xu_test.astype(np.float32)) - mean) / std).detach().numpy(), axis=1)
    pos_new = np.sum([1 for i in range(Xu_test.shape[0]) if y_test_pred[i]==1])/Xu_test.shape[0]
    
    init_times = 0
    
    while pos_new <= pos_old:
    	time_before = time.time()
        ocp = OCPpendulum(mean.item(), std.item(), model)
        # ocp = OCPpendulum(clf, X_iter)
        init_times = init_times + time.time() - time_before

        Xu_iter = np.random.uniform(low=[q_min-(q_max-q_min)/5, v_min-(v_max-v_min)/5], high=[q_max+(q_max-q_min)/5, v_max+(v_max-v_min)/5], size=(B,2))
        
        X_iter = np.empty((0, 2))
        y_iter = np.empty((0, 2))
        # y_iter = np.empty((0, 1))

        for n in range(Xu_iter.shape[0]):
            q0 = Xu_iter[n, 0]
            v0 = Xu_iter[n, 1]

            if q0>=q_min and q0<=q_max and v0>=v_min and v0<=v_max:
                res = ocp.compute_problem(q0, v0)
                if res == 1:
                    X_iter = np.append(X_iter, [[q0, v0]], axis=0)

                    inp = (torch.Tensor([ocp.ocp_solver.get(0, 'x')]) - mean) / std
                    out = model(inp)
                    y_pred = np.argmax(out.detach().numpy(), axis=1)

                    if y_pred == 1 and ocp.ocp_solver.get_cost() < 0.:
                        y_iter = np.append(y_iter, [[0, 1]], axis=0)
                        # y_iter = np.append(y_iter, 1)
                    else:
                        y_iter = np.append(y_iter, [[1, 0]], axis=0)
                        # y_iter = np.append(y_iter, 0)
            else:
                X_iter = np.append(X_iter, [[q0, v0]], axis=0)
                y_iter = np.append(y_iter, [[1, 0]], axis=0)
                # y_iter = np.append(y_iter, 0)

        # del model
        # model = NeuralNet(input_size, hidden_size, output_size).to(device)

        pos_old = pos_new
        y_test_pred = np.argmax(model((torch.from_numpy(Xu_test.astype(np.float32)) - mean) / std).detach().numpy(), axis=1)
        pos_new = np.sum([1 for i in range(Xu_test.shape[0]) if y_test_pred[i]==1])/Xu_test.shape[0]

        # print(pos,neg)

        # del model, optimizer
        # model = NeuralNet(input_size, hidden_size, output_size).to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        it = 0
        val = 1

        # Train the model
        while val > loss_stop and it <= it_max:

            ind = random.sample(range(X_iter.shape[0]), n_minibatch)

            X_iter_tensor = torch.from_numpy(X_iter[ind].astype(np.float32))
            y_iter_tensor = torch.from_numpy(y_iter[ind].astype(np.float32))
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

        # clf.fit(X_iter, y_iter)
        
        h = 0.02
        xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))

        with torch.no_grad():
            # Plot the results:
            plt.figure()
            h = 0.01
            x_min, x_max = q_min-h, q_max+h
            y_min, y_max = v_min, v_max
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            inp = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
            inp = (inp - mean) / std
            out = model(inp)
            y_pred = np.argmax(out.numpy(), axis=1)
            Z = y_pred.reshape(xx.shape)
            z = [
                0 if np.array_equal(y_iter[x], [1, 0]) else 1
                for x in range(y_iter.shape[0])
            ]
            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
            scatter = plt.scatter(
                X_iter[:, 0], X_iter[:, 1], c=z, marker=".", alpha=0.5, cmap=plt.cm.Paired
            )
            plt.xlim([q_min, q_max])
            plt.ylim([v_min, v_max])
            plt.xlabel("Initial position [rad]")
            plt.ylabel("Initial velocity [rad/s]")
            plt.title("Classifier")
            hand = scatter.legend_elements()[0]
            plt.legend(handles=hand, labels=("Non viable", "Viable"))
            plt.grid(True)
            # plt.show()

            # Plot of the decision function:
            plt.figure()
            y_plot = out.numpy()[:,0]
            y_plot = y_plot.reshape(xx.shape)
            levels = np.linspace(y_plot.min(), y_plot.max(), 20)
            plt.contourf(xx, yy, y_plot, levels=levels, cmap='plasma')
            this = plt.contour(xx, yy, y_plot, levels=levels, colors=('k',), linewidths=(1,))
            plt.clabel(this, fmt='%2.1f', colors='w', fontsize=11)
            #df = plt.contour(xx, yy, out, levels=[0], linewidths=(2,), colors=("k",))
            #plt.clabel(df, fmt='%2.1f', colors='w', fontsize=11)
            plt.xlabel('Initial position [rad]')
            plt.ylabel('Initial velocity [rad/s]')
            plt.title('Decision function')
            plt.xlim([q_min, q_max])
            plt.ylim([v_min, v_max])
            plt.grid()

        # # Plot the results:
        # plt.figure(figsize=(6, 5))
        # out = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        # out = out.reshape(xx.shape)
        # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # plt.contour(xx, yy, out, levels=[0], linewidths=(2,), colors=("k",))
        # scatter = plt.scatter(
        #     X_iter[:, 0],
        #     X_iter[:, 1],
        #     c=y_iter,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([0.0, np.pi / 2 - 0.01])
        # plt.ylim([-10.0, 10.0])
        # plt.xlabel("Initial position [rad]")
        # plt.ylabel("Initial velocity [rad/s]")
        # plt.title("Classifier")
        # hand = scatter.legend_elements()[0]
        # plt.legend(handles=hand, labels=("Non viable", "Viable"), loc='upper right')
        # plt.grid()

        # # Plot of the decision function:
        # plt.figure(figsize=(6, 5))
        # levels = np.linspace(out.min(), out.max(), 10)
        # plt.contourf(xx, yy, out, levels=levels, cmap='plasma')
        # this = plt.contour(xx, yy, out, levels=levels, colors=('k',), linewidths=(1,))
        # plt.clabel(this, fmt='%2.1f', colors='w', fontsize=11)
        # #df = plt.contour(xx, yy, out, levels=[0], linewidths=(2,), colors=("k",))
        # #plt.clabel(df, fmt='%2.1f', colors='w', fontsize=11)
        # plt.xlabel('Initial position [rad]')
        # plt.ylabel('Initial velocity [rad/s]')
        # plt.title('Decision function')
        # plt.grid()

        del ocp

    print(time.time() - start_time - init_times)
    plt.show()
    # pr.print_stats(sort="cumtime")
