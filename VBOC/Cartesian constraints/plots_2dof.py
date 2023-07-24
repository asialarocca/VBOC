import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import random
from matplotlib.patches import Circle
from doublependulum_class_fixedveldir import OCPdoublependulumINIT

def plots_2dof(X_save, q_min, q_max, v_min, v_max, model_dir, mean_dir, std_dir, device,ocp): #, model_dir_2, mean_dir_2, std_dir_2

    # Plot all training data:
    plt.figure(figsize=(6, 4))
    plt.scatter(X_save[:,0],X_save[:,1],s=0.1)
    plt.xlim([q_min, q_max])
    plt.ylim([q_min, q_max])
    plt.grid(True)
    plt.ylabel('$q_2$ (rad)')
    plt.xlabel('$q_1$ (rad)')
    plt.title("Training dataset positions")
    plt.figure(figsize=(6, 4))
    plt.scatter(X_save[:,2],X_save[:,3],s=0.1)
    plt.grid(True)
    plt.xlim([v_min, v_max])
    plt.ylim([v_min, v_max])
    plt.ylabel('$\dot{q}_2$ (rad/s)')
    plt.xlabel('$\dot{q}_1$ (rad/s)')
    plt.title("Training dataset velocities")

    l1 = ocp.l1
    l2 = ocp.l2
    theta1 = np.pi / 4 + np.pi 
    theta2 = np.pi + np.pi / 8

    circle = plt.Circle((ocp.x_c, ocp.y_c), ocp.radius, color='b')
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.add_patch(circle)
    # for theta1 in np.linspace(ocp.thetamin, ocp.thetamax, 10):
    #     for theta2 in np.linspace(ocp.thetamin, ocp.thetamax, 10):
    plt.plot(0, 0, marker = 'o', color='k', markersize=10)
    plt.plot([0,l1*np.sin(theta1),l1*np.sin(theta1) + l2*np.sin(theta2)], [0,l1*np.cos(theta1),l1*np.cos(theta1) + l2*np.cos(theta2)], marker = 'o', color='k')
    plt.plot([0,l1*np.sin(theta1-np.pi/12),l1*np.sin(theta1-np.pi/12) + l2*np.sin(theta2-np.pi/12)], [0,l1*np.cos(theta1-np.pi/12),l1*np.cos(theta1-np.pi/12) + l2*np.cos(theta2-np.pi/12)], marker = 'o', color='gray',alpha=0.2)
    plt.xlim([-l1-l2+0.7, l1+l2-0.7])
    plt.ylim([-l1-l2-0.1, 0.1])
    plt.axis('off')

    # Show the resulting set approximation:
    with torch.no_grad():
        h = 0.01
        xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))
        xrav = xx.ravel()
        yrav = yy.ravel()

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
        # xit = []
        # yit = []
        # for i in range(X_save.shape[0]):
        #     if (
        #         norm(X_save[i][0] - (q_min + q_max) / 2) < 0.01
        #         and norm(X_save[i][2]) < 0.1
        #     ):
        #         xit.append(X_save[i][1])
        #         yit.append(X_save[i][3])
        # plt.plot(
        #     xit,
        #     yit,
        #     "ko",
        #     markersize=2
        # )
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.ylabel('$\dot{q}_2$ (rad/s)')
        plt.xlabel('$q_2$ (rad)')
        plt.grid()
        plt.title("Set section at $q_1=\pi$ rad and $\dot{q}_1=0$ rad/s")

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
        # xit = []
        # yit = []
        # for i in range(X_save.shape[0]):
        #     if (
        #         norm(X_save[i][1] - (q_min + q_max) / 2) < 0.01
        #         and norm(X_save[i][3]) < 0.1
        #     ):
        #         xit.append(X_save[i][0])
        #         yit.append(X_save[i][2])
        # plt.plot(
        #     xit,
        #     yit,
        #     "ko",
        #     markersize=2
        # )
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.ylabel('$\dot{q}_1$S (rad/s)')
        plt.xlabel('$q_1$ (rad)')
        plt.grid()
        plt.title("Set section at $q_2=\pi$ rad and $\dot{q}_2=0$ rad/s")

        # Plots:
        h = 0.05
        xx, yy = np.meshgrid(np.arange(v_min, v_max+h, h), np.arange(v_min, v_max+h, h))
        xrav = xx.ravel()
        yrav = yy.ravel()

        for _ in range(5):
            q1ran = q_min + random.random() * (q_max-q_min)
            q2ran = q_min + random.random() * (q_max-q_min)
            # q1ran = np.pi - np.arcsin(0.6) 
            # q2ran = np.pi + np.arcsin(0.6)

            # Plot the results:
            plt.figure()
            inp = np.float32(
                    np.c_[
                        q1ran * np.ones(xrav.shape[0]),
                        q2ran * np.ones(xrav.shape[0]),
                        xrav,
                        yrav,
                        np.empty(yrav.shape[0]),
                    ]
                )
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
                    norm(X_save[i][0] - q1ran) < 0.01
                    and norm(X_save[i][1] - q2ran) < 0.01
                ):
                    xit.append(X_save[i][2])
                    yit.append(X_save[i][3])
            plt.plot(
                xit,
                yit,
                "ko",
                markersize=2
            )
            plt.xlim([v_min, v_max])
            plt.ylim([v_min, v_max])
            plt.grid()
            plt.title("q1="+str(q1ran)+" q2="+str(q2ran)+" RT")

            # # Plot the results:
            # plt.figure()
            # inp = np.float32(
            #         np.c_[
            #             q1ran * np.ones(xrav.shape[0]),
            #             q2ran * np.ones(xrav.shape[0]),
            #             xrav,
            #             yrav,
            #             np.empty(yrav.shape[0]),
            #         ]
            #     )
            # for i in range(inp.shape[0]):
            #     vel_norm = norm([inp[i][2],inp[i][3]])
            #     inp[i][0] = (inp[i][0] - mean_dir_2) / std_dir_2
            #     inp[i][1] = (inp[i][1] - mean_dir_2) / std_dir_2
            #     if vel_norm != 0:
            #         inp[i][2] = inp[i][2] / vel_norm
            #         inp[i][3] = inp[i][3] / vel_norm
            #     inp[i][4] = vel_norm
            # out = (model_dir_2(torch.from_numpy(inp[:,:4].astype(np.float32)).to(device))).cpu().numpy() 
            # y_pred = np.empty(out.shape)
            # for i in range(len(out)):
            #     if inp[i][4] > out[i]:
            #         y_pred[i] = 0
            #     else:
            #         y_pred[i] = 1
            # Z = y_pred.reshape(xx.shape)
            # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
            # xit = []
            # yit = []
            # for i in range(X_save.shape[0]):
            #     if (
            #         norm(X_save[i][0] - q1ran) < 0.01
            #         and norm(X_save[i][1] - q2ran) < 0.01
            #     ):
            #         xit.append(X_save[i][2])
            #         yit.append(X_save[i][3])
            # plt.plot(
            #     xit,
            #     yit,
            #     "ko",
            #     markersize=2
            # )
            # plt.xlim([v_min, v_max])
            # plt.ylim([v_min, v_max])
            # plt.grid()
            # plt.title("q1="+str(q1ran)+" q2="+str(q2ran)+" RT2")

    plt.show()

# ocp = OCPdoublependulumINIT()
# sim = SYMdoublependulumINIT()

# # Position, velocity and torque bounds:
# v_max = ocp.dthetamax
# v_min = - ocp.dthetamax
# q_max = ocp.thetamax
# q_min = ocp.thetamin
# tau_max = ocp.Cmax

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pytorch device

# input_layers = ocp.ocp.dims.nx - 1
# hidden_layers = (input_layers - 1) * 100
# output_layers = 1
# learning_rate = 1e-3

# # Load training data:
# X_save = np.load('data_' + str(2) + 'dof_vboc_' + str(int(v_max)) + '.npy')

# # Model and optimizer:
# model_dir = NeuralNetRegression(input_layers, hidden_layers, output_layers).to(device)
# criterion_dir = nn.MSELoss()
# optimizer_dir = torch.optim.Adam(model_dir.parameters(), lr=learning_rate)

# mean_dir = torch.load('mean_' + str(2) + 'dof_vboc_' + str(int(v_max)) + '_' + str(hidden_layers))
# std_dir = torch.load('std_' + str(2) + 'dof_vboc_' + str(int(v_max)) + '_' + str(hidden_layers))

# model_dir.load_state_dict(torch.load('model_2dof_vboc_10_300_0.5_0.73790205'))

# # mean_dir_2 = torch.load('mean_' + str(2) + 'dof_vboc_' + str(int(v_max)) + '_' + str(hidden_layers) + '_nocons')
# # std_dir_2 = torch.load('std_' + str(2) + 'dof_vboc_' + str(int(v_max)) + '_' + str(hidden_layers) + '_nocons')

# # model_dir_2 = NeuralNetRegression(input_layers, hidden_layers, output_layers).to(device)
# # model_dir_2.load_state_dict(torch.load('model_2dof_vboc_10_300_0.5_2.4007833_nocons'))

# plots_2dof(X_save, q_min, q_max, v_min, v_max, model_dir, mean_dir, std_dir, device) #, model_dir_2, mean_dir_2, std_dir_2

# plt.show()