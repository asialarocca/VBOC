import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
import warnings
warnings.filterwarnings("ignore")
import torch
import random

def plots_3dof(X_save, q_min, q_max, v_min, v_max, model_dir, mean_dir, std_dir, device):

    # Plot all training data:
    plt.figure()
    ax = plt.axes(projection ="3d")
    ax.scatter3D(X_save[:,0],X_save[:,1],X_save[:,2])
    plt.title("OCP dataset positions")
    plt.figure()
    ax = plt.axes(projection ="3d")
    ax.scatter3D(X_save[:,3],X_save[:,4],X_save[:,5])
    plt.title("OCP dataset velocities")

    # Show the resulting set approximation:
    with torch.no_grad():
        h = 0.01
        xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))
        xrav = xx.ravel()
        yrav = yy.ravel()

        plt.figure()
        inp = np.float32(
                np.c_[(q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                        (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                        xrav,
                        np.zeros(yrav.shape[0]),
                        np.zeros(yrav.shape[0]),
                        yrav,
                        np.empty(yrav.shape[0]),
                        ]
            )
        for i in range(inp.shape[0]):
            vel_norm = norm([inp[i,3],inp[i,4],inp[i,5]])
            inp[i][0] = (inp[i][0] - mean_dir) / std_dir
            inp[i][1] = (inp[i][1] - mean_dir) / std_dir
            inp[i][2] = (inp[i][2] - mean_dir) / std_dir
            if vel_norm != 0:
                inp[i][4] = inp[i][4] / vel_norm
                inp[i][3] = inp[i][3] / vel_norm
                inp[i][5] = inp[i][5] / vel_norm
            inp[i][6] = vel_norm
        out = (model_dir(torch.from_numpy(inp[:,:6].astype(np.float32)).to(device))).cpu().numpy() 
        y_pred = np.empty(out.shape)
        for i in range(len(out)):
            if inp[i][6] > out[i]:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        xit = []
        yit = []
        for i in range(len(X_save)):
            if (
                norm(X_save[i][0] - (q_min + q_max) / 2) < 0.1 and
                norm(X_save[i][1] - (q_min + q_max) / 2) < 0.1
                and norm(X_save[i][3]) < 0.1
                and norm(X_save[i][4]) < 0.1
            ):
                xit.append(X_save[i][2])
                yit.append(X_save[i][5])
        plt.plot(
            xit,
            yit,
            "ko",
            markersize=2
        )
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.ylabel('$\dot{q}_3$')
        plt.xlabel('$q_3$')
        plt.grid()
        plt.title("Classifier section")

        # Plot the results:
        plt.figure()
        inp = np.float32(
                np.c_[(q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                        xrav,
                        (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                        np.zeros(yrav.shape[0]),
                        yrav,
                        np.zeros(yrav.shape[0]),
                        np.empty(yrav.shape[0]),
                        ]
            )
        for i in range(inp.shape[0]):
            vel_norm = norm([inp[i,3],inp[i,4],inp[i,5]])
            inp[i][0] = (inp[i][0] - mean_dir) / std_dir
            inp[i][1] = (inp[i][1] - mean_dir) / std_dir
            inp[i][2] = (inp[i][2] - mean_dir) / std_dir
            if vel_norm != 0:
                inp[i][4] = inp[i][4] / vel_norm
                inp[i][3] = inp[i][3] / vel_norm
                inp[i][5] = inp[i][5] / vel_norm
            inp[i][6] = vel_norm
        out = (model_dir(torch.from_numpy(inp[:,:6].astype(np.float32)).to(device))).cpu().numpy() 
        y_pred = np.empty(out.shape)
        for i in range(len(out)):
            if inp[i][6] > out[i]:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        xit = []
        yit = []
        for i in range(len(X_save)):
            if (
                norm(X_save[i][0] - (q_min + q_max) / 2) < 0.1 and
                norm(X_save[i][2] - (q_min + q_max) / 2) < 0.1
                and norm(X_save[i][3]) < 0.1
                and norm(X_save[i][5]) < 0.1
            ):
                xit.append(X_save[i][1])
                yit.append(X_save[i][4])
        plt.plot(
            xit,
            yit,
            "ko",
            markersize=2
        )
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.ylabel('$\dot{q}_2$')
        plt.xlabel('$q_2$')
        plt.grid()
        plt.title("Classifier section")

        # Plot the results:
        plt.figure()
        inp = np.float32(
                np.c_[xrav,
                        (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                        (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                        yrav,
                        np.zeros(yrav.shape[0]),
                        np.zeros(yrav.shape[0]),
                        np.empty(yrav.shape[0]),
                        ]
            )
        for i in range(inp.shape[0]):
            vel_norm = norm([inp[i,3],inp[i,4],inp[i,5]])
            inp[i][0] = (inp[i][0] - mean_dir) / std_dir
            inp[i][1] = (inp[i][1] - mean_dir) / std_dir
            inp[i][2] = (inp[i][2] - mean_dir) / std_dir
            if vel_norm != 0:
                inp[i][4] = inp[i][4] / vel_norm
                inp[i][3] = inp[i][3] / vel_norm
                inp[i][5] = inp[i][5] / vel_norm
            inp[i][6] = vel_norm
        out = (model_dir(torch.from_numpy(inp[:,:6].astype(np.float32)).to(device))).cpu().numpy() 
        y_pred = np.empty(out.shape)
        for i in range(len(out)):
            if inp[i][6] > out[i]:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        xit = []
        yit = []
        for i in range(len(X_save)):
            if (
                norm(X_save[i][1] - (q_min + q_max) / 2) < 0.1 and
                norm(X_save[i][2] - (q_min + q_max) / 2) < 0.1
                and norm(X_save[i][4]) < 0.1
                and norm(X_save[i][5]) < 0.1
            ):
                xit.append(X_save[i][0])
                yit.append(X_save[i][3])
        plt.plot(
            xit,
            yit,
            "ko",
            markersize=2
        )
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.ylabel('$\dot{q}_1$')
        plt.xlabel('$q_1$')
        plt.grid()
        plt.title("Classifier section")

        # for _ in range(10):
        #     q1ran = q_min + random.random() * (q_max-q_min)
        #     q2ran = q_min + random.random() * (q_max-q_min)
        #     q3ran = q_min + random.random() * (q_max-q_min)

        #     plt.figure()
        #     ax = plt.axes(projection ="3d")
        #     xit = np.zeros((1,6))
        #     for i in range(len(X_save)):
        #         if (
        #             norm(X_save[i][1] - q1ran) < 0.1 and
        #             norm(X_save[i][2] - q2ran) < 0.1 and
        #             norm(X_save[i][3] - q3ran) < 0.1
        #         ):
        #             xit = np.append(xit,[X_save[i]], axis=0)
        #     ax.scatter3D(xit[:,3],xit[:,4],xit[:,5])
        #     plt.title("q1="+str(q1ran)+" q2="+str(q2ran)+" q3="+str(q3ran))

    plt.show()
