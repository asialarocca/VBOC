import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
import warnings
warnings.filterwarnings("ignore")
import torch

def plots_2dof(X_save, q_min, q_max, v_min, v_max, model_dir, mean_dir, std_dir, device):

    # Plot all training data:
    plt.figure()
    plt.scatter(X_save[:,0],X_save[:,1],s=0.1)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("OCP dataset positions")
    plt.figure()
    plt.scatter(X_save[:,2],X_save[:,3],s=0.1)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("OCP dataset velocities")

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
        plt.ylabel('$\dot{q}_2$')
        plt.xlabel('$q_2$')
        plt.grid()
        plt.title("Classifier section")

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
        plt.ylabel('$\dot{q}_1$')
        plt.xlabel('$q_1$')
        plt.grid()
        plt.title("Classifier section")

    plt.show()
