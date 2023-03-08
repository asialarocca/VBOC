import numpy as np
import random 
import matplotlib.pyplot as plt
from sklearn import svm
from numpy.linalg import norm as norm
import time
from double_pendulum_ocp_class_novellimits import OCPdoublependulumRINIT, SYMdoublependulumINIT
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from my_nn import NeuralNet


if __name__ == "__main__":

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPdoublependulumRINIT()
    sim = SYMdoublependulumINIT()

    # Position and velocity bounds:
    v_max = 20
    v_min = - 20
    q_max = np.pi / 2 + np.pi
    q_min = np.pi

    with torch.no_grad():

        # Hyper-parameters for nn:
        input_size = 4
        hidden_size = 4 * 100
        output_size = 2
        learning_rate = 0.001

        # Device configuration
        device = torch.device("cpu")

        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(torch.load('data_vel_20/model_2pendulum_20'))

        mean, std = torch.tensor(1.9635), torch.tensor(9.2003)

        X_save = np.load('data_vel_20/data_al_20.npy')

        # Plots:
        h = 0.02
        xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
        xrav = xx.ravel()
        yrav = yy.ravel()

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
        #             xrav,
        #             np.zeros(yrav.shape[0]),
        #             yrav,
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     xit.append(X_save[i][1])
        #     yit.append(X_save[i][3])
        #     if X_save[i][5] < 0.5:
        #         cit.append(0)
        #     else:
        #         cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q1 and v1 centered ALL POINTS")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             xrav,
        #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
        #             yrav,
        #             np.zeros(yrav.shape[0]),
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     xit.append(X_save[i][0])
        #     yit.append(X_save[i][2])
        #     if X_save[i][5] < 0.5:
        #         cit.append(0)
        #     else:
        #         cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q2 and v2 centered ALL POINTS")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
        #             xrav,
        #             np.zeros(yrav.shape[0]),
        #             yrav,
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][0] - (q_min + q_max) / 2) < 0.1/10
        #         and norm(X_save[i][2]) < 1./10
        #     ):
        #         xit.append(X_save[i][1])
        #         yit.append(X_save[i][3])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q1 and v1 centered")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
        #             xrav,
        #             3 * np.ones(yrav.shape[0]),
        #             yrav,
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][0] - (q_min + q_max) / 2) < 0.1/10
        #         and norm(X_save[i][2] - 3) < 1./10
        #     ):
        #         xit.append(X_save[i][1])
        #         yit.append(X_save[i][3])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q1 centered and v1 not centered (1)")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
        #             xrav,
        #             -3 * np.ones(yrav.shape[0]),
        #             yrav,
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][0] - (q_min + q_max) / 2) < 0.1/10
        #         and norm(X_save[i][2] + 3) < 1./10
        #     ):
        #         xit.append(X_save[i][1])
        #         yit.append(X_save[i][3])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q1 centered and v1 not centered (2)")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             3.4 * np.ones(xrav.shape[0]),
        #             xrav,
        #             np.zeros(yrav.shape[0]),
        #             yrav,
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][0] - 3.4) < 0.1/10
        #         and norm(X_save[i][2]) < 1./10
        #     ):
        #         xit.append(X_save[i][1])
        #         yit.append(X_save[i][3])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q1 not centered and v1 centered (1)")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             4.4 * np.ones(xrav.shape[0]),
        #             xrav,
        #             np.zeros(yrav.shape[0]),
        #             yrav,
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][0] - 4.4) < 0.1/10
        #         and norm(X_save[i][2]) < 1./10
        #     ):
        #         xit.append(X_save[i][1])
        #         yit.append(X_save[i][3])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q1 not centered and v1 centered (2)")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             xrav,
        #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
        #             yrav,
        #             np.zeros(yrav.shape[0]),
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][1] - (q_min + q_max) / 2) < 0.1/10
        #         and norm(X_save[i][3]) < 1./10
        #     ):
        #         xit.append(X_save[i][0])
        #         yit.append(X_save[i][2])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q2 and v2 centered")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             xrav,
        #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
        #             yrav,
        #             3 * np.ones(yrav.shape[0]),
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][1] - (q_min + q_max) / 2) < 0.1/10
        #         and norm(X_save[i][3] - 3) < 1./10
        #     ):
        #         xit.append(X_save[i][0])
        #         yit.append(X_save[i][2])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q2 centered and v2 not centered (1)")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             xrav,
        #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
        #             yrav,
        #             - 3 * np.ones(yrav.shape[0]),
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][1] - (q_min + q_max) / 2) < 0.1/10
        #         and norm(X_save[i][3] + 3) < 1./10
        #     ):
        #         xit.append(X_save[i][0])
        #         yit.append(X_save[i][2])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q2 centered and v2 not centered (2)")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             xrav,
        #             3.4 * np.ones(xrav.shape[0]),
        #             yrav,
        #             np.zeros(yrav.shape[0]),
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][1] - 3.4) < 0.1/10
        #         and norm(X_save[i][3]) < 1./10
        #     ):
        #         xit.append(X_save[i][0])
        #         yit.append(X_save[i][2])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q2 not centered and v2 centered (1)")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             xrav,
        #             4.4 * np.ones(xrav.shape[0]),
        #             yrav,
        #             np.zeros(yrav.shape[0]),
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][1] - 4.4) < 0.1/10
        #         and norm(X_save[i][3]) < 1./10
        #     ):
        #         xit.append(X_save[i][0])
        #         yit.append(X_save[i][2])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q2 not centered and v2 centered (2)")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             xrav,
        #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
        #             np.zeros(yrav.shape[0]),
        #             yrav,
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][1] - (q_min + q_max) / 2) < 0.1/10
        #         and norm(X_save[i][2]) < 1./10
        #     ):
        #         xit.append(X_save[i][0])
        #         yit.append(X_save[i][3])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q2 and v1 centered")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
        #             xrav,
        #             yrav,
        #             np.zeros(yrav.shape[0]), 
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][0] - (q_min + q_max) / 2) < 0.1/10
        #         and norm(X_save[i][3]) < 1./10
        #     ):
        #         xit.append(X_save[i][1])
        #         yit.append(X_save[i][2])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q1 and v2 centered")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             xrav,
        #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
        #             4. * np.ones(yrav.shape[0]),
        #             yrav,
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][1] - (q_min + q_max) / 2) < 0.1/10
        #         and norm(X_save[i][2] - 4.) < 1./10
        #     ):
        #         xit.append(X_save[i][0])
        #         yit.append(X_save[i][3])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q2 and v1 not centered")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
        #             xrav,
        #             yrav,
        #             4. * np.ones(yrav.shape[0]),
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][0] - (q_min + q_max) / 2) < 0.1/10
        #         and norm(X_save[i][3] - 4.) < 1./10
        #     ):
        #         xit.append(X_save[i][2])
        #         yit.append(X_save[i][3])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q1 and v2 not centered")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             xrav,
        #             q_max - (q_min + q_max) / 6 * np.ones(xrav.shape[0]),
        #             np.zeros(yrav.shape[0]),
        #             yrav,
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][1] - (q_max - (q_min + q_max) / 6)) < 0.1/10
        #         and norm(X_save[i][2]) < 1./10
        #     ):
        #         xit.append(X_save[i][0])
        #         yit.append(X_save[i][3])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q2 not centered and v1 centered")

        # plt.figure()
        # inp = torch.from_numpy(
        #     np.float32(
        #         np.c_[
        #             q_max - (q_min + q_max) / 6 * np.ones(xrav.shape[0]),
        #             xrav,
        #             yrav,
        #             np.zeros(yrav.shape[0]), 
        #         ]
        #     )
        # )
        # inp = (inp - mean) / std
        # out = model(inp)
        # y_pred = np.argmax(out.numpy(), axis=1)
        # Z = y_pred.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # xit = []
        # yit = []
        # cit = []
        # for i in range(len(X_save)):
        #     if (
        #         norm(X_save[i][0] - (q_max - (q_min + q_max) / 6)) < 0.1/10
        #         and norm(X_save[i][3]) < 1./10
        #     ):
        #         xit.append(X_save[i][1])
        #         yit.append(X_save[i][2])
        #         if X_save[i][5] < 0.5:
        #             cit.append(0)
        #         else:
        #             cit.append(1)
        # plt.scatter(
        #     xit,
        #     yit,
        #     c=cit,
        #     marker=".",
        #     alpha=0.5,
        #     cmap=plt.cm.Paired,
        # )
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q1 not centered and v2 centered")

        xx, yy = np.meshgrid(np.arange(v_min, v_max, h), np.arange(v_min, v_max, h))
        xrav = xx.ravel()
        yrav = yy.ravel()

        for _ in range(10):
            # q1ran = q_min + random.random() * (q_max-q_min)
            q1ran = q_min
            q2ran = q_min + random.random() * (q_max-q_min)

            plt.figure()
            inp = torch.from_numpy(
                np.float32(
                    np.c_[
                        q1ran * np.ones(xrav.shape[0]),
                        q2ran * np.ones(xrav.shape[0]),
                        xrav,
                        yrav, 
                    ]
                )
            )
            inp = (inp - mean) / std
            out = model(inp)
            y_pred = np.argmax(out.numpy(), axis=1)
            Z = y_pred.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
            xit = []
            yit = []
            cit = []
            for i in range(len(X_save)):
                if (
                    norm(X_save[i][0] - q1ran) < 0.01
                    and norm(X_save[i][1] - q2ran) < 0.01
                ):
                    xit.append(X_save[i][2])
                    yit.append(X_save[i][3])
                    if X_save[i][5] < 0.5:
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
            plt.xlim([v_min, v_max])
            plt.ylim([v_min, v_max])
            plt.grid()
            plt.title("q1="+str(q1ran)+"q2="+str(q2ran))

    print("Execution time: %s seconds" % (time.time() - start_time))

    plt.show()
