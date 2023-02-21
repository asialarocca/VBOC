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
from my_nn import NeuralNet, NeuralNetGuess


if __name__ == "__main__":

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPdoublependulumRINIT()
    sim = SYMdoublependulumINIT()

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = - ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    with torch.no_grad():

        # Hyper-parameters for nn:
        input_size = 4
        hidden_size = 4 * 100
        output_size = 2
        learning_rate = 0.001

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(torch.load('data_vel_5/model_2pendulum_5'))

        mean, std = torch.tensor(1.9635), torch.tensor(7.0253)

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
        # plt.xlim([q_min, q_max])
        # plt.ylim([v_min, v_max])
        # plt.grid()
        # plt.title("q1 not centered and v2 centered")

        xx, yy = np.meshgrid(np.arange(v_min, v_max, h), np.arange(v_min, v_max, h))
        xrav = xx.ravel()
        yrav = yy.ravel()

        for _ in range(10):
            q1ran = q_min + random.random() * (q_max-q_min)
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
            plt.xlim([v_min, v_max])
            plt.ylim([v_min, v_max])
            plt.grid()
            plt.title("q1="+str(q1ran)+"q2="+str(q2ran))

    print("Execution time: %s seconds" % (time.time() - start_time))

    plt.show()
