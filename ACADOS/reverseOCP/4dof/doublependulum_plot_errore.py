import numpy as np
import random
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from numpy.linalg import norm as norm
import time
from doublependulum_class_fixedveldir import OCPdoublependulumRINIT, SYMdoublependulumINIT
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from my_nn import NeuralNet, NeuralNetRegression
import math


if __name__ == "__main__":

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPdoublependulumRINIT()
    sim = SYMdoublependulumINIT()

    # Position, velocity and torque bounds:
    v_max = ocp.dthetamax
    v_min = - ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin
    tau_max = ocp.Cmax

    with torch.no_grad():

        # Device configuration
        device = torch.device("cpu")

        model_al = NeuralNet(4, 400, 2).to(device)
        model_al.load_state_dict(torch.load('data_vel_20/model_2pendulum_20'))

        mean, std = torch.tensor(1.9635), torch.tensor(9.2003)

        x_sol = np.load('x_sol.npy')
        u_sol = np.load('u_sol.npy')
        x_sym = np.load('x_sym.npy')
        p = np.load('p.npy')

        x_sol_new = np.copy(x_sol)
        u_sol_new = np.copy(u_sol)

        print('Original OCP')
        print(p[0] * x_sol[0][2] + p[1] * x_sol[0][3])
        print(x_sol[0][2:4])
        print('-----------------------------------------------')

        N = ocp.N

        eps = 1e-4

        joint_sel = 1
        joint_oth = int(1 - joint_sel)
        q_init_oth = x_sol[0][joint_oth]
        q_init_sel = q_max
        q_fin_sel = q_min
        q_init_lb = np.array([q_init_oth, q_init_oth, v_min, v_min, 1e-2])
        q_init_ub = np.array([q_init_oth, q_init_oth, v_max, v_max, 1e-2])
        if q_init_sel == q_min:
            q_init_lb[joint_sel] = q_min + eps
            q_init_ub[joint_sel] = q_min + eps
        else:
            q_init_lb[joint_sel] = q_max - eps
            q_init_ub[joint_sel] = q_max - eps
        q_fin_lb = np.array([q_min, q_min, 0., 0., 1e-2])
        q_fin_ub = np.array([q_max, q_max, 0., 0., 1e-2])
        q_fin_lb[joint_sel] = q_fin_sel
        q_fin_ub[joint_sel] = q_fin_sel

        cost = 1.

        while True:
            ocp.ocp_solver.reset()

            for i in range(len(u_sol_new)):
                ocp.ocp_solver.set(i, 'x', x_sol_new[i])
                ocp.ocp_solver.set(i, 'u', u_sol_new[i])
                ocp.ocp_solver.set(i, 'p', p)
                ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1e-2]))
                ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1e-2]))
                ocp.ocp_solver.constraints_set(i, 'lbu', np.array([-tau_max, -tau_max]))
                ocp.ocp_solver.constraints_set(i, 'ubu', np.array([tau_max, tau_max]))
                ocp.ocp_solver.constraints_set(i, 'C', np.array([[0., 0., 0., 0., 0.]]))
                ocp.ocp_solver.constraints_set(i, 'D', np.array([[0., 0.]]))
                ocp.ocp_solver.constraints_set(i, 'lg', np.array([0.]))
                ocp.ocp_solver.constraints_set(i, 'ug', np.array([0.]))

            ocp.ocp_solver.constraints_set(0, "lbx", q_init_lb)
            ocp.ocp_solver.constraints_set(0, "ubx", q_init_ub)
            ocp.ocp_solver.constraints_set(0, "C", np.array([[0., 0., p[1], -p[0], 0.]]))

            u_guess = np.array([ocp.g*ocp.l1*(ocp.m1+ocp.m2)*math.sin(x_sol_new[len(u_sol_new)][0]),ocp.g*ocp.l2*ocp.m2*math.sin(x_sol_new[len(u_sol_new)][1])])

            for i in range(len(u_sol_new), N):
                ocp.ocp_solver.set(i, 'x', x_sol_new[len(u_sol_new)])
                ocp.ocp_solver.set(i, 'u', u_guess)
                ocp.ocp_solver.set(i, 'p', p)
                ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1e-2]))
                ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1e-2]))
                ocp.ocp_solver.constraints_set(i, 'lbu', np.array([-tau_max, -tau_max]))
                ocp.ocp_solver.constraints_set(i, 'ubu', np.array([tau_max, tau_max]))
                ocp.ocp_solver.constraints_set(i, 'C', np.array([[0., 0., 0., 0., 0.]]))
                ocp.ocp_solver.constraints_set(i, 'D', np.array([[0., 0.]]))
                ocp.ocp_solver.constraints_set(i, 'lg', np.array([0.]))
                ocp.ocp_solver.constraints_set(i, 'ug', np.array([0.]))

            ocp.ocp_solver.constraints_set(N, "lbx", q_fin_lb)
            ocp.ocp_solver.constraints_set(N, "ubx", q_fin_ub)
            ocp.ocp_solver.set(N, 'x', x_sol_new[len(u_sol_new)])
            ocp.ocp_solver.set(N, 'p', p)

            status = ocp.ocp_solver.solve()

            if status == 0:
                cost_new = ocp.ocp_solver.get_cost()
                print(cost_new)
                ocp.ocp_solver.print_statistics()

                if cost_new > float(f'{cost:.6f}') - 1e-6:
                    print(N-1, "steps are enough")
                    break

                cost = cost_new

                x_sol_new = np.empty((N+1,5))
                u_sol_new = np.empty((N,2))

                for i in range(N):
                    x_sol_new[i] = ocp.ocp_solver.get(i, "x")
                    u_sol_new[i] = ocp.ocp_solver.get(i, "u")

                x_sol_new[N] = ocp.ocp_solver.get(N, "x")

                print(x_sol_new[0][2:4])
                print('-----------------------------------------------')

                N = N + 1
                ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
                ocp.ocp_solver.update_qp_solver_cond_N(N)
            else:
                print('FAILED')
                ocp.ocp_solver.print_statistics()
                break

        x_out = x_sym[0][:4]
        print('Iter', 0)
        print(x_sym[0][:4], x_out)

        for f in range(1, ocp.N+1):
            # print(x_sol[i], x_sym[i])

            sim.acados_integrator.set("u", u_sol[f-1])
            sim.acados_integrator.set("x", x_out)
            status = sim.acados_integrator.solve()
            x_out = sim.acados_integrator.get("x")

            if x_out[0] > q_max or x_out[0] < q_min or x_out[1] > q_max or x_out[1] < q_min or x_out[2] > v_max or x_out[2] < v_min or x_out[3] > v_max or x_out[3] < v_min:
                print('OUTTTTTTTTTTTTTTTTTTTTTTTTT')

            if x_sol[f][0] > q_max - eps or x_sol[f][0] < q_min + eps or x_sol[f][1] > q_max - eps or x_sol[f][1] < q_min + eps or x_sol[f][2] > v_max - eps or x_sol[f][2] < v_min + eps or x_sol[f][3] > v_max - eps or x_sol[f][3] < v_min + eps:
                print('SOLUTION NEAR THE BORDER', x_sol[f])

            print('Iter', f)
            print(x_sym[f][:4], x_out)

                            # u_sym = np.copy(u_sol[f-1])
                                
                            # if u_sym[0] < -tau_max + eps:
                            #     u_sym[0] = u_sym[0] + eps
                            #     tlim = True
                            # else:
                            #     if u_sym[0] > tau_max - eps:
                            #         u_sym[0] = u_sym[0] - eps
                            #         tlim = True
                            #     else:
                            #         if u_sym[1] > tau_max - eps:
                            #             u_sym[1] = u_sym[1] - eps
                            #             tlim = True
                            #         else:
                            #             if u_sym[1] < -tau_max + eps:
                            #                 u_sym[1] = u_sym[1] + eps
                            #                 tlim = True
                            #             else:
                            #                 print('no torque at limit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            #                 tlim = False

                            # if tlim:
                            #     sim.acados_integrator.set("u", u_sym)
                            #     sim.acados_integrator.set("x", x_out)
                            #     sim.acados_integrator.set("T", dt_sym)
                            #     status = sim.acados_integrator.solve()
                            #     x_out = sim.acados_integrator.get("x")

                            #     isout = False

                            #     if x_out[0] > q_max or x_out[0] < q_min or x_out[1] > q_max or x_out[1] < q_min or x_out[2] > v_max or x_out[2] < v_min or x_out[3] > v_max or x_out[3] < v_min:
                            #         print('tutto ok')
                            #         isout = True
                            #     else:
                            #         for i in range(f+1, N):
                            #             u_sym = np.copy(u_sol[i-1])

                            #             sim.acados_integrator.set("u", u_sym)
                            #             sim.acados_integrator.set("x", x_out)
                            #             sim.acados_integrator.set("T", dt_sym)
                            #             status = sim.acados_integrator.solve()
                            #             x_out = sim.acados_integrator.get("x")

                            #             if x_out[0] > q_max or x_out[0] < q_min or x_out[1] > q_max or x_out[1] < q_min or x_out[2] > v_max or x_out[2] < v_min or x_out[3] > v_max or x_out[3] < v_min:
                            #                 print('tutto ok')
                            #                 isout = True
                            #                 break

                            #         if isout == False:
                            #             print('ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        # ocp = OCPdoublependulumRINIT()

        # ocp.ocp_solver.reset()

        # for i in range(ocp.N):
        #     ocp.ocp_solver.set(i, 'x', x_sol[i])
        #     ocp.ocp_solver.set(i, 'u', u_sol_new[i])
        #     ocp.ocp_solver.set(i, 'p', p)

        # ocp.ocp_solver.constraints_set(0, "lbx", np.array([x_sym[0][0], x_sym[0][1], x_sym[0][2], x_sym[0][3], 1e-2]))
        # ocp.ocp_solver.constraints_set(0, "ubx", np.array([x_sym[0][0], x_sym[0][1], x_sym[0][2], x_sym[0][3], 1e-2]))

        # ocp.ocp_solver.constraints_set(ocp.N, "lbx", q_fin_lb)
        # ocp.ocp_solver.constraints_set(ocp.N, "ubx", q_fin_ub)
        # ocp.ocp_solver.set(ocp.N, 'x', x_sol[ocp.N])
        # ocp.ocp_solver.set(ocp.N, 'p', p)

        # status = ocp.ocp_solver.solve()

        # if status != 0:
        #     print('The initial state is already optimal')

        X_save = np.load('data_vel_20/data_al_20.npy')

        h = 0.02
        xx, yy = np.meshgrid(np.arange(v_min, v_max, h), np.arange(v_min, v_max, h))
        xrav = xx.ravel()
        yrav = yy.ravel()

        for l in range(33,60):
            q1ran = x_sol[l][0]
            q2ran = x_sol[l][1]

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
            out = model_al(inp)
            y_pred = np.argmax(out.numpy(), axis=1)
            Z = y_pred.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
            plt.plot(
                x_sol[l][2],
                x_sol[l][3],
                "ko",
            )
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
            plt.title(str(l)+"q1="+str(q1ran)+"q2="+str(q2ran))

    print("Execution time: %s seconds" % (time.time() - start_time))

    plt.show()
