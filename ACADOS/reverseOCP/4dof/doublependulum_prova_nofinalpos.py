import numpy as np
import random 
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
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
    tau_max = ocp.Cmax

    with torch.no_grad():

        # Hyper-parameters for nn:
        input_size = 4
        hidden_size = 4 * 100
        output_size = 2
        learning_rate = 0.001

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_al = NeuralNet(input_size, hidden_size, output_size).to(device)
        model_al.load_state_dict(torch.load('model_2pendulum_30'))

        # mean, std = torch.tensor(1.9635), torch.tensor(3.0036) # max vel = 5
        # mean_al, std_al = torch.tensor(1.9635), torch.tensor(7.0253) # max vel = 15
        mean_al, std_al = torch.tensor(1.9635), torch.tensor(13.6191)

    # # Initialization of the SVM classifier:
    # clf = svm.SVC(C=1e6, kernel='rbf', class_weight='balanced')

    model_rev = NeuralNet(input_size, hidden_size, output_size).to(device)

    loss_stop = 0.2 
    beta = 0.8
    n_minibatch = 512
    B = 1e4
    it_max = 1e5

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model_rev.parameters(), lr=learning_rate)

    eps = 1e-2
    multip = 1.
    num = 100
    
    X_save = np.array([[(q_min+q_max)/2, (q_min+q_max)/2, 0., 0., 0, 1]])
    
    for _ in range(num):
    # while True:
    #     if X_save.shape[0] >= 750000:
    #         break

        ran1 = random.choice([-1, 1]) * random.random() * multip
        ran2 = random.choice([-1, 1]) * random.random() * multip
        p = np.array([0., ran1, -1, ran2, 0.])

        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_max, q_min, 0., 0., 1e-2]))
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_max, q_max, 0., 0., 1e-2]))

        ocp.ocp_solver.reset()

        ocp.ocp_solver.set(ocp.N, 'x', np.array([q_max, (q_min + q_max)/2, 0., 0., 1e-2]))
        ocp.ocp_solver.set(ocp.N, 'p', p)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N, endpoint=False)):
            x_guess = np.array([(1-tau)*q_min + tau*q_max, (q_min + q_max)/2, 2*(1-tau)*(q_max-q_min), 0., 1e-2]) # v_max
            ocp.ocp_solver.set(i, 'x', x_guess)
            ocp.ocp_solver.set(i, 'p', p)
            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1e-2])) 
            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1e-2])) 

        ocp.ocp_solver.constraints_set(0, "lbx", np.array([q_min + eps, q_min, v_min, v_min, 1e-2]))
        ocp.ocp_solver.constraints_set(0, "ubx", np.array([q_min + eps, q_max, v_max, v_max, 1e-2]))

        status = ocp.ocp_solver.solve()

        print('--------------------------------------------')

        if status == 0:
            print('INITIAL OCP SOLVED')

            x0 = ocp.ocp_solver.get(0, "x")
            u0 = ocp.ocp_solver.get(0, "u")

            x_sol = np.empty((ocp.N+1,5))
            u_sol = np.empty((ocp.N,2))

            dt_sym = ocp.ocp_solver.get(0, "x")[4]

            x_sol[0] = x0
            u_sol[0] = u0

            for i in range(1, ocp.N):
                x_sol[i] = ocp.ocp_solver.get(i, "x")
                u_sol[i] = ocp.ocp_solver.get(i, "u")

            x_sol[ocp.N] = ocp.ocp_solver.get(ocp.N, "x")

            # p_mintime = np.array([0., 0., 0., 0., 1.])

            # ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

            # ocp.ocp_solver.reset()

            # for i in range(1, ocp.N):
            #     ocp.ocp_solver.set(i, 'x', x_sol[i])
            #     ocp.ocp_solver.set(i, 'u', u_sol[i])
            #     ocp.ocp_solver.set(i, 'p', p_mintime)
            #     ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, 0., v_min, 0.])) 
            #     ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1e-2])) 

            # ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_max, q_fin, 0., 0., 0.]))
            # ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_max, q_fin, 0., 0., 1e-2]))

            # ocp.ocp_solver.set(ocp.N, 'x', x_sol[ocp.N])
            # ocp.ocp_solver.set(ocp.N, 'p', p_mintime)

            # ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x0[0], x0[1], x0[2], x0[3], 0.])) 
            # ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x0[0], x0[1], x0[2], x0[3], 1e-2])) 

            # ocp.ocp_solver.set(0, 'x', x_sol[0])
            # ocp.ocp_solver.set(0, 'u', u_sol[0])
            # ocp.ocp_solver.set(0, 'p', p_mintime)

            # status = ocp.ocp_solver.solve()

            # if status == 0:
            #     x_sol[0] = ocp.ocp_solver.get(0, "x")
            #     u_sol[0] = ocp.ocp_solver.get(0, "u")

            #     for i in range(1, ocp.N):
            #         x_sol[i] = ocp.ocp_solver.get(i, "x")
            #         u_sol[i] = ocp.ocp_solver.get(i, "u")

            #     x_sol[ocp.N] = ocp.ocp_solver.get(ocp.N, "x")

            #     dt_sym = ocp.ocp_solver.get(0, "x")[4]

            #     print('MIN TIME SOLVED')

            x0 = np.copy(x_sol[0])
            x1_eps = x0[0] - eps * p[0] / (q_max - q_min)
            x2_eps = x0[1] - eps * p[1] / (q_max - q_min)
            x3_eps = x0[2] - eps * p[2] / (2*v_max)
            x4_eps = x0[3] - eps * p[3] / (2*v_max)
            x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

            X_save = np.append(X_save, [[x_sol[ocp.N][0] + eps, x_sol[ocp.N][1], x_sol[ocp.N][2], x_sol[ocp.N][3], 1, 0]], axis = 0)
            X_save = np.append(X_save, [[x_sol[ocp.N][0], x_sol[ocp.N][1], x_sol[ocp.N][2], x_sol[ocp.N][3], 0, 1]], axis = 0)

            xv_state = np.full((ocp.N+1,1),2)
            xv_state[ocp.N] = 1

            if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                is_x_at_limit = True
                print('Stato inziale al limite')
                xv_state[0] = 1
            else:
                is_x_at_limit = False
                print('Stato inziale non al limite')
                xv_state[0] = 0

            X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 1, 0]], axis = 0)
            X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 0, 1]], axis = 0)

            for f in range(1, ocp.N):

                print('Stato ', f)

                if is_x_at_limit:
                    if x_sol[f][0] > q_max - eps or x_sol[f][0] < q_min + eps or x_sol[f][1] > q_max - eps or x_sol[f][1] < q_min + eps or x_sol[f][2] > v_max - eps or x_sol[f][2] < v_min + eps or x_sol[f][3] > v_max - eps or x_sol[f][3] < v_min + eps:
                        # sono su dX
                        is_x_at_limit = True
                        print('Sono su dX, ottenuto controllando i limiti di stato')
                        xv_state[f] = 1

                        x_sym = np.copy(x_sol[f][:4])

                        if x_sol[f][0] > q_max - eps:
                            x_sym[0] = q_max + eps
                        if x_sol[f][0] < q_min + eps:
                            x_sym[0] = q_min - eps
                        if x_sol[f][1] > q_max - eps:
                            x_sym[1] = q_max + eps
                        if x_sol[f][1] < q_min + eps:
                            x_sym[1] = q_min - eps
                        if x_sol[f][2] > v_max - eps:
                            x_sym[2] = v_max + eps
                        if x_sol[f][2] < v_min + eps:
                            x_sym[2] = v_min - eps
                        if x_sol[f][3] > v_max - eps:
                            x_sym[3] = v_max + eps
                        if x_sol[f][3] < v_min + eps:
                            x_sym[3] = v_min - eps
                    else:
                        # sono o su dV o in V
                        is_x_at_limit = False

                        if x_sol[f-1][0] > q_max - eps or x_sol[f-1][0] < q_min + eps:
                            print('Distacco dal limite avvenuto su q1 o v1 neg, interrompo')
                            break

                        ocp.ocp_solver.reset()

                        p = np.array([0., 0., -1., 0., 0.])

                        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

                        ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], dt_sym])) 
                        ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x_sol[f][0], x_sol[f][1], v_max, x_sol[f][3], dt_sym])) 
                        ocp.ocp_solver.set(0, 'x', x_sol[f])
                        ocp.ocp_solver.set(0, 'u', u_sol[f])
                        ocp.ocp_solver.set(0, 'p', p)

                        for i in range(ocp.N - f, ocp.N+1):
                            ocp.ocp_solver.set(i, 'x', x_sol[ocp.N])
                            ocp.ocp_solver.set(i, 'p', p)
                            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                        ocp.ocp_solver.constraints_set(ocp.N - f, "lbx", x_sol[ocp.N])
                        ocp.ocp_solver.constraints_set(ocp.N - f, "ubx", x_sol[ocp.N])

                        for i in range(1, ocp.N - f):
                            ocp.ocp_solver.set(i, 'x', x_sol[i+f])
                            ocp.ocp_solver.set(i, 'u', u_sol[i+f])
                            ocp.ocp_solver.set(i, 'p', p)
                            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                        ocp.ocp_solver.options_set('qp_tol_stat', 1e-2)

                        status = ocp.ocp_solver.solve()

                        ocp.ocp_solver.options_set('qp_tol_stat', 1e-6)

                        if status == 0:
                            print('INTERMEDIATE OCP SOLVED')

                            if ocp.ocp_solver.get(0, "x")[2] > x_sol[f][2] + 1e-2:
                                print('new vel: ', ocp.ocp_solver.get(0, "x")[2], 'old vel: ', x_sol[f][2])
                                break

                                # # sono o su dv o su dx
                                # for i in range(f, ocp.N):
                                #     x_sol[i] = ocp.ocp_solver.get(i-f, "x")
                                #     u_sol[i] = ocp.ocp_solver.get(i-f, "u")

                                # x_sym = np.copy(x_sol[f])
                                # x1_eps = x_sym[0] - eps * p[0]
                                # x2_eps = x_sym[1] - eps * p[1]
                                # x3_eps = x_sym[2] - eps * p[2]
                                # x4_eps = x_sym[3] - eps * p[3]
                                # x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

                                # if x_sym[0] > q_max - eps or x_sym[0] < q_min + eps or x_sym[1] > q_max - eps or x_sym[1] < q_min + eps or x_sym[2] > v_max - eps or x_sym[2] < v_min+ eps or x_sym[3] > v_max - eps or x_sym[3] < v_min + eps:
                                #     print('Sono su dX, ottenuto tramite la nuova soluzione')
                                #     print(x_sol[f], x_sym)
                                #     is_x_at_limit = True
                                #     xv_state[f] = 1
                                # else:
                                #     print('Sono su dV, ottenuto tramite la nuova soluzione')
                                #     print(x_sol[f], x_sym)
                                #     is_x_at_limit = False
                                #     xv_state[f] = 0
                            else:
                                print('Sono su dV, ottenuto tramite la nuova soluzione')
                                print(x_sol[f], x_sym)
                                is_x_at_limit = False
                                xv_state[f] = 0

                                # sono su dv, posso simulare
                                x_sym = np.copy(x_sol[f][:4])
                                x1_eps = x_sym[0] - eps * p[0] / (q_max - q_min)
                                x2_eps = x_sym[1] - eps * p[1] / (q_max - q_min)
                                x3_eps = x_sym[2] - eps * p[2] / (2*v_max)
                                x4_eps = x_sym[3] - eps * p[3] / (2*v_max)
                                x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

                                if x_sym[2] > v_max:
                                    x_sym[2] = v_max
                        else:
                            print('INTERMEDIATE OCP FAILED')
                            break
                        
                else:
                    u_sym = np.copy(u_sol[f-1])
                    sim.acados_integrator.set("u", u_sym)
                    sim.acados_integrator.set("x", x_sym)
                    sim.acados_integrator.set("T", dt_sym)
                    status = sim.acados_integrator.solve()
                    x_sym = sim.acados_integrator.get("x")

                    if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                        # sono su dX
                        print('Sono su dX, ottenuto tramite la simulazione')
                        print(x_sol[f], x_sym)
                        is_x_at_limit = True
                        xv_state[f] = 1
                    else:
                        # sono su dV
                        print('Sono su dV, ottenuto tramite la simulazione')
                        print(x_sol[f], x_sym)
                        is_x_at_limit = False
                        xv_state[f] = 0

                X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 1, 0]], axis = 0)
                X_save = np.append(X_save, [[x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], 0, 1]], axis = 0)

                # sigmoid = nn.Sigmoid()

                # with torch.no_grad():
                #     out_v = model((torch.from_numpy(np.float32([x_sol[f][:4]])).to(device) - mean) / std).numpy()
                #     out_uv = model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std).numpy()
                #     prob_xu = sigmoid(model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std))
                #     etp = entropy(prob_xu, axis=1)

                    # if out_uv[0,0] < out_v[0,0]: # or np.argmax(out_uv, axis=1) == 1:
                    #     print('possibile errore di direzione')
                    #     print('1 set, at the limit, x: ', x_sol[f], out_v)
                    #     print('1 set, at the limit, x_sym: ', x_sym, out_uv)

                    # if etp < 1e-2:
                    #     print('Possibile errore')
                    #     print('1 set, at the limit, x: ', x_sol[f], out_v)
                    #     print('1 set, at the limit, x_sym: ', x_sym, out_uv)

                    #     # u_sym = np.copy(u_sol[f-1])
                            
                    #     # if u_sym[0] < -tau_max + eps:
                    #     #     u_sym[0] = u_sym[0] + eps
                    #     #     tlim = True
                    #     # else:
                    #     #     if u_sym[0] > tau_max - eps:
                    #     #         u_sym[0] = u_sym[0] - eps
                    #     #         tlim = True
                    #     #     else:
                    #     #         if u_sym[1] > tau_max - eps:
                    #     #             u_sym[1] = u_sym[1] - eps
                    #     #             tlim = True
                    #     #         else:
                    #     #             if u_sym[1] < -tau_max + eps:
                    #     #                 u_sym[1] = u_sym[1] + eps
                    #     #                 tlim = True
                    #     #             else:
                    #     #                 print('no torque at limit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    #     #                 tlim = False

                    #     # if tlim:
                    #     #     sim.acados_integrator.set("u", u_sym)
                    #     #     sim.acados_integrator.set("x", x_sym)
                    #     #     sim.acados_integrator.set("T", dt_sym)
                    #     #     status = sim.acados_integrator.solve()
                    #     #     x_sym = sim.acados_integrator.get("x")

                    #     #     isout = False

                    #     #     if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                    #     #         print('tutto ok')
                    #     #         isout = True
                    #     #     else:
                    #     #         for i in range(f+1, ocp.N):
                    #     #             u_sym = np.copy(u_sol[i-1])

                    #     #             sim.acados_integrator.set("u", u_sym)
                    #     #             sim.acados_integrator.set("x", x_sym)
                    #     #             sim.acados_integrator.set("T", dt_sym)
                    #     #             status = sim.acados_integrator.solve()
                    #     #             x_sym = sim.acados_integrator.get("x")

                    #     #             if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                    #     #                 print('tutto ok')
                    #     #                 isout = True
                    #     #                 break

                    #     #         if isout == False:
                    #     #             print('ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                    #     p = np.array([0., 0., -1., 0., 0.])

                    #     ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

                    #     ocp.ocp_solver.reset()

                    #     for i in range(ocp.N - f, ocp.N+1):
                    #         ocp.ocp_solver.set(i, 'x', np.array([q_max, q_fin, 0., 0., dt_sym]))
                    #         ocp.ocp_solver.set(i, 'p', p)

                    #         ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                    #         ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                    #     ocp.ocp_solver.constraints_set(ocp.N - f, "lbx", np.array([q_max, q_fin, 0., 0., dt_sym]))
                    #     ocp.ocp_solver.constraints_set(ocp.N - f, "ubx", np.array([q_max, q_fin, 0., 0., dt_sym]))

                    #     for i in range(1, ocp.N - f):
                    #         ocp.ocp_solver.set(i, 'x', x_sol[i+f])
                    #         ocp.ocp_solver.set(i, 'u', u_sol[i+f])
                    #         ocp.ocp_solver.set(i, 'p', p)

                    #         ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, 0., v_min, dt_sym])) 
                    #         ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                    #     ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], dt_sym])) 
                    #     ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x_sol[f][0], x_sol[f][1], v_max, x_sol[f][3], dt_sym])) 

                    #     ocp.ocp_solver.set(0, 'x', x_sol[f])
                    #     ocp.ocp_solver.set(0, 'u', u_sol[f])
                    #     ocp.ocp_solver.set(0, 'p', p)

                    #     ocp.ocp_solver.options_set('qp_tol_stat', 1e-2)

                    #     status = ocp.ocp_solver.solve()

                    #     ocp.ocp_solver.options_set('qp_tol_stat', 1e-6)

                    #     if status == 0:
                    #         if abs(ocp.ocp_solver.get(0, "x")[2] - x_sol[f][2]) > 1e-2:
                    #             print('velocita piu estrema trovata!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    #             print('v1 max: ', ocp.ocp_solver.get(0, "x")[2], 'v1 used: ', x_sol[f][2])
                    #             break
                    #         else:
                    #             print('in realtà sembra tutto ok')

            # t = np.linspace(0, ocp.N*dt_sym, ocp.N+1)
            # t = range(ocp.N+1)

            # plt.figure()
            # plt.subplot(2, 1, 1)
            # plt.step(t, np.append([u_sol[0, 0]], u_sol[:, 0]))
            # plt.ylabel('$C1$')
            # plt.xlabel('$t$')
            # plt.hlines(ocp.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.hlines(-ocp.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.ylim([-1.2*ocp.Cmax, 1.2*ocp.Cmax])
            # plt.title('Controls')
            # plt.grid()
            # plt.subplot(2, 1, 2)
            # plt.step(t, np.append([u_sol[0, 1]], u_sol[:, 1]))
            # plt.ylabel('$C2$')
            # plt.xlabel('$t$')
            # plt.hlines(ocp.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.hlines(-ocp.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.ylim([-1.2*ocp.Cmax, 1.2*ocp.Cmax])
            # plt.grid()

            # plt.figure()
            # plt.subplot(4, 1, 1)
            # plt.scatter(t, x_sol[:, 0], c = xv_state, linestyle = 'dotted', linewidths = 0.2, marker = '.', cmap=plt.cm.Paired)
            # plt.ylabel('$theta1$')
            # plt.xlabel('$t$')
            # plt.hlines(ocp.thetamax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.hlines(ocp.thetamin, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.title('States')
            # plt.grid()
            # plt.subplot(4, 1, 2)
            # plt.scatter(t, x_sol[:, 1], c = xv_state, linestyle = 'dotted', linewidths = 0.2, marker = '.', cmap=plt.cm.Paired)
            # plt.ylabel('$theta2$')
            # plt.xlabel('$t$')
            # plt.hlines(ocp.thetamax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.hlines(ocp.thetamin, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.grid()
            # plt.subplot(4, 1, 3)
            # plt.scatter(t, x_sol[:, 2], c = xv_state, linestyle = 'dotted', linewidths = 0.2, marker = '.', cmap=plt.cm.Paired)
            # plt.ylabel('$dtheta1$')
            # plt.xlabel('$t$')
            # plt.hlines(ocp.dthetamax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.hlines(-ocp.dthetamax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.grid()
            # plt.subplot(4, 1, 4)
            # plt.scatter(t, x_sol[:, 3], c = xv_state, linestyle = 'dotted', linewidths = 0.2, marker = '.', cmap=plt.cm.Paired)
            # plt.ylabel('$dtheta2$')
            # plt.xlabel('$t$')
            # plt.hlines(ocp.dthetamax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.hlines(-ocp.dthetamax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.grid()

        ran1 = random.choice([-1, 1]) * random.random() * multip
        ran2 = random.choice([-1, 1]) * random.random() * multip
        p = np.array([0., ran1, 1, ran2, 0.])

        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_min, q_min, 0., 0., 1e-2]))
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_min, q_max, 0., 0., 1e-2]))

        ocp.ocp_solver.reset()

        ocp.ocp_solver.set(ocp.N, 'x', np.array([q_min, (q_min + q_max)/2, 0., 0., 1e-2]))
        ocp.ocp_solver.set(ocp.N, 'p', p)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N, endpoint=False)):
            x_guess = np.array([(1-tau)*q_max + tau*q_min, (q_min + q_max)/2, -2*(1-tau)*(q_max-q_min), 0., 1e-2]) # v_min
            ocp.ocp_solver.set(i, 'x', x_guess)
            ocp.ocp_solver.set(i, 'p', p)
            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1e-2])) 
            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1e-2])) 

        ocp.ocp_solver.constraints_set(0, "lbx", np.array([q_max - eps, q_min, v_min, v_min, 1e-2]))
        ocp.ocp_solver.constraints_set(0, "ubx", np.array([q_max - eps, q_max, v_max, v_max, 1e-2]))

        status = ocp.ocp_solver.solve()

        print('--------------------------------------------')

        if status == 0:
            print('INITIAL OCP SOLVED')

            x0 = ocp.ocp_solver.get(0, "x")
            u0 = ocp.ocp_solver.get(0, "u")

            x_sol = np.empty((ocp.N+1,5))
            u_sol = np.empty((ocp.N,2))

            dt_sym = ocp.ocp_solver.get(0, "x")[4]

            x_sol[0] = x0
            u_sol[0] = u0

            for i in range(1, ocp.N):
                x_sol[i] = ocp.ocp_solver.get(i, "x")
                u_sol[i] = ocp.ocp_solver.get(i, "u")

            x_sol[ocp.N] = ocp.ocp_solver.get(ocp.N, "x")

            # p_mintime = np.array([0., 0., 0., 0., 1.])

            # ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

            # ocp.ocp_solver.reset()

            # for i in range(1, ocp.N):
            #     ocp.ocp_solver.set(i, 'x', x_sol[i])
            #     ocp.ocp_solver.set(i, 'u', u_sol[i])
            #     ocp.ocp_solver.set(i, 'p', p_mintime)
            #     ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 0.])) 
            #     ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, 0., v_max, 1e-2])) 

            # ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_min, q_fin, 0., 0., 0.]))
            # ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_min, q_fin, 0., 0., 1e-2]))

            # ocp.ocp_solver.set(ocp.N, 'x', x_sol[ocp.N])
            # ocp.ocp_solver.set(ocp.N, 'p', p_mintime)

            # ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x0[0], x0[1], x0[2], x0[3], 0.])) 
            # ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x0[0], x0[1], x0[2], x0[3], 1e-2])) 

            # ocp.ocp_solver.set(0, 'x', x_sol[0])
            # ocp.ocp_solver.set(0, 'u', u_sol[0])
            # ocp.ocp_solver.set(0, 'p', p_mintime)

            # status = ocp.ocp_solver.solve()

            # if status == 0:
            #     x_sol[0] = ocp.ocp_solver.get(0, "x")
            #     u_sol[0] = ocp.ocp_solver.get(0, "u")

            #     for i in range(1, ocp.N):
            #         x_sol[i] = ocp.ocp_solver.get(i, "x")
            #         u_sol[i] = ocp.ocp_solver.get(i, "u")

            #     x_sol[ocp.N] = ocp.ocp_solver.get(ocp.N, "x")

            #     dt_sym = ocp.ocp_solver.get(0, "x")[4]

            #     print('MIN TIME SOLVED')

            x0 = np.copy(x_sol[0])
            x1_eps = x0[0] - eps * p[0] / (q_max - q_min)
            x2_eps = x0[1] - eps * p[1] / (q_max - q_min)
            x3_eps = x0[2] - eps * p[2] / (2*v_max)
            x4_eps = x0[3] - eps * p[3] / (2*v_max)
            x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

            X_save = np.append(X_save, [[x_sol[ocp.N][0] - eps, x_sol[ocp.N][1], x_sol[ocp.N][2], x_sol[ocp.N][3], 1, 0]], axis = 0)
            X_save = np.append(X_save, [[x_sol[ocp.N][0], x_sol[ocp.N][1], x_sol[ocp.N][2], x_sol[ocp.N][3], 0, 1]], axis = 0)

            xv_state = np.full((ocp.N+1,1),2)
            xv_state[ocp.N] = 1

            if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                is_x_at_limit = True
                print('Stato inziale al limite')
                xv_state[0] = 1
            else:
                is_x_at_limit = False
                print('Stato inziale non al limite')
                xv_state[0] = 0

            X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 1, 0]], axis = 0)
            X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 0, 1]], axis = 0)

            for f in range(1, ocp.N):

                print('Stato ', f)

                if is_x_at_limit:
                    if x_sol[f][0] > q_max - eps or x_sol[f][0] < q_min + eps or x_sol[f][1] > q_max - eps or x_sol[f][1] < q_min + eps or x_sol[f][2] > v_max - eps or x_sol[f][2] < v_min + eps or x_sol[f][3] > v_max - eps or x_sol[f][3] < v_min + eps:
                        # sono su dX
                        is_x_at_limit = True
                        print('Sono su dX, ottenuto controllando i limiti di stato')
                        xv_state[f] = 1

                        x_sym = np.copy(x_sol[f][:4])

                        if x_sol[f][0] > q_max - eps:
                            x_sym[0] = q_max + eps
                        if x_sol[f][0] < q_min + eps:
                            x_sym[0] = q_min - eps
                        if x_sol[f][1] > q_max - eps:
                            x_sym[1] = q_max + eps
                        if x_sol[f][1] < q_min + eps:
                            x_sym[1] = q_min - eps
                        if x_sol[f][2] > v_max - eps:
                            x_sym[2] = v_max + eps
                        if x_sol[f][2] < v_min + eps:
                            x_sym[2] = v_min - eps
                        if x_sol[f][3] > v_max - eps:
                            x_sym[3] = v_max + eps
                        if x_sol[f][3] < v_min + eps:
                            x_sym[3] = v_min - eps
                    else:
                        # sono o su dV o in V
                        is_x_at_limit = False

                        if x_sol[f-1][0] > q_max - eps or x_sol[f-1][0] < q_min + eps:
                            print('Distacco dal limite avvenuto su q1, interrompo')
                            break

                        ocp.ocp_solver.reset()

                        p = np.array([0., 0., 1., 0., 0.])

                        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

                        ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x_sol[f][0], x_sol[f][1], v_min, x_sol[f][3], dt_sym])) 
                        ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], dt_sym])) 
                        ocp.ocp_solver.set(0, 'x', x_sol[f])
                        ocp.ocp_solver.set(0, 'u', u_sol[f])
                        ocp.ocp_solver.set(0, 'p', p)

                        for i in range(ocp.N - f, ocp.N+1):
                            ocp.ocp_solver.set(i, 'x', x_sol[ocp.N])
                            ocp.ocp_solver.set(i, 'p', p)
                            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                        ocp.ocp_solver.constraints_set(ocp.N - f, "lbx", x_sol[ocp.N])
                        ocp.ocp_solver.constraints_set(ocp.N - f, "ubx", x_sol[ocp.N])

                        for i in range(1, ocp.N - f):
                            ocp.ocp_solver.set(i, 'x', x_sol[i+f])
                            ocp.ocp_solver.set(i, 'u', u_sol[i+f])
                            ocp.ocp_solver.set(i, 'p', p)
                            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                        ocp.ocp_solver.options_set('qp_tol_stat', 1e-2)

                        status = ocp.ocp_solver.solve()

                        ocp.ocp_solver.options_set('qp_tol_stat', 1e-6)

                        if status == 0:
                            print('INTERMEDIATE OCP SOLVED')

                            if ocp.ocp_solver.get(0, "x")[2] < x_sol[f][2] - 1e-2:
                                print('new vel: ', ocp.ocp_solver.get(0, "x")[2], 'old vel: ', x_sol[f][2])
                                break

                                # # sono o su dv o su dx
                                # for i in range(f, ocp.N):
                                #     x_sol[i] = ocp.ocp_solver.get(i-f, "x")
                                #     u_sol[i] = ocp.ocp_solver.get(i-f, "u")

                                # x_sym = np.copy(x_sol[f])
                                # x1_eps = x_sym[0] - eps * p[0]
                                # x2_eps = x_sym[1] - eps * p[1]
                                # x3_eps = x_sym[2] - eps * p[2]
                                # x4_eps = x_sym[3] - eps * p[3]
                                # x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

                                # if x_sym[0] > q_max - eps or x_sym[0] < q_min + eps or x_sym[1] > q_max - eps or x_sym[1] < q_min + eps or x_sym[2] > v_max - eps or x_sym[2] < v_min+ eps or x_sym[3] > v_max - eps or x_sym[3] < v_min + eps:
                                #     print('Sono su dX, ottenuto tramite la nuova soluzione')
                                #     print(x_sol[f], x_sym)
                                #     is_x_at_limit = True
                                #     xv_state[f] = 1
                                # else:
                                #     print('Sono su dV, ottenuto tramite la nuova soluzione')
                                #     print(x_sol[f], x_sym)
                                #     is_x_at_limit = False
                                #     xv_state[f] = 0
                            else:
                                print('Sono su dV, ottenuto tramite la nuova soluzione')
                                print(x_sol[f], x_sym)
                                is_x_at_limit = False
                                xv_state[f] = 0

                                # sono su dv, posso simulare
                                x_sym = np.copy(x_sol[f][:4])
                                x1_eps = x_sym[0] - eps * p[0] / (q_max - q_min)
                                x2_eps = x_sym[1] - eps * p[1] / (q_max - q_min)
                                x3_eps = x_sym[2] - eps * p[2] / (2*v_max)
                                x4_eps = x_sym[3] - eps * p[3] / (2*v_max)
                                x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

                                if x_sym[2] < v_min:
                                    x_sym[2] = v_min
                        else:
                            print('INTERMEDIATE OCP FAILED')
                            break
                        
                else:
                    u_sym = np.copy(u_sol[f-1])
                    sim.acados_integrator.set("u", u_sym)
                    sim.acados_integrator.set("x", x_sym)
                    sim.acados_integrator.set("T", dt_sym)
                    status = sim.acados_integrator.solve()
                    x_sym = sim.acados_integrator.get("x")

                    if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                        # sono su dX
                        print('Sono su dX, ottenuto tramite la simulazione')
                        is_x_at_limit = True
                        xv_state[f] = 1
                    else:
                        # sono su dV
                        print('Sono su dV, ottenuto tramite la simulazione')
                        is_x_at_limit = False
                        xv_state[f] = 0

                X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 1, 0]], axis = 0)
                X_save = np.append(X_save, [[x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], 0, 1]], axis = 0)

                # with torch.no_grad():
                #     out_v = model((torch.from_numpy(np.float32([x_sol[f][:4]])).to(device) - mean) / std).numpy()
                #     out_uv = model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std).numpy()
                #     # if out_uv[0,0] < out_v[0,0]: # or np.argmax(out_uv, axis=1) == 1:
                #     #     print('2 set, at the limit, x: ', x_sol[f], out_v)
                #     #     print('2 set, at the limit, x_sym: ', x_sym, out_uv)

                #     if out_uv[0,1] > 5:
                #         print('Possibile errore!')
                #         print('2 set, at the limit, x: ', x_sol[f], out_v)
                #         print('2 set, at the limit, x_sym: ', x_sym, out_uv)

                #         # u_sym = np.copy(u_sol[f-1])
                            
                #         # if u_sym[0] > tau_max - eps:
                #         #     u_sym[0] = u_sym[0] - eps
                #         #     tlim = True
                #         # else:
                #         #     if u_sym[0] < -tau_max + eps:
                #         #         u_sym[0] = u_sym[0] + eps
                #         #         tlim = True
                #         #     else:
                #         #         if u_sym[1] > tau_max - eps:
                #         #             u_sym[1] = u_sym[1] - eps
                #         #             tlim = True
                #         #         else:
                #         #             if u_sym[1] < -tau_max + eps:
                #         #                 u_sym[1] = u_sym[1] + eps
                #         #                 tlim = True
                #         #             else:
                #         #                 print('no torque at limit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                #         #                 tlim = False

                #         # if tlim:
                #         #     sim.acados_integrator.set("u", u_sym)
                #         #     sim.acados_integrator.set("x", x_sym)
                #         #     sim.acados_integrator.set("T", dt_sym)
                #         #     status = sim.acados_integrator.solve()
                #         #     x_sym = sim.acados_integrator.get("x")

                #         #     isout = False

                #         #     if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                #         #         print('tutto ok')
                #         #         isout = True
                #         #     else:
                #         #         for i in range(f+1, ocp.N):
                #         #             u_sym = np.copy(u_sol[i-1])

                #         #             sim.acados_integrator.set("u", u_sym)
                #         #             sim.acados_integrator.set("x", x_sym)
                #         #             sim.acados_integrator.set("T", dt_sym)
                #         #             status = sim.acados_integrator.solve()
                #         #             x_sym = sim.acados_integrator.get("x")

                #         #             if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                #         #                 print('tutto ok')
                #         #                 isout = True
                #         #                 break

                #         #         if isout == False:
                #         #             print('ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                #         p = np.array([0., 0., 1., 0., 0.])

                #         ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

                #         ocp.ocp_solver.reset()

                #         for i in range(ocp.N - f, ocp.N+1):
                #             ocp.ocp_solver.set(i, 'x', np.array([q_min, q_fin, 0., 0., dt_sym]))
                #             ocp.ocp_solver.set(i, 'p', p)

                #             ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                #             ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                #         ocp.ocp_solver.constraints_set(ocp.N - f, "lbx", np.array([q_min, q_fin, 0., 0., dt_sym]))
                #         ocp.ocp_solver.constraints_set(ocp.N - f, "ubx", np.array([q_min, q_fin, 0., 0., dt_sym]))

                #         for i in range(1, ocp.N - f):
                #             ocp.ocp_solver.set(i, 'x', x_sol[i+f])
                #             ocp.ocp_solver.set(i, 'u', u_sol[i+f])
                #             ocp.ocp_solver.set(i, 'p', p)

                #             ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                #             ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, 0., v_max, dt_sym])) 

                #         ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x_sol[f][0], x_sol[f][1], v_min, x_sol[f][3], dt_sym])) 
                #         ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], dt_sym])) 

                #         ocp.ocp_solver.set(0, 'x', x_sol[f])
                #         ocp.ocp_solver.set(0, 'u', u_sol[f])
                #         ocp.ocp_solver.set(0, 'p', p)

                #         ocp.ocp_solver.options_set('qp_tol_stat', eps)

                #         status = ocp.ocp_solver.solve()

                #         ocp.ocp_solver.options_set('qp_tol_stat', 1e-6)

                #         if status == 0:
                #             if abs(ocp.ocp_solver.get(0, "x")[2] - x_sol[f][2]) > eps:
                #                 print('velocita piu estrema trovata!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                #                 print('v1 max: ', ocp.ocp_solver.get(0, "x")[2], 'v1 used: ', x_sol[f][2])
                #             else:
                #                 print('in realtà sembra tutto ok')

        ran1 = random.choice([-1, 1]) * random.random() * multip
        ran2 = random.choice([-1, 1]) * random.random() * multip
        p = np.array([ran1, 0., ran2, -1, 0.])

        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_min, q_max, 0., 0., 1e-2]))
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_max, q_max, 0., 0., 1e-2]))

        ocp.ocp_solver.reset()

        ocp.ocp_solver.set(ocp.N, 'x', np.array([(q_min + q_max)/2, q_max, 0., 0., 1e-2]))
        ocp.ocp_solver.set(ocp.N, 'p', p)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N, endpoint=False)):
            x_guess = np.array([(q_min + q_max)/2, (1-tau)*q_min + tau*q_max, 0., 2*(1-tau)*(q_max-q_min), 1e-2]) # v_max
            ocp.ocp_solver.set(i, 'x', x_guess)
            ocp.ocp_solver.set(i, 'p', p)
            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1e-2])) 
            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1e-2])) 

        ocp.ocp_solver.constraints_set(0, "lbx", np.array([q_min, q_min + eps, v_min, v_min, 1e-2]))
        ocp.ocp_solver.constraints_set(0, "ubx", np.array([q_max, q_min + eps, v_max, v_max, 1e-2]))

        status = ocp.ocp_solver.solve()

        print('--------------------------------------------')

        if status == 0:
            print('INITIAL OCP SOLVED')

            x0 = ocp.ocp_solver.get(0, "x")
            u0 = ocp.ocp_solver.get(0, "u")

            x_sol = np.empty((ocp.N+1,5))
            u_sol = np.empty((ocp.N,2))

            dt_sym = 1e-2

            x_sol[0] = x0
            u_sol[0] = u0

            for i in range(1, ocp.N):
                x_sol[i] = ocp.ocp_solver.get(i, "x")
                u_sol[i] = ocp.ocp_solver.get(i, "u")

            x_sol[ocp.N] = ocp.ocp_solver.get(ocp.N, "x")

            # p_mintime = np.array([0., 0., 0., 0., 1.])

            # ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

            # ocp.ocp_solver.reset()

            # for i in range(1, ocp.N):
            #     ocp.ocp_solver.set(i, 'x', x_sol[i])
            #     ocp.ocp_solver.set(i, 'u', u_sol[i])
            #     ocp.ocp_solver.set(i, 'p', p_mintime)
            #     ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, 0., 0.])) 
            #     ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1e-2])) 

            # ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_fin, q_max, 0., 0., 0.]))
            # ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_fin, q_max, 0., 0., 1e-2]))

            # ocp.ocp_solver.set(ocp.N, 'x', x_sol[ocp.N])
            # ocp.ocp_solver.set(ocp.N, 'p', p_mintime)

            # ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x0[0], x0[1], x0[2], x0[3], 0.])) 
            # ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x0[0], x0[1], x0[2], x0[3], 1e-2])) 

            # ocp.ocp_solver.set(0, 'x', x_sol[0])
            # ocp.ocp_solver.set(0, 'u', u_sol[0])
            # ocp.ocp_solver.set(0, 'p', p_mintime)

            # status = ocp.ocp_solver.solve()

            # if status == 0:
            #     x_sol[0] = ocp.ocp_solver.get(0, "x")
            #     u_sol[0] = ocp.ocp_solver.get(0, "u")

            #     for i in range(1, ocp.N):
            #         x_sol[i] = ocp.ocp_solver.get(i, "x")
            #         u_sol[i] = ocp.ocp_solver.get(i, "u")

            #     x_sol[ocp.N] = ocp.ocp_solver.get(ocp.N, "x")

            #     dt_sym = ocp.ocp_solver.get(0, "x")[4]

            #     print('MIN TIME SOLVED')

            x0 = np.copy(x_sol[0])
            x1_eps = x0[0] - eps * p[0] / (q_max - q_min)
            x2_eps = x0[1] - eps * p[1] / (q_max - q_min)
            x3_eps = x0[2] - eps * p[2] / (2*v_max)
            x4_eps = x0[3] - eps * p[3] / (2*v_max)
            x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

            X_save = np.append(X_save, [[x_sol[ocp.N][0], x_sol[ocp.N][1] + eps, x_sol[ocp.N][2], x_sol[ocp.N][3], 1, 0]], axis = 0)
            X_save = np.append(X_save, [[x_sol[ocp.N][0], x_sol[ocp.N][1], x_sol[ocp.N][2], x_sol[ocp.N][3], 0, 1]], axis = 0)

            xv_state = np.full((ocp.N+1,1),2)
            xv_state[ocp.N] = 1

            if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                is_x_at_limit = True
                print('Stato inziale al limite')
                xv_state[0] = 1
            else:
                is_x_at_limit = False
                print('Stato inziale non al limite')
                xv_state[0] = 0

            X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 1, 0]], axis = 0)
            X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 0, 1]], axis = 0)

            for f in range(1, ocp.N):

                print('Stato ', f)

                if is_x_at_limit:
                    if x_sol[f][0] > q_max - eps or x_sol[f][0] < q_min + eps or x_sol[f][1] > q_max - eps or x_sol[f][1] < q_min + eps or x_sol[f][2] > v_max - eps or x_sol[f][2] < v_min + eps or x_sol[f][3] > v_max - eps or x_sol[f][3] < v_min + eps:
                        # sono su dX
                        is_x_at_limit = True
                        print('Sono su dX, ottenuto controllando i limiti di stato')
                        xv_state[f] = 1

                        x_sym = np.copy(x_sol[f][:4])

                        if x_sol[f][0] > q_max - eps:
                            x_sym[0] = q_max + eps
                        if x_sol[f][0] < q_min + eps:
                            x_sym[0] = q_min - eps
                        if x_sol[f][1] > q_max - eps:
                            x_sym[1] = q_max + eps
                        if x_sol[f][1] < q_min + eps:
                            x_sym[1] = q_min - eps
                        if x_sol[f][2] > v_max - eps:
                            x_sym[2] = v_max + eps
                        if x_sol[f][2] < v_min + eps:
                            x_sym[2] = v_min - eps
                        if x_sol[f][3] > v_max - eps:
                            x_sym[3] = v_max + eps
                        if x_sol[f][3] < v_min + eps:
                            x_sym[3] = v_min - eps
                    else:
                        # sono o su dV o in V
                        is_x_at_limit = False

                        if x_sol[f-1][1] > q_max - eps or x_sol[f-1][1] < q_min + eps:
                            print('Distacco dal limite avvenuto su q2, interrompo')
                            break

                        ocp.ocp_solver.reset()

                        p = np.array([0., 0., 0., -1., 0.])

                        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

                        ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], dt_sym])) 
                        ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], v_max, dt_sym])) 
                        ocp.ocp_solver.set(0, 'x', x_sol[f])
                        ocp.ocp_solver.set(0, 'u', u_sol[f])
                        ocp.ocp_solver.set(0, 'p', p)

                        for i in range(ocp.N - f, ocp.N+1):
                            ocp.ocp_solver.set(i, 'x', x_sol[ocp.N])
                            ocp.ocp_solver.set(i, 'p', p)
                            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                        ocp.ocp_solver.constraints_set(ocp.N - f, "lbx", x_sol[ocp.N])
                        ocp.ocp_solver.constraints_set(ocp.N - f, "ubx", x_sol[ocp.N])

                        for i in range(1, ocp.N - f):
                            ocp.ocp_solver.set(i, 'x', x_sol[i+f])
                            ocp.ocp_solver.set(i, 'u', u_sol[i+f])
                            ocp.ocp_solver.set(i, 'p', p)
                            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                        ocp.ocp_solver.options_set('qp_tol_stat', 1e-2)

                        status = ocp.ocp_solver.solve()

                        ocp.ocp_solver.options_set('qp_tol_stat', 1e-6)

                        if status == 0:
                            print('INTERMEDIATE OCP SOLVED')

                            if ocp.ocp_solver.get(0, "x")[3] > x_sol[f][3] + 1e-2:
                                print('new vel: ', ocp.ocp_solver.get(0, "x")[3], 'old vel: ', x_sol[f][3])
                                break

                                # # sono o su dv o su dx
                                # for i in range(f, ocp.N):
                                #     x_sol[i] = ocp.ocp_solver.get(i-f, "x")
                                #     u_sol[i] = ocp.ocp_solver.get(i-f, "u")

                                # x_sym = np.copy(x_sol[f])
                                # x1_eps = x_sym[0] - eps * p[0]
                                # x2_eps = x_sym[1] - eps * p[1]
                                # x3_eps = x_sym[2] - eps * p[2]
                                # x4_eps = x_sym[3] - eps * p[3]
                                # x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

                                # if x_sym[0] > q_max - eps or x_sym[0] < q_min + eps or x_sym[1] > q_max - eps or x_sym[1] < q_min + eps or x_sym[2] > v_max - eps or x_sym[2] < v_min+ eps or x_sym[3] > v_max - eps or x_sym[3] < v_min + eps:
                                #     print('Sono su dX, ottenuto tramite la nuova soluzione')
                                #     print(x_sol[f], x_sym)
                                #     is_x_at_limit = True
                                #     xv_state[f] = 1
                                # else:
                                #     print('Sono su dV, ottenuto tramite la nuova soluzione')
                                #     print(x_sol[f], x_sym)
                                #     is_x_at_limit = False
                                #     xv_state[f] = 0
                            else:
                                print('Sono su dV, ottenuto tramite la nuova soluzione')
                                print(x_sol[f], x_sym)
                                is_x_at_limit = False
                                xv_state[f] = 0

                                # sono su dv, posso simulare
                                x_sym = np.copy(x_sol[f][:4])
                                x1_eps = x_sym[0] - eps * p[0] / (q_max - q_min)
                                x2_eps = x_sym[1] - eps * p[1] / (q_max - q_min)
                                x3_eps = x_sym[2] - eps * p[2] / (2*v_max)
                                x4_eps = x_sym[3] - eps * p[3] / (2*v_max)
                                x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

                                if x_sym[3] > v_max:
                                    x_sym[3] = v_max
                        else:
                            print('INTERMEDIATE OCP FAILED')
                            break
                        
                else:
                    u_sym = np.copy(u_sol[f-1])
                    sim.acados_integrator.set("u", u_sym)
                    sim.acados_integrator.set("x", x_sym)
                    sim.acados_integrator.set("T", dt_sym)
                    status = sim.acados_integrator.solve()
                    x_sym = sim.acados_integrator.get("x")

                    if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                        # sono su dX
                        print('Sono su dX, ottenuto tramite la simulazione')
                        is_x_at_limit = True
                        xv_state[f] = 1
                    else:
                        # sono su dV
                        print('Sono su dV, ottenuto tramite la simulazione')
                        is_x_at_limit = False
                        xv_state[f] = 0

                X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 1, 0]], axis = 0)
                X_save = np.append(X_save, [[x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], 0, 1]], axis = 0)

                # with torch.no_grad():
                #     out_v = model((torch.from_numpy(np.float32([x_sol[f][:4]])).to(device) - mean) / std).numpy()
                #     out_uv = model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std).numpy()
                #     # if out_uv[0,0] < out_v[0,0]: # or np.argmax(out_uv, axis=1) == 1:
                #     #     print('3 set, at the limit, x: ', x_sol[f], out_v)
                #     #     print('3 set, at the limit, x_sym: ', x_sym, out_uv)

                #     if out_uv[0,1] > 5:
                #         print('Possibile errore!')
                #         print('3 set, at the limit, x: ', x_sol[f], out_v)
                #         print('3 set, at the limit, x_sym: ', x_sym, out_uv)

                #         # u_sym = np.copy(u_sol[f-1])
                            
                #         # if u_sym[1] < -tau_max + eps:
                #         #     u_sym[1] = u_sym[1] + eps
                #         #     tlim = True
                #         # else:
                #         #     if u_sym[1] > tau_max - eps:
                #         #         u_sym[1] = u_sym[1] - eps
                #         #         tlim = True
                #         #     else:
                #         #         if u_sym[0] > tau_max - eps:
                #         #             u_sym[0] = u_sym[0] - eps
                #         #             tlim = True
                #         #         else:
                #         #             if u_sym[0] < -tau_max + eps:
                #         #                 u_sym[0] = u_sym[0] + eps
                #         #                 tlim = True
                #         #             else:
                #         #                 print('no torque at limit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                #         #                 tlim = False

                #         # if tlim:
                #         #     sim.acados_integrator.set("u", u_sym)
                #         #     sim.acados_integrator.set("x", x_sym)
                #         #     sim.acados_integrator.set("T", dt_sym)
                #         #     status = sim.acados_integrator.solve()
                #         #     x_sym = sim.acados_integrator.get("x")

                #         #     isout = False

                #         #     if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                #         #         print('tutto ok')
                #         #         isout = True
                #         #     else:
                #         #         for i in range(f+1, ocp.N):
                #         #             u_sym = np.copy(u_sol[i-1])

                #         #             sim.acados_integrator.set("u", u_sym)
                #         #             sim.acados_integrator.set("x", x_sym)
                #         #             sim.acados_integrator.set("T", dt_sym)
                #         #             status = sim.acados_integrator.solve()
                #         #             x_sym = sim.acados_integrator.get("x")

                #         #             if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                #         #                 print('tutto ok')
                #         #                 isout = True
                #         #                 break

                #         #         if isout == False:
                #         #             print('ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                #         p = np.array([0., 0., 0., -1., 0.])

                #         ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

                #         ocp.ocp_solver.reset()

                #         for i in range(ocp.N - f, ocp.N+1):
                #             ocp.ocp_solver.set(i, 'x', np.array([q_fin, q_max, 0., 0., dt_sym]))
                #             ocp.ocp_solver.set(i, 'p', p)

                #             ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                #             ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                #         ocp.ocp_solver.constraints_set(ocp.N - f, "lbx", np.array([q_fin, q_max, 0., 0., dt_sym]))
                #         ocp.ocp_solver.constraints_set(ocp.N - f, "ubx", np.array([q_fin, q_max, 0., 0., dt_sym]))

                #         for i in range(1, ocp.N - f):
                #             ocp.ocp_solver.set(i, 'x', x_sol[i+f])
                #             ocp.ocp_solver.set(i, 'u', u_sol[i+f])
                #             ocp.ocp_solver.set(i, 'p', p)

                #             ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, 0., dt_sym])) 
                #             ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                #         ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], dt_sym])) 
                #         ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], v_max, dt_sym])) 

                #         ocp.ocp_solver.set(0, 'x', x_sol[f])
                #         ocp.ocp_solver.set(0, 'u', u_sol[f])
                #         ocp.ocp_solver.set(0, 'p', p)

                #         ocp.ocp_solver.options_set('qp_tol_stat', eps)

                #         status = ocp.ocp_solver.solve()

                #         ocp.ocp_solver.options_set('qp_tol_stat', 1e-6)

                #         if status == 0:
                #             if abs(ocp.ocp_solver.get(0, "x")[3] - x_sol[f][3]) > eps:
                #                 print('velocita piu estrema trovata!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                #                 print('v2 max: ', ocp.ocp_solver.get(0, "x")[3], 'v2 used: ', x_sol[f][3])
                #             else:
                #                 print('in realtà sembra tutto ok')

        ran1 = random.choice([-1, 1]) * random.random() * multip
        ran2 = random.choice([-1, 1]) * random.random() * multip
        p = np.array([ran1, 0., ran2, 1., 0.])

        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_min, q_min, 0., 0., 1e-2]))
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_max, q_min, 0., 0., 1e-2]))

        ocp.ocp_solver.reset()

        ocp.ocp_solver.set(ocp.N, 'x', np.array([(q_min+q_max)/2, q_min, 0., 0., 1e-2]))
        ocp.ocp_solver.set(ocp.N, 'p', p)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N, endpoint=False)):
            x_guess = np.array([(q_min+q_max)/2, (1-tau)*q_max + tau*q_min, 0.,  -2*(1-tau)*(q_max-q_min), 1e-2]) # v_min
            ocp.ocp_solver.set(i, 'x', x_guess)
            ocp.ocp_solver.set(i, 'p', p)
            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1e-2])) 
            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1e-2])) 

        ocp.ocp_solver.constraints_set(0, "lbx", np.array([q_min, q_max - eps, v_min, v_min, 1e-2]))
        ocp.ocp_solver.constraints_set(0, "ubx", np.array([q_max, q_max - eps, v_max, v_max, 1e-2]))

        status = ocp.ocp_solver.solve()

        print('--------------------------------------------')

        if status == 0:
            print('INITIAL OCP SOLVED')

            x0 = ocp.ocp_solver.get(0, "x")
            u0 = ocp.ocp_solver.get(0, "u")

            x_sol = np.empty((ocp.N+1,5))
            u_sol = np.empty((ocp.N,2))

            dt_sym = 1e-2

            x_sol[0] = x0
            u_sol[0] = u0

            for i in range(1, ocp.N):
                x_sol[i] = ocp.ocp_solver.get(i, "x")
                u_sol[i] = ocp.ocp_solver.get(i, "u")

            x_sol[ocp.N] = ocp.ocp_solver.get(ocp.N, "x")

            # p_mintime = np.array([0., 0., 0., 0., 1.])

            # ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

            # ocp.ocp_solver.reset()

            # for i in range(1, ocp.N):
            #     ocp.ocp_solver.set(i, 'x', x_sol[i])
            #     ocp.ocp_solver.set(i, 'u', u_sol[i])
            #     ocp.ocp_solver.set(i, 'p', p_mintime)
            #     ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 0.])) 
            #     ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, 0., 1e-2])) 

            # ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_fin, q_min, 0., 0., 0.]))
            # ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_fin, q_min, 0., 0., 1e-2]))

            # ocp.ocp_solver.set(ocp.N, 'x', x_sol[ocp.N])
            # ocp.ocp_solver.set(ocp.N, 'p', p_mintime)

            # ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x0[0], x0[1], x0[2], x0[3], 0.])) 
            # ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x0[0], x0[1], x0[2], x0[3], 1e-2])) 

            # ocp.ocp_solver.set(0, 'x', x_sol[0])
            # ocp.ocp_solver.set(0, 'u', u_sol[0])
            # ocp.ocp_solver.set(0, 'p', p_mintime)

            # status = ocp.ocp_solver.solve()

            # if status == 0:
            #     x_sol[0] = ocp.ocp_solver.get(0, "x")
            #     u_sol[0] = ocp.ocp_solver.get(0, "u")

            #     for i in range(1, ocp.N):
            #         x_sol[i] = ocp.ocp_solver.get(i, "x")
            #         u_sol[i] = ocp.ocp_solver.get(i, "u")

            #     x_sol[ocp.N] = ocp.ocp_solver.get(ocp.N, "x")

            #     dt_sym = ocp.ocp_solver.get(0, "x")[4]

            #     print('MIN TIME SOLVED')

            x0 = np.copy(x_sol[0])
            x1_eps = x0[0] - eps * p[0] / (q_max - q_min)
            x2_eps = x0[1] - eps * p[1] / (q_max - q_min)
            x3_eps = x0[2] - eps * p[2] / (2*v_max)
            x4_eps = x0[3] - eps * p[3] / (2*v_max)
            x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

            X_save = np.append(X_save, [[x_sol[ocp.N][0], x_sol[ocp.N][1] - eps, x_sol[ocp.N][2], x_sol[ocp.N][3], 1, 0]], axis = 0)
            X_save = np.append(X_save, [[x_sol[ocp.N][0], x_sol[ocp.N][1], x_sol[ocp.N][2], x_sol[ocp.N][3], 0, 1]], axis = 0)

            xv_state = np.full((ocp.N+1,1),2)
            xv_state[ocp.N] = 1

            if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                is_x_at_limit = True
                print('Stato inziale al limite')
                xv_state[0] = 1
            else:
                is_x_at_limit = False
                print('Stato inziale non al limite')
                xv_state[0] = 0

            X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 1, 0]], axis = 0)
            X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 0, 1]], axis = 0)

            for f in range(1, ocp.N):

                print('Stato ', f)

                if is_x_at_limit:
                    if x_sol[f][0] > q_max - eps or x_sol[f][0] < q_min + eps or x_sol[f][1] > q_max - eps or x_sol[f][1] < q_min + eps or x_sol[f][2] > v_max - eps or x_sol[f][2] < v_min + eps or x_sol[f][3] > v_max - eps or x_sol[f][3] < v_min + eps:
                        # sono su dX
                        is_x_at_limit = True
                        print('Sono su dX, ottenuto controllando i limiti di stato')
                        xv_state[f] = 1

                        x_sym = np.copy(x_sol[f][:4])

                        if x_sol[f][0] > q_max - eps:
                            x_sym[0] = q_max + eps
                        if x_sol[f][0] < q_min + eps:
                            x_sym[0] = q_min - eps
                        if x_sol[f][1] > q_max - eps:
                            x_sym[1] = q_max + eps
                        if x_sol[f][1] < q_min + eps:
                            x_sym[1] = q_min - eps
                        if x_sol[f][2] > v_max - eps:
                            x_sym[2] = v_max + eps
                        if x_sol[f][2] < v_min + eps:
                            x_sym[2] = v_min - eps
                        if x_sol[f][3] > v_max - eps:
                            x_sym[3] = v_max + eps
                        if x_sol[f][3] < v_min + eps:
                            x_sym[3] = v_min - eps
                    else:
                        # sono o su dV o in V
                        is_x_at_limit = False

                        if x_sol[f-1][1] > q_max - eps or x_sol[f-1][1] < q_min + eps:
                            print('Distacco dal limite avvenuto su q2, interrompo')
                            break

                        ocp.ocp_solver.reset()

                        p = np.array([0., 0., 0., 1., 0.])

                        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

                        ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], v_min, dt_sym])) 
                        ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], dt_sym])) 
                        ocp.ocp_solver.set(0, 'x', x_sol[f])
                        ocp.ocp_solver.set(0, 'u', u_sol[f])
                        ocp.ocp_solver.set(0, 'p', p)

                        for i in range(ocp.N - f, ocp.N+1):
                            ocp.ocp_solver.set(i, 'x', x_sol[ocp.N])
                            ocp.ocp_solver.set(i, 'p', p)
                            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                        ocp.ocp_solver.constraints_set(ocp.N - f, "lbx", x_sol[ocp.N])
                        ocp.ocp_solver.constraints_set(ocp.N - f, "ubx", x_sol[ocp.N])

                        for i in range(1, ocp.N - f):
                            ocp.ocp_solver.set(i, 'x', x_sol[i+f])
                            ocp.ocp_solver.set(i, 'u', u_sol[i+f])
                            ocp.ocp_solver.set(i, 'p', p)
                            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                        ocp.ocp_solver.options_set('qp_tol_stat', 1e-2)

                        status = ocp.ocp_solver.solve()

                        ocp.ocp_solver.options_set('qp_tol_stat', 1e-6)

                        if status == 0:
                            print('INTERMEDIATE OCP SOLVED')

                            if ocp.ocp_solver.get(0, "x")[3] < x_sol[f][3] - 1e-2:
                                print('new vel: ', ocp.ocp_solver.get(0, "x")[3], 'old vel: ', x_sol[f][3])
                                break

                                # # sono o su dv o su dx
                                # for i in range(f, ocp.N):
                                #     x_sol[i] = ocp.ocp_solver.get(i-f, "x")
                                #     u_sol[i] = ocp.ocp_solver.get(i-f, "u")

                                # x_sym = np.copy(x_sol[f])
                                # x1_eps = x_sym[0] - eps * p[0]
                                # x2_eps = x_sym[1] - eps * p[1]
                                # x3_eps = x_sym[2] - eps * p[2]
                                # x4_eps = x_sym[3] - eps * p[3]
                                # x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

                                # if x_sym[0] > q_max - eps or x_sym[0] < q_min + eps or x_sym[1] > q_max - eps or x_sym[1] < q_min + eps or x_sym[2] > v_max - eps or x_sym[2] < v_min+ eps or x_sym[3] > v_max - eps or x_sym[3] < v_min + eps:
                                #     print('Sono su dX, ottenuto tramite la nuova soluzione')
                                #     print(x_sol[f], x_sym)
                                #     is_x_at_limit = True
                                #     xv_state[f] = 1
                                # else:
                                #     print('Sono su dV, ottenuto tramite la nuova soluzione')
                                #     print(x_sol[f], x_sym)
                                #     is_x_at_limit = False
                                #     xv_state[f] = 0
                            else:
                                print('Sono su dV, ottenuto tramite la nuova soluzione')
                                print(x_sol[f], x_sym)
                                is_x_at_limit = False
                                xv_state[f] = 0

                                # sono su dv, posso simulare
                                x_sym = np.copy(x_sol[f][:4])
                                x1_eps = x_sym[0] - eps * p[0] / (q_max - q_min)
                                x2_eps = x_sym[1] - eps * p[1] / (q_max - q_min)
                                x3_eps = x_sym[2] - eps * p[2] / (2*v_max)
                                x4_eps = x_sym[3] - eps * p[3] / (2*v_max)
                                x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

                                if x_sym[3] < v_min:
                                    x_sym[3] = v_min
                        else:
                            print('INTERMEDIATE OCP FAILED')
                            break
                        
                else:
                    u_sym = np.copy(u_sol[f-1])
                    sim.acados_integrator.set("u", u_sym)
                    sim.acados_integrator.set("x", x_sym)
                    sim.acados_integrator.set("T", dt_sym)
                    status = sim.acados_integrator.solve()
                    x_sym = sim.acados_integrator.get("x")

                    if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                        # sono su dX
                        print('Sono su dX, ottenuto tramite la simulazione')
                        is_x_at_limit = True
                        xv_state[f] = 1
                    else:
                        # sono su dV
                        print('Sono su dV, ottenuto tramite la simulazione')
                        is_x_at_limit = False
                        xv_state[f] = 0

                X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 1, 0]], axis = 0)
                X_save = np.append(X_save, [[x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], 0, 1]], axis = 0)

                # with torch.no_grad():
                #     out_v = model((torch.from_numpy(np.float32([x_sol[f][:4]])).to(device) - mean) / std).numpy()
                #     out_uv = model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std).numpy()
                #     # if out_uv[0,0] < out_v[0,0]: # or np.argmax(out_uv, axis=1) == 1:
                #     #     print('4 set, at the limit, x: ', x_sol[f], out_v)
                #     #     print('4 set, at the limit, x_sym: ', x_sym, out_uv)

                #     if out_uv[0,1] > 5:
                #         print('Possibile errore!')
                #         print('4 set, at the limit, x: ', x_sol[f], out_v)
                #         print('4 set, at the limit, x_sym: ', x_sym, out_uv)

                #         # u_sym = np.copy(u_sol[f-1])
                            
                #         # if u_sym[0] > tau_max - eps:
                #         #     u_sym[0] = u_sym[0] - eps
                #         #     tlim = True
                #         # else:
                #         #     if u_sym[0] < -tau_max + eps:
                #         #         u_sym[0] = u_sym[0] + eps
                #         #         tlim = True
                #         #     else:
                #         #         if u_sym[1] > tau_max - eps:
                #         #             u_sym[1] = u_sym[1] - eps
                #         #             tlim = True
                #         #         else:
                #         #             if u_sym[1] < -tau_max + eps:
                #         #                 u_sym[1] = u_sym[1] + eps
                #         #                 tlim = True
                #         #             else:
                #         #                 print('no torque at limit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                #         #                 tlim = False

                #         # if tlim:
                #         #     sim.acados_integrator.set("u", u_sym)
                #         #     sim.acados_integrator.set("x", x_sym)
                #         #     sim.acados_integrator.set("T", dt_sym)
                #         #     status = sim.acados_integrator.solve()
                #         #     x_sym = sim.acados_integrator.get("x")

                #         #     isout = False

                #         #     if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                #         #         print('tutto ok')
                #         #         isout = True
                #         #     else:
                #         #         for i in range(f+1, ocp.N):
                #         #             u_sym = np.copy(u_sol[i-1])

                #         #             sim.acados_integrator.set("u", u_sym)
                #         #             sim.acados_integrator.set("x", x_sym)
                #         #             sim.acados_integrator.set("T", dt_sym)
                #         #             status = sim.acados_integrator.solve()
                #         #             x_sym = sim.acados_integrator.get("x")

                #         #             if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                #         #                 print('tutto ok')
                #         #                 isout = True
                #         #                 break

                #         #         if isout == False:
                #         #             print('ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                #         p = np.array([0., 0., 0., 1., 0.])

                #         ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

                #         ocp.ocp_solver.reset()

                #         for i in range(ocp.N - f, ocp.N+1):
                #             ocp.ocp_solver.set(i, 'x', np.array([q_fin, q_min, 0., 0., dt_sym]))
                #             ocp.ocp_solver.set(i, 'p', p)

                #             ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                #             ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, dt_sym])) 

                #         ocp.ocp_solver.constraints_set(ocp.N - f, "lbx", np.array([q_fin, q_min, 0., 0., dt_sym]))
                #         ocp.ocp_solver.constraints_set(ocp.N - f, "ubx", np.array([q_fin, q_min, 0., 0., dt_sym]))

                #         for i in range(1, ocp.N - f):
                #             ocp.ocp_solver.set(i, 'x', x_sol[i+f])
                #             ocp.ocp_solver.set(i, 'u', u_sol[i+f])
                #             ocp.ocp_solver.set(i, 'p', p)

                #             ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, dt_sym])) 
                #             ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, 0., dt_sym])) 

                #         ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], v_min, dt_sym])) 
                #         ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], dt_sym])) 

                #         ocp.ocp_solver.set(0, 'x', x_sol[f])
                #         ocp.ocp_solver.set(0, 'u', u_sol[f])
                #         ocp.ocp_solver.set(0, 'p', p)

                #         ocp.ocp_solver.options_set('qp_tol_stat', eps)

                #         status = ocp.ocp_solver.solve()

                #         ocp.ocp_solver.options_set('qp_tol_stat', 1e-6)

                #         if status == 0:
                #             if abs(ocp.ocp_solver.get(0, "x")[3] - x_sol[f][3]) > eps:
                #                 print('velocita piu estrema trovata!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                #                 print('v2 max: ', ocp.ocp_solver.get(0, "x")[3], 'v2 used: ', x_sol[f][3])
                #             else:
                #                 print('in realtà sembra tutto ok')

    # np.save('data_reverse_100_new.npy', np.asarray(X_save))

    # X_save = np.load('data_reverse_100_new.npy')

    ind = random.sample(range(len(X_save)), int(len(X_save)*0.7))
    X_train = np.array([X_save[i] for i in ind])
    X_test = np.array([X_save[i] for i in range(len(X_save)) if i not in ind])

    # clf.fit(X_train[:,:4], X_train[:,5])

    # print("Accuracy training:", metrics.accuracy_score(X_train[:,5], clf.predict(X_train[:,:4])))
    # print("Accuracy testing:", metrics.accuracy_score(X_test[:,5], clf.predict(X_test[:,:4])))

    it = 1
    val = 1
    val_prev = 2

    # mean_rev, std_rev = torch.mean(torch.Tensor(X_train[:,:4])).to(device), torch.std(torch.Tensor(X_train[:,:4])).to(device)
    mean_rev, std_rev = mean_al, std_al

    # Train the model
    while val > loss_stop:
        ind = random.sample(range(len(X_train)), n_minibatch)

        X_iter_tensor = torch.Tensor([X_train[i][:4] for i in ind]).to(device)
        y_iter_tensor = torch.Tensor([X_train[i][4:] for i in ind]).to(device)
        X_iter_tensor = (X_iter_tensor - mean_rev) / std_rev

        # Zero the gradients
        for param in model_rev.parameters():
            param.grad = None

        # Forward pass
        outputs = model_rev(X_iter_tensor)
        loss = criterion(outputs, y_iter_tensor)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        val = beta * val + (1 - beta) * loss.item()

        it += 1

        if it % B == 0: 
            val_prev = (val + val_prev)/2
            print(val)

        if it >= it_max and val > val_prev - 1e-3:
            break

    # print(it, val)

    with torch.no_grad():
        from torchmetrics.classification import BinaryAccuracy
        target = torch.Tensor(X_train[:,5])
        inp = torch.Tensor(X_train[:,:4])
        inp = (inp - mean_rev) / std_rev
        out = model_rev(inp)
        preds = torch.Tensor(np.argmax(out.numpy(), axis=1))
        metric = BinaryAccuracy()

        print('Accuracy nn train', metric(preds, target))

    with torch.no_grad():
        target = torch.Tensor(X_test[:,5])
        inp = torch.Tensor(X_test[:,:4])
        inp = (inp - mean_rev) / std_rev
        out = model_rev(inp)
        preds = torch.Tensor(np.argmax(out.numpy(), axis=1))
        metric = BinaryAccuracy()

        print('Accuracy nn test', metric(preds, target))

    data_al = np.load('data_al_novellim.npy')

    data_boundary = np.array([data_al[i] for i in range(len(data_al)) if abs(model_al((torch.Tensor(data_al[i,:4]) - mean_al) / std_al)[0]) < 5])

    with torch.no_grad():
        target = torch.Tensor(data_al[:,5])
        inp = torch.Tensor(data_al[:,:4])
        inp = (inp - mean_rev) / std_rev
        out = model_rev(inp)
        preds = torch.Tensor(np.argmax(out.numpy(), axis=1))
        metric = BinaryAccuracy()

        print('Accuracy nn AL', metric(preds, target))

    with torch.no_grad():
        target = torch.Tensor(data_boundary[:,5])
        inp = torch.Tensor(data_boundary[:,:4])
        inp = (inp - mean_rev) / std_rev
        out = model_rev(inp)
        preds = torch.Tensor(np.argmax(out.numpy(), axis=1))
        metric = BinaryAccuracy()

        print('Accuracy nn AL boundary', metric(preds, target))

    # print("Accuracy AL data:", metrics.accuracy_score(data_al[:,5], clf.predict(data_al[:,:4])))

    # print("Accuracy AL data boundary:", metrics.accuracy_score(data_boundary[:,5], clf.predict(data_boundary[:,:4])))

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
        inp = (inp - mean_rev) / std_rev
        out = model_rev(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        xit = []
        yit = []
        cit = []
        for i in range(len(X_save)):
            if (
                norm(X_save[i][0] - (q_min + q_max) / 2) < 0.1
                and norm(X_save[i][2]) < 1.
            ):
                xit.append(X_save[i][1])
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
        inp = (inp - mean_rev) / std_rev
        out = model_rev(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        xit = []
        yit = []
        cit = []
        for i in range(len(X_save)):
            if (
                norm(X_save[i][1] - (q_min + q_max) / 2) < 0.1
                and norm(X_save[i][3]) < 1.
            ):
                xit.append(X_save[i][0])
                yit.append(X_save[i][2])
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
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.grid()
        plt.title("First actuator")

    # # Plot the results:
    # plt.figure()
    # h = 0.02
    # xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    # xrav = xx.ravel()
    # yrav = yy.ravel()
    # Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
    #                 np.zeros(yrav.shape[0]), yrav])
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # cit = []
    # for i in range(len(X_save)):
    #     if (
    #         norm(X_save[i][0] - (q_min + q_max) / 2) < 0.1
    #         and norm(X_save[i][2]) < 1.
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
    # plt.title("Second actuator")

    # plt.figure()
    # Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
    #                 np.zeros(yrav.shape[0])])
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # cit = []
    # for i in range(len(X_save)):
    #     if (
    #         norm(X_save[i][1] - (q_min + q_max) / 2) < 0.1
    #         and norm(X_save[i][3]) < 1.
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
    # plt.title("First actuator")

    # # Plot the results:
    # plt.figure()
    # h = 0.02
    # xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    # xrav = xx.ravel()
    # yrav = yy.ravel()
    # Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
    #                 np.zeros(yrav.shape[0]), yrav])
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # cit = []
    # for i in range(len(X_save)):
    #     if (
    #         norm(X_save[i][0] - (q_min + q_max) / 2) < 0.2
    #         and norm(X_save[i][2]) < 2.
    #     ):
    #         xit.append(X_save[i][1])
    #         yit.append(X_save[i][3])
    #         if X_save[i][4] < 0.5:
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
    # plt.title("Second actuator")

    # plt.figure()
    # Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
    #                 np.zeros(yrav.shape[0])])
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # cit = []
    # for i in range(len(X_save)):
    #     if (
    #         norm(X_save[i][1] - (q_min + q_max) / 2) < 0.2
    #         and norm(X_save[i][3]) < 2.
    #     ):
    #         xit.append(X_save[i][0])
    #         yit.append(X_save[i][2])
    #         if X_save[i][4] < 0.5:
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
    # plt.title("First actuator")

    # # Plot the results:
    # plt.figure()
    # h = 0.02
    # xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    # xrav = xx.ravel()
    # yrav = yy.ravel()
    # Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
    #                 np.zeros(yrav.shape[0]), yrav])
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # cit = []
    # for i in range(len(X_save)):
    #     if (
    #         norm(X_save[i][0] - (q_min + q_max) / 2) < 0.3
    #         and norm(X_save[i][2]) < 3.
    #     ):
    #         xit.append(X_save[i][1])
    #         yit.append(X_save[i][3])
    #         if X_save[i][4] < 0.5:
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
    # plt.title("Second actuator")

    # plt.figure()
    # Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
    #                 np.zeros(yrav.shape[0])])
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # cit = []
    # for i in range(len(X_save)):
    #     if (
    #         norm(X_save[i][1] - (q_min + q_max) / 2) < 0.3
    #         and norm(X_save[i][3]) < 3.
    #     ):
    #         xit.append(X_save[i][0])
    #         yit.append(X_save[i][2])
    #         if X_save[i][4] < 0.5:
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
    # plt.title("First actuator")

    # # Plot the results:
    # plt.figure()
    # h = 0.02
    # xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    # xrav = xx.ravel()
    # yrav = yy.ravel()
    # Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
    #                 np.zeros(yrav.shape[0]), yrav])
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # cit = []
    # for i in range(len(X_save)):
    #     if (
    #         norm(X_save[i][0] - (q_min + q_max) / 2) < 0.4
    #         and norm(X_save[i][2]) < 4.
    #     ):
    #         xit.append(X_save[i][1])
    #         yit.append(X_save[i][3])
    #         if X_save[i][4] < 0.5:
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
    # plt.title("Second actuator")

    # plt.figure()
    # Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
    #                 np.zeros(yrav.shape[0])])
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # cit = []
    # for i in range(len(X_save)):
    #     if (
    #         norm(X_save[i][1] - (q_min + q_max) / 2) < 0.4
    #         and norm(X_save[i][3]) < 4.
    #     ):
    #         xit.append(X_save[i][0])
    #         yit.append(X_save[i][2])
    #         if X_save[i][4] < 0.5:
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
    # plt.title("First actuator")

    # # Plot the results:
    # h = 0.02
    # xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    # xrav = xx.ravel()
    # yrav = yy.ravel()

    # plt.figure()
    # Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
    #                 np.zeros(yrav.shape[0]), yrav])
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # cit = []
    # for i in range(len(X_save)):
    #     if (
    #         norm(X_save[i][0] - (q_min + q_max) / 2) < 0.5
    #         and norm(X_save[i][2]) < 5.
    #     ):
    #         xit.append(X_save[i][1])
    #         yit.append(X_save[i][3])
    #         if X_save[i][4] < 0.5:
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
    # plt.title("Second actuator")

    # plt.figure()
    # Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
    #                 np.zeros(yrav.shape[0])])
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # cit = []
    # for i in range(len(X_save)):
    #     if (
    #         norm(X_save[i][1] - (q_min + q_max) / 2) < 0.5
    #         and norm(X_save[i][3]) < 5.
    #     ):
    #         xit.append(X_save[i][0])
    #         yit.append(X_save[i][2])
    #         if X_save[i][4] < 0.5:
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
    # plt.title("First actuator")

    print("Execution time: %s seconds" % (time.time() - start_time))

    plt.show()
