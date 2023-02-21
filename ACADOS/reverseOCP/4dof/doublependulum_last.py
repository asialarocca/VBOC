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
    tau_max = ocp.Cmax

    with torch.no_grad():

        # Hyper-parameters for nn:
        input_size = 4
        hidden_size = 4 * 100
        output_size = 2
        learning_rate = 0.001

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(torch.load('model_2pendulum'))

        mean, std = torch.tensor(1.9635), torch.tensor(3.0036)

    # Initialization of the SVM classifier:
    clf = svm.SVC(C=1e6, kernel='rbf', class_weight='balanced')

    eps = 1e-2
    multip = 1.
    num = 10
    
    X_save = np.array([[(q_min+q_max)/2, (q_min+q_max)/2, 0., 0., 1]])
    
    for _ in range(num):
               
        q_fin = q_min + random.random() * (q_max-q_min)

        ran1 = random.choice([-1, 1]) * random.random() * multip
        ran2 = random.choice([-1, 1]) * random.random() * multip
        p = np.array([0., ran1, -1, ran2, 0.])

        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 0.01))

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_max, q_fin, 0., 0., 1.]))
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_max, q_fin, 0., 0., 1.]))

        ocp.ocp_solver.reset()

        ocp.ocp_solver.set(ocp.N, 'x', np.array([q_max, q_fin, 0., 0., 1.]))
        ocp.ocp_solver.set(ocp.N, 'p', p)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N, endpoint=False)):
            x_guess = np.array([(1-tau)*q_min + tau*q_max, q_fin, v_max, 0., 1.])
            ocp.ocp_solver.set(i, 'x', x_guess)
            ocp.ocp_solver.set(i, 'p', p)
            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1.])) 
            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1.])) 

        ocp.ocp_solver.constraints_set(0, "lbx", np.array([q_min, q_min, v_min, v_min, 1.]))
        ocp.ocp_solver.constraints_set(0, "ubx", np.array([q_min, q_max, v_max, v_max, 1.]))

        status = ocp.ocp_solver.solve()

        if status == 0:
            print('INITIAL OCP SOLVED')

            x0 = ocp.ocp_solver.get(0, "x")
            u0 = ocp.ocp_solver.get(0, "u")

            x_sol = np.empty((ocp.N+1,5))
            u_sol = np.empty((ocp.N,2))

            dt_sym = 1e-2

            x_sol[0] = [x0[0], x0[1], x0[2], x0[3], 1e-2]
            u_sol[0] = u0

            for i in range(1, ocp.N):
                prev_sol = ocp.ocp_solver.get(i, "x")
                x_sol[i] = [prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3], 1e-2]
                u_sol[i] = ocp.ocp_solver.get(i, "u")

            prev_sol = ocp.ocp_solver.get(ocp.N, "x")
            x_sol[ocp.N] = [prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3], 1e-2]

            p_mintime = np.array([0., 0., 0., 0., 1.])

            ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

            ocp.ocp_solver.reset()

            for i in range(1, ocp.N):
                ocp.ocp_solver.set(i, 'x', x_sol[i])
                ocp.ocp_solver.set(i, 'u', u_sol[i])
                ocp.ocp_solver.set(i, 'p', p_mintime)
                ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 0.])) 
                ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1e-2])) 

            ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_max, q_fin, 0., 0., 0.]))
            ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_max, q_fin, 0., 0., 1e-2]))

            ocp.ocp_solver.set(ocp.N, 'x', x_sol[ocp.N])
            ocp.ocp_solver.set(ocp.N, 'p', p_mintime)

            ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x0[0], x0[1], x0[2], x0[3], 0.])) 
            ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x0[0], x0[1], x0[2], x0[3], 1e-2])) 

            ocp.ocp_solver.set(0, 'x', x_sol[0])
            ocp.ocp_solver.set(0, 'u', u_sol[0])
            ocp.ocp_solver.set(0, 'p', p_mintime)

            status = ocp.ocp_solver.solve()

            if status == 0:
                print('MIN TIME SOLVED')

                x_sol[0] = ocp.ocp_solver.get(0, "x")
                u_sol[0] = ocp.ocp_solver.get(0, "u")

                for i in range(1, ocp.N):
                    x_sol[i] = ocp.ocp_solver.get(i, "x")
                    u_sol[i] = ocp.ocp_solver.get(i, "u")

                x_sol[ocp.N] = ocp.ocp_solver.get(ocp.N, "x")

                dt_sym = ocp.ocp_solver.get(0, "x")[4]

            x0 = np.copy(x_sol[0])
            x1_eps = x0[0] - eps * p[0] / (q_max - q_min)
            x2_eps = x0[1] - eps * p[1] / (q_max - q_min)
            x3_eps = x0[2] - eps * p[2] / (2*v_max)
            x4_eps = x0[3] - eps * p[3] / (2*v_max)
            x_sym = np.array([x0[0], x2_eps, x3_eps, x4_eps])

            X_save = np.append(X_save, [[x_sol[ocp.N][0] + 1e-2, x_sol[ocp.N][1], x_sol[ocp.N][2], x_sol[ocp.N][3], 0]], axis = 0)
            X_save = np.append(X_save, [[x_sol[ocp.N][0], x_sol[ocp.N][1], x_sol[ocp.N][2], x_sol[ocp.N][3], 1]], axis = 0)

            X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
            X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)

            for f in range(1, ocp.N):

                if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] > v_max or x_sym[2] < v_min or x_sym[3] > v_max or x_sym[3] < v_min:
                    is_x_at_limit_before = True
                else:
                    is_x_at_limit_before = False

                

                if is_x_at_limit_now == True: 
                    x_sym = np.copy(x_sol[f][:4])

                    if x_sol[f][0] > q_max - 1e-4:
                        x_sym[0] = q_max + 1e-4
                    if x_sol[f][0] < q_min + 1e-4:
                        x_sym[0] = q_min - 1e-4
                    if x_sol[f][1] > q_max - 1e-4:
                        x_sym[1] = q_max + 1e-4
                    if x_sol[f][1] < q_min + 1e-4:
                        x_sym[1] = q_min - 1e-4
                    if x_sol[f][2] > v_max - 1e-4:
                        x_sym[2] = v_max + 1e-4
                    if x_sol[f][2] < v_min + 1e-4:
                        x_sym[2] = v_min - 1e-4
                    if x_sol[f][3] > v_max - 1e-4:
                        x_sym[3] = v_max + 1e-4
                    if x_sol[f][3] < v_min + 1e-4:
                        x_sym[3] = v_min - 1e-4

                    is_x_at_limit_before = is_x_at_limit_now
                else: 
                    if is_x_at_limit_before == True: # sono o su dV o in V, risolvi un nuovo OCP
                        if x_sol[f-1][0] > q_max - 1e-4 or x_sol[f-1][0] < q_min + 1e-4:
                            break

                        if x_sol[f-1][2] > v_max - 1e-4:
                            p = np.array([0., 0., -1., 0., 0.]) # dovrei anche minimizzare il tempo?
                        else:
                            break # i don't know what to do here

                        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), dt_sym))

                        ocp.ocp_solver.reset()

                        for i in range(ocp.N - f, ocp.N+1):
                            ocp.ocp_solver.set(i, 'x', np.array([q_max, q_fin, 0., 0., 1.]))
                            ocp.ocp_solver.set(i, 'p', p)

                            ocp.ocp_solver.constraints_set(i, "lbx", np.array([q_max, q_fin, 0., 0., 1.]))
                            ocp.ocp_solver.constraints_set(i, "ubx", np.array([q_max, q_fin, 0., 0., 1.]))

                        for i in range(1, ocp.N - f):
                            ocp.ocp_solver.set(i, 'x', x_sol[i+f])
                            ocp.ocp_solver.set(i, 'u', u_sol[i+f])
                            ocp.ocp_solver.set(i, 'p', p)
                            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1.])) 
                            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1.])) 

                        ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], 1.])) 
                        ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x_sol[f][0], x_sol[f][1], v_max, x_sol[f][3], 1.])) 

                        ocp.ocp_solver.set(0, 'x', np.array([x_sol[f][0], x_sol[f][1], v_max, x_sol[f][3], 1.]))
                        ocp.ocp_solver.set(0, 'u', u_sol[f])
                        ocp.ocp_solver.set(0, 'p', p)

                        status = ocp.ocp_solver.solve()

                        if status == 0:
                            if ocp.ocp_solver.get(0, "x")[2] == x_sol[f][2]:
                                # sono su dv, posso simulare
                                x_sym = np.copy(x_sol[f][2])
                                x_sym[2] = x_sym[2] - eps * p[2] / (2*v_max)
                                
                                is_x_at_limit_before = False

                                if x_sym[2] < v_min:
                                    x_sym[2] = v_min 
                                if x_sym[2] > v_max:
                                    x_sym[2] = v_max

                                status_in = True
                            else:
                                # sono o su dv o su dx
                                for i in range(f, ocp.N):
                                    x_sol[i] = ocp.ocp_solver.get(i-f, "x")
                                    u_sol[i] = ocp.ocp_solver.get(i-f, "u")

                                x_sym = np.copy(x_sol[f][2])
                                x_sym[2] = x_sym[2] - eps * p[2] / (2*v_max)

                                if x_sol[f][0] > q_max - 1e-4 or x_sol[f][0] < q_min + 1e-4 or x_sol[f][1] > q_max - 1e-4 or x_sol[f][1] < q_min + 1e-4 or x_sol[f][2] > v_max - 1e-4 or x_sol[f][2] < v_min + 1e-4 or x_sol[f][3] > v_max - 1e-4 or x_sol[f][3] < v_min + 1e-4:
                                    is_x_at_limit_before = True
                                else:
                                    is_x_at_limit_before = False

                                    if x_sym[2] < v_min:
                                        x_sym[2] = v_min 
                                    if x_sym[2] > v_max:
                                        x_sym[2] = v_max

                                    status_in = True
                        else:
                            break

                    else: # sono su dV, continua a simulare
                        if x_sym[0] >= q_min and x_sym[0] <= q_max and x_sym[1] >= q_min and x_sym[1] <= q_max and x_sym[2] <= v_max and x_sym[2] >= v_min and x_sym[3] <= v_max and x_sym[3] >= v_min:
                            u_sym = np.copy(u_sol[f-1])
                            sim.acados_integrator.set("u", u_sym)
                            sim.acados_integrator.set("x", x_sym)
                            sim.acados_integrator.set("T", dt_sym)
                            status = sim.acados_integrator.solve()
                            x_sym = sim.acados_integrator.get("x")
                        else:
                            if status_in:
                                status_in = False
                                x_sym_out = np.copy(x_sym)

                            x_sym = np.copy(x_sol[f])

                            if x_sym_out[0] < q_min or x_sym_out[0] > q_max:
                                x_sym[0] = x_sym_out[0]
                            if x_sym_out[2] < v_min or x_sym_out[2] > v_max:
                                x_sym[2] = x_sym_out[2]
                            if x_sym_out[1] < q_min or x_sym_out[1] > q_max:
                                x_sym[1] = x_sym_out[1]
                            if x_sym_out[3] < v_min or x_sym_out[3] > v_max:
                                x_sym[3] = x_sym_out[3]

                X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
                X_save = np.append(X_save, [[x_sol[f][0], x_sol[f][1], x_sol[f][2], x_sol[f][3], 1]], axis = 0)

            t = np.linspace(0, ocp.N*dt_sym, ocp.N+1)

            plt.figure()
            plt.subplot(2, 1, 1)
            line, = plt.step(t, np.append([u_sol[0, 0]], u_sol[:, 0]))
            plt.ylabel('$C1$')
            plt.xlabel('$t$')
            plt.hlines(ocp.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.hlines(-ocp.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.ylim([-1.2*ocp.Cmax, 1.2*ocp.Cmax])
            plt.title('Controls')
            plt.grid()
            plt.subplot(2, 1, 2)
            line, = plt.step(t, np.append([u_sol[0, 1]], u_sol[:, 1]))
            plt.ylabel('$C2$')
            plt.xlabel('$t$')
            plt.hlines(ocp.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.hlines(-ocp.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.ylim([-1.2*ocp.Cmax, 1.2*ocp.Cmax])
            plt.grid()

            plt.figure()
            plt.subplot(4, 1, 1)
            line, = plt.plot(t, x_sol[:, 0])
            plt.ylabel('$theta1$')
            plt.xlabel('$t$')
            plt.hlines(ocp.thetamax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.hlines(ocp.thetamin, t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.title('States')
            plt.grid()
            plt.subplot(4, 1, 2)
            line, = plt.plot(t, x_sol[:, 1])
            plt.ylabel('$theta2$')
            plt.xlabel('$t$')
            plt.hlines(ocp.thetamax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.hlines(ocp.thetamin, t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.grid()
            plt.subplot(4, 1, 3)
            line, = plt.plot(t, x_sol[:, 2])
            plt.ylabel('$dtheta1$')
            plt.xlabel('$t$')
            plt.hlines(ocp.dthetamax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.hlines(-ocp.dthetamax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.grid()
            plt.subplot(4, 1, 4)
            line, = plt.plot(t, x_sol[:, 3])
            plt.ylabel('$dtheta2$')
            plt.xlabel('$t$')
            plt.hlines(ocp.dthetamax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.hlines(-ocp.dthetamax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.grid()



            # if x_sym[0] > q_max or x_sym[0] < q_min or x_sym[1] > q_max or x_sym[1] < q_min or x_sym[2] < v_min or x_sym[2] > v_max or x_sym[3] < v_min or x_sym[3] > v_max:
            #     print('AT THE LIMIT')
                
            #     # with torch.no_grad():
            #     #     out_v = model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std).numpy()
            #     #     out_uv = model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std).numpy()
            #     #     if out_uv[0,0] < out_v[0,0] or np.argmax(out_uv, axis=1) == 1:
            #     #         print('1 set, at the limit, x0: ', x0, out_v)
            #     #         print('1 set, at the limit, x_sym0: ', x_sym, out_uv)

            #     if status == 0:
            #         dt_sym = ocp.ocp_solver.get(0, "x")[4]

            #         x_sol = np.empty((ocp.N+1,4))
            #         u_sol = np.empty((ocp.N,2))

            #         current_val = ocp.ocp_solver.get(0, "x")
            #         x_sol[0] = np.copy(current_val[:4])

            #         # print('x0: ', x_sol[0])

            #         check_init = True

            #         for i in range(1, ocp.N):
            #             current_val = ocp.ocp_solver.get(i, "x")
            #             x_sol[i] = np.copy(current_val[:4])
            #             current_u = ocp.ocp_solver.get(i-1, "u")
            #             u_sol[i-1] = current_u

            #             if not ((abs(current_u[0]) < tau_max + 1e-4 and abs(current_u[0]) > tau_max - 1e-4) or (abs(current_u[1]) < tau_max + 1e-4 and abs(current_u[1]) > tau_max - 1e-4)):
            #                 is_u_at_limit = False
            #             else:
            #                 is_u_at_limit = True
                        
            #             if abs(current_val[0] - q_max) < 1e-4 or abs(current_val[0] - q_min) < 1e-4 or abs(current_val[1] - q_max) < 1e-4 or abs(current_val[1] - q_min) < 1e-4 or abs(current_val[2] - v_max) < 1e-4 or abs(current_val[2] - v_min) < 1e-4 or abs(current_val[3] - v_max) < 1e-4 or abs(current_val[3] - v_min) < 1e-4:
            #                 is_x_at_limit = True
            #             else:
            #                 is_x_at_limit = False

            #             # with torch.no_grad():
            #             #     print('x: ', x_sol[i], model((torch.from_numpy(np.float32([x_sol[i]])).to(device) - mean) / std).numpy(), is_x_at_limit, is_u_at_limit)

            #             # until is_x_at_limit, it is on dX and the unviable data can be generated and used
            #             # when is_x_at_limit switvhes to False there are two possibilities:
            #             #   if is_u_at_limit then the sol is on dV and it can be used
            #             #   alternatively the sol is not valid because it's not on the boundary

            #             if check_init:
            #                 if is_x_at_limit:
            #                     x_sym = np.copy(current_val[:4])

            #                     if abs(current_val[0] - q_max) < 1e-4:
            #                         x_sym[0] = q_max + 1e-4
            #                     if abs(current_val[0] - q_min) < 1e-4:
            #                         x_sym[0] = q_min - 1e-4
            #                     if abs(current_val[1] - q_max) < 1e-4:
            #                         x_sym[1] = q_max + 1e-4
            #                     if abs(current_val[1] - q_min) < 1e-4:
            #                         x_sym[1] = q_min - 1e-4
            #                     if abs(current_val[2] - v_max) < 1e-4:
            #                         x_sym[2] = v_max + 1e-4
            #                     if abs(current_val[2] - v_min) < 1e-4:
            #                         x_sym[2] = v_min - 1e-4
            #                     if abs(current_val[3] - v_max) < 1e-4:
            #                         x_sym[3] = v_max + 1e-4
            #                     if abs(current_val[3] - v_min) < 1e-4:
            #                         x_sym[3] = v_min - 1e-4
                            
            #                     X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
            #                     X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)

            #                     with torch.no_grad():
            #                         out_v = model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std).numpy()
            #                         out_uv = model((torch.from_numpy(np.float32([x_sym])).to(device) - mean) / std).numpy()
            #                         if out_uv[0,0] < out_v[0,0] or np.argmax(out_uv, axis=1) == 1:
            #                             print('1 set, at the limit, x: ', current_val[:4], out_v)
            #                             print('1 set, at the limit, x_sym: ', x_sym, out_uv)
            #                 else:
            #                     if is_u_at_limit:
            #                         # idea: if I am on dV but not touching a state limit, at least one control is saturated and this is needed to remain in V. 
            #                         # If I apply a different control => i get an unviable state
            #                         print('The solution is valid, starting to simulate')
                                    
            #                         u_sym = ocp.ocp_solver.get(i, "u")
            #                         x_sym = np.copy(current_val[:4])

            #                         if u_sym[0] > tau_max - 1e-4:
            #                             u_sym[0] = tau_max - 1e-2
            #                         if u_sym[0] < -tau_max + 1e-4:
            #                             u_sym[0] = -tau_max + 1e-2
            #                         if u_sym[1] > tau_max - 1e-4:
            #                             u_sym[1] = tau_max - 1e-2
            #                         if u_sym[1] < -tau_max + 1e-4:
            #                             u_sym[1] = -tau_max + 1e-2
                                    
            #                         sim.acados_integrator.set("u", u_sym)
            #                         sim.acados_integrator.set("x", x_sym)
            #                         sim.acados_integrator.set("T", dt_sym)
            #                         status = sim.acados_integrator.solve()
            #                         x_sym = sim.acados_integrator.get("x")

            #                         X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
            #                         X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)

            #                         with torch.no_grad():
            #                             out_v = model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std).numpy()
            #                             out_uv = model((torch.from_numpy(np.float32([x_sym])).to(device) - mean) / std).numpy()
            #                             if out_uv[0,0] < out_v[0,0] or np.argmax(out_uv, axis=1) == 1:
            #                                 print('1 set, at the limit simulation, x: ', current_val[:4], out_v)
            #                                 print('1 set, at the limit simulation, x_sym: ', x_sym, out_uv)
                                    
            #                         check_init = False
            #                         status_in = True

            #                     else:
            #                         # the solution is not min time
            #                         print('The solution is not min time')

            #                         # dt_sym_first = ocp.ocp_solver.get(0, "x")[4]
            #                         # print('FIRST MIN TIME SOLVED', dt_sym_first)

            #                         # p = np.array([0., 0., 0., 0., 0.])

            #                         # x_guess = np.empty((ocp.N+1,5))
            #                         # u_guess = np.empty((ocp.N,2))

            #                         # prev_sol = ocp.ocp_solver.get(0, "x")
            #                         # x_guess[0] = [prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3], 1.]
            #                         # u_guess[0] = ocp.ocp_solver.get(0, "u")
            #                         # ocp.ocp_solver.set(0, 'p', p)
            #                         # ocp.ocp_solver.constraints_set(0, 'lbx', x_guess[0]) 
            #                         # ocp.ocp_solver.constraints_set(0, 'ubx', x_guess[0])

            #                         # for i in range(1, ocp.N):
            #                         #     prev_sol = ocp.ocp_solver.get(i, "x")
            #                         #     x_guess[i] = [prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3], 1.]
            #                         #     u_guess[i] = ocp.ocp_solver.get(i, "u")
            #                         #     ocp.ocp_solver.set(i, 'p', p)
            #                         #     ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1.])) 
            #                         #     ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1.])) 

            #                         # prev_sol = ocp.ocp_solver.get(ocp.N, "x")
            #                         # x_guess[ocp.N] = [prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3], 1.]
            #                         # ocp.ocp_solver.set(ocp.N, 'p', p)
            #                         # ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_max, q_fin, 0., 0., 1.]))
            #                         # ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_max, q_fin, 0., 0., 1.]))

            #                         # time_up = dt_sym_first
            #                         # time_down = 0.

            #                         # it = 0

            #                         # while it <= 10:
            #                         # # for i, time in enumerate(np.linspace(0, dt_sym_first, 10, endpoint=True)):
            #                         #     time_dt = time_down + (time_up - time_down)/2
            #                         #     it = it + 1

            #                         #     ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), time_dt))

            #                         #     ocp.ocp_solver.reset()

            #                         #     for i in range(ocp.N):
            #                         #         ocp.ocp_solver.set(i, 'x', x_guess[i])
            #                         #         ocp.ocp_solver.set(i, 'u', u_guess[i])
            #                         #     ocp.ocp_solver.set(ocp.N, 'x', x_guess[ocp.N])

            #                         #     status = ocp.ocp_solver.solve()

            #                         #     if status == 0:
            #                         #         dt_sym_second = ocp.ocp_solver.get(0, "x")[4]
            #                         #         print('SECOND MIN TIME SOLVED', time_dt)

            #                         #         time_up = time_dt
            #                         #     else:
            #                         #         print('SECOND MIN TIME FAILED AT', time_dt)

            #                         #         time_down = time_dt

            #                         # print(time_up)

            #                         break
            #             else:
            #                 if x_sym[0] >= q_min and x_sym[0] <= q_max and x_sym[1] >= q_min and x_sym[1] <= q_max and x_sym[2] <= v_max and x_sym[2] >= v_min and x_sym[3] <= v_max and x_sym[3] >= v_min and status_in:
            #                     u_sym = ocp.ocp_solver.get(i, "u")   
            #                     sim.acados_integrator.set("u", u_sym)
            #                     sim.acados_integrator.set("x", x_sym)
            #                     sim.acados_integrator.set("T", dt_sym)
            #                     status = sim.acados_integrator.solve()
            #                     x_sym = sim.acados_integrator.get("x")
            #                 else:
            #                     if status_in:
            #                         status_in = False
            #                         x_sym_out = x_sym

            #                     x_sym = np.copy(current_val[:4])

            #                     if x_sym_out[0] < q_min or x_sym_out[0] > q_max:
            #                         x_sym[0] = x_sym_out[0]
            #                     if x_sym_out[2] < v_min or x_sym_out[2] > v_max:
            #                         x_sym[2] = x_sym_out[2]
            #                     if x_sym_out[1] < q_min or x_sym_out[1] > q_max:
            #                         x_sym[1] = x_sym_out[1]
            #                     if x_sym_out[3] < v_min or x_sym_out[3] > v_max:
            #                         x_sym[3] = x_sym_out[3]
                            
            #                 X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
            #                 X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)

            #                 with torch.no_grad():
            #                     out_v = model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std).numpy()
            #                     out_uv = model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std).numpy()
            #                     if out_uv[0,0] < out_v[0,0] or np.argmax(out_uv, axis=1) == 1:
            #                         print('1 set, at the limit simulation, x: ', current_val[:4], out_v)
            #                         print('1 set, at the limit simulation, x_sym: ', x_sym, out_uv, 'status: ', status_in)

            #         current_val = ocp.ocp_solver.get(ocp.N, "x")
            #         x_sol[ocp.N] = np.copy(current_val[:4])

            #         if abs(current_val[0] - q_max) < 1e-4:
            #             x_sym[0] = q_max + 1e-4
            #         if abs(current_val[0] - q_min) < 1e-4:
            #             x_sym[0] = q_min - 1e-4
            #         if abs(current_val[1] - q_max) < 1e-4:
            #             x_sym[1] = q_max + 1e-4
            #         if abs(current_val[1] - q_min) < 1e-4:
            #             x_sym[1] = q_min - 1e-4
            #         if abs(current_val[2] - v_max) < 1e-4:
            #             x_sym[2] = v_max + 1e-4
            #         if abs(current_val[2] - v_min) < 1e-4:
            #             x_sym[2] = v_min - 1e-4
            #         if abs(current_val[3] - v_max) < 1e-4:
            #             x_sym[3] = v_max + 1e-4
            #         if abs(current_val[3] - v_min) < 1e-4:
            #             x_sym[3] = v_min - 1e-4
                
            #         X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
            #         X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)





            #         # dt_sym = ocp.ocp_solver.get(0, "x")[4]

            #         # tm = np.linspace(0, ocp.N*dt_sym, ocp.N+1)
            #         # to = np.linspace(0, 1., ocp.N+1)

            #         # plt.figure()
            #         # plt.subplot(2, 1, 1)
            #         # line, = plt.step(to, np.append([u_guess[0, 0]], u_guess[:, 0]))
            #         # line, = plt.step(tm, np.append([u_sol[0, 0]], u_sol[:, 0]))
            #         # plt.ylabel('$C1$')
            #         # plt.xlabel('$t$')
            #         # plt.hlines(ocp.Cmax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         # plt.hlines(-ocp.Cmax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         # plt.ylim([-1.2*ocp.Cmax, 1.2*ocp.Cmax])
            #         # plt.title('Controls, X0 AT THE LIMIT')
            #         # plt.grid()
            #         # plt.subplot(2, 1, 2)
            #         # line, = plt.step(to, np.append([u_guess[0, 1]], u_guess[:, 1]))
            #         # line, = plt.step(tm, np.append([u_sol[0, 1]], u_sol[:, 1]))
            #         # plt.ylabel('$C2$')
            #         # plt.xlabel('$t$')
            #         # plt.hlines(ocp.Cmax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         # plt.hlines(-ocp.Cmax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         # plt.ylim([-1.2*ocp.Cmax, 1.2*ocp.Cmax])
            #         # plt.grid()

            #         # plt.figure()
            #         # plt.subplot(4, 1, 1)
            #         # line, = plt.plot(to, x_guess[:, 0])
            #         # line, = plt.plot(tm, x_sol[:, 0])
            #         # plt.ylabel('$theta1$')
            #         # plt.xlabel('$t$')
            #         # plt.hlines(ocp.thetamax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         # plt.hlines(ocp.thetamin, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         # plt.title('States, X0 AT THE LIMIT')
            #         # plt.grid()
            #         # plt.subplot(4, 1, 2)
            #         # line, = plt.plot(to, x_guess[:, 1])
            #         # line, = plt.plot(tm, x_sol[:, 1])
            #         # plt.ylabel('$theta2$')
            #         # plt.xlabel('$t$')
            #         # plt.hlines(ocp.thetamax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         # plt.hlines(ocp.thetamin, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         # plt.grid()
            #         # plt.subplot(4, 1, 3)
            #         # line, = plt.plot(to, x_guess[:, 2])
            #         # line, = plt.plot(tm, x_sol[:, 2])
            #         # plt.ylabel('$dtheta1$')
            #         # plt.xlabel('$t$')
            #         # plt.hlines(ocp.dthetamax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         # plt.hlines(-ocp.dthetamax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         # plt.grid()
            #         # plt.subplot(4, 1, 4)
            #         # line, = plt.plot(to, x_guess[:, 3])
            #         # line, = plt.plot(tm, x_sol[:, 3])
            #         # plt.ylabel('$dtheta2$')
            #         # plt.xlabel('$t$')
            #         # plt.hlines(ocp.dthetamax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         # plt.hlines(-ocp.dthetamax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         # plt.grid()

            #         # plt.show()

                    
   
                    
            # else:
            #     print('NOT AT THE LIMIT')

            #     # with torch.no_grad():
            #     #     out_v = model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std).numpy()
            #     #     out_uv = model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std).numpy()
            #     #     if out_uv[0,0] < out_v[0,0] or np.argmax(out_uv, axis=1) == 1:
            #     #         print('1 set, not at the limit, x0: ', x0, out_v)
            #     #         print('1 set, not at the limit, x_sym0: ', x_sym, out_uv)

            #     if status == 0:                 
            #         x_sol = np.empty((ocp.N+1,4))
            #         u_sol = np.empty((ocp.N,2))

            #         x_sol[0] = [x0[0], x0[1], x0[2], x0[3]]
            #         u_sol[0] = u0

            #         dt_sym = ocp.ocp_solver.get(0, "x")[4]

            #         tm = np.linspace(0, ocp.N*dt_sym, ocp.N+1)
            #         to = np.linspace(0, 1., ocp.N+1)

            #         status_in = True

            #         for f in range(ocp.N):
            #             current_val = ocp.ocp_solver.get(f+1, "x")
            #             x_sol[f+1] = [current_val[0], current_val[1], current_val[2], current_val[3]]
            #             if f != ocp.N-1:
            #                 u_sol[f+1] = ocp.ocp_solver.get(f+1, "u")
                        
            #             if x_sym[0] >= q_min and x_sym[0] <= q_max and x_sym[1] >= q_min and x_sym[1] <= q_max and x_sym[2] <= v_max and x_sym[2] >= v_min and x_sym[3] <= v_max and x_sym[3] >= v_min and status_in:
            #                 u_sym = ocp.ocp_solver.get(f, "u")     
            #                 sim.acados_integrator.set("u", u_sym)
            #                 sim.acados_integrator.set("x", x_sym)
            #                 sim.acados_integrator.set("T", dt_sym)
            #                 status = sim.acados_integrator.solve()
            #                 x_sym = sim.acados_integrator.get("x")
            #             else:
            #                 if status_in:
            #                     status_in = False
            #                     x_sym_out = x_sym

            #                 x_sym = np.copy(current_val[:4])

            #                 if x_sym_out[0] < q_min or x_sym_out[0] > q_max:
            #                     x_sym[0] = x_sym_out[0]
            #                 if x_sym_out[2] < v_min or x_sym_out[2] > v_max:
            #                     x_sym[2] = x_sym_out[2]
            #                 if x_sym_out[1] < q_min or x_sym_out[1] > q_max:
            #                     x_sym[1] = x_sym_out[1]
            #                 if x_sym_out[3] < v_min or x_sym_out[3] > v_max:
            #                     x_sym[3] = x_sym_out[3]
                        
            #             X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
            #             X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)

            #             with torch.no_grad():
            #                 out_v = model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std).numpy()
            #                 out_uv = model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std).numpy()
            #                 if out_uv[0,0] < out_v[0,0] or np.argmax(out_uv, axis=1) == 1:
            #                     print('1 set, not at the limit, x: ', current_val[:4], out_v)
            #                     print('1 set, not at the limit, x_sym: ', x_sym, out_uv, 'status: ', status_in)

            #         plt.figure()
            #         plt.subplot(2, 1, 1)
            #         line, = plt.step(to, np.append([u_guess[0, 0]], u_guess[:, 0]))
            #         line, = plt.step(tm, np.append([u_sol[0, 0]], u_sol[:, 0]))
            #         plt.ylabel('$C1$')
            #         plt.xlabel('$t$')
            #         plt.hlines(ocp.Cmax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         plt.hlines(-ocp.Cmax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         plt.ylim([-1.2*ocp.Cmax, 1.2*ocp.Cmax])
            #         plt.title('Controls, X0 NOT AT THE LIMIT')
            #         plt.grid()
            #         plt.subplot(2, 1, 2)
            #         line, = plt.step(to, np.append([u_guess[0, 1]], u_guess[:, 1]))
            #         line, = plt.step(tm, np.append([u_sol[0, 1]], u_sol[:, 1]))
            #         plt.ylabel('$C2$')
            #         plt.xlabel('$t$')
            #         plt.hlines(ocp.Cmax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         plt.hlines(-ocp.Cmax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         plt.ylim([-1.2*ocp.Cmax, 1.2*ocp.Cmax])
            #         plt.grid()

            #         plt.figure()
            #         plt.subplot(4, 1, 1)
            #         line, = plt.plot(to, x_guess[:, 0])
            #         line, = plt.plot(tm, x_sol[:, 0])
            #         plt.ylabel('$theta1$')
            #         plt.xlabel('$t$')
            #         plt.hlines(ocp.thetamax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         plt.hlines(ocp.thetamin, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         plt.title('States, X0 NOT AT THE LIMIT')
            #         plt.grid()
            #         plt.subplot(4, 1, 2)
            #         line, = plt.plot(to, x_guess[:, 1])
            #         line, = plt.plot(tm, x_sol[:, 1])
            #         plt.ylabel('$theta2$')
            #         plt.xlabel('$t$')
            #         plt.hlines(ocp.thetamax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         plt.hlines(ocp.thetamin, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         plt.grid()
            #         plt.subplot(4, 1, 3)
            #         line, = plt.plot(to, x_guess[:, 2])
            #         line, = plt.plot(tm, x_sol[:, 2])
            #         plt.ylabel('$dtheta1$')
            #         plt.xlabel('$t$')
            #         plt.hlines(ocp.dthetamax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         plt.hlines(-ocp.dthetamax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         plt.grid()
            #         plt.subplot(4, 1, 4)
            #         line, = plt.plot(to, x_guess[:, 3])
            #         line, = plt.plot(tm, x_sol[:, 3])
            #         plt.ylabel('$dtheta2$')
            #         plt.xlabel('$t$')
            #         plt.hlines(ocp.dthetamax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         plt.hlines(-ocp.dthetamax, to[0], to[-1], linestyles='dashed', alpha=0.7)
            #         plt.grid()

        # # ----------------------------------------------------------

        # q_fin = q_min + random.random() * (q_max-q_min)

        # ran1 = random.choice([-1, 1]) * random.random() * multip
        # ran2 = random.choice([-1, 1]) * random.random() * multip
        # ran = random.choice([-1, 1]) * random.random()
        # p = np.array([ran, ran1, random.random(), ran2, 0.])

        # ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 0.01))

        # ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_min, q_fin, 0., 0., 1.]))
        # ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_min, q_fin, 0., 0., 1.]))

        # ocp.ocp_solver.reset()

        # ocp.ocp_solver.set(ocp.N, 'x', np.array([q_min, q_fin, 0., 0., 1.]))
        # ocp.ocp_solver.set(ocp.N, 'p', p)

        # for i, tau in enumerate(np.linspace(0, 1, ocp.N, endpoint=False)):
        #     x_guess = np.array([(1-tau)*q_max + tau*q_min, q_fin, (1-tau)*v_min, 0., 1.])
        #     ocp.ocp_solver.set(i, 'x', x_guess)
        #     ocp.ocp_solver.set(i, 'p', p)
        #     ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1.])) 
        #     ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1.])) 

        # status = ocp.ocp_solver.solve()

        # if status == 0:
        #     x0 = ocp.ocp_solver.get(0, "x")
        #     x1_eps = x0[0] - eps * p[0] / (q_max - q_min)
        #     x2_eps = x0[1] - eps * p[1] / (q_max - q_min)
        #     x3_eps = x0[2] - eps * p[2] / (2*v_max)
        #     x4_eps = x0[3] - eps * p[3] / (2*v_max)
        #     x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

        #     if ran < 0:
        #         q_comp1 = q_max
        #     else:
        #         q_comp1 = q_min

        #     if ran1 < 0:
        #         q_comp2 = q_max
        #     else:
        #         q_comp2 = q_min

        #     if ran2 < 0:
        #         v_comp2 = v_max
        #     else:
        #         v_comp2 = v_min

        #     if abs(x0[0] - q_comp1) < 1e-4 or abs(x0[1] - q_comp2) < 1e-4 or abs(x0[3] - v_comp2) < 1e-4 or abs(x0[2] - v_min) < 1e-4: # if you are touching the limits in the cost direction
        #         X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
        #         X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)

        #         # print('second set, at the limit, x0: ', x0, model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std))
        #     else:
        #         x_sol = np.array([[0.,0.,0.,0.]])
        #         u_sol = np.array([[0.,0.]])

        #         for i in range(ocp.N):
        #             prev_sol = ocp.ocp_solver.get(i, "x")
        #             x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3]]], axis = 0)
        #             # X_save = np.append(X_save, [[prev_sol[0], prev_sol[1], 2]], axis = 0) # da togliere
        #             prev_sol = ocp.ocp_solver.get(i, "u")
        #             u_sol = np.append(u_sol, [[prev_sol[0], prev_sol[1]]], axis = 0)

        #         prev_sol = ocp.ocp_solver.get(ocp.N, "x")
        #         x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3]]], axis = 0)

        #         p = np.array([0., 0., 0., 0., 1.])

        #         ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

        #         ocp.ocp_solver.constraints_set(ocp.N, 'lbx', np.array([q_min, q_fin, 0., 0., 0.])) 
        #         ocp.ocp_solver.constraints_set(ocp.N, 'ubx', np.array([q_min, q_fin, 0., 0., 1e-2])) 

        #         ocp.ocp_solver.reset()

        #         ocp.ocp_solver.set(ocp.N, 'x', np.array([q_min, q_fin, 0., 0.,  1e-2]))
        #         ocp.ocp_solver.set(ocp.N, 'p', p)

        #         for i, tau in enumerate(np.linspace(0, 1, ocp.N)):
        #             prev_sol = x_sol[i+1]
        #             x_guess = np.array([prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3], 1e-2])
        #             # x_guess = (1-tau)*np.array([x0[0], x0[1], 2e-3]) + tau*np.array([q_max, 0., 2e-3])
        #             u_guess = u_sol[i+1]
        #             ocp.ocp_solver.set(i, 'x', x_guess)
        #             ocp.ocp_solver.set(i, 'u', u_guess)
        #             ocp.ocp_solver.set(i, 'p', p)
        #             ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min,  0.])) 
        #             ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max,  1e-2])) 

        #         ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x0[0], x0[1], x0[2], x0[3], 0.])) 
        #         ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x0[0], x0[1], x0[2], x0[3], 1e-2])) 

        #         status = ocp.ocp_solver.solve()

        #         # print('second set, minimum time, status: ', status)

        #         if status == 0:
        #             if x_sym[0] > q_max:
        #                 x_sym[0] = q_max
        #             if x_sym[0] < q_min:
        #                 x_sym[0] = q_min
        #             if x_sym[1] > q_max:
        #                 x_sym[1] = q_max
        #             if x_sym[1] < q_min:
        #                 x_sym[1] = q_min
        #             if x_sym[2] < v_min:
        #                 x_sym[2] = v_min
        #             if x_sym[3] > v_max:
        #                 x_sym[3] = v_max
        #             if x_sym[3] < v_min:
        #                 x_sym[3] = v_min

        #             X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)

        #             with torch.no_grad():
        #                 out_v = model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std).numpy()
        #                 out_uv = model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std).numpy()
        #                 if out_uv[0,0] < out_v[0,0] or np.argmax(out_uv, axis=1) == 1:
        #                     print('2 set, not at the limit, x0: ', x0, out_v)
        #                     print('2 set, not at the limit, x_sym: ', x_sym, out_uv)

        #             dt_sym = ocp.ocp_solver.get(0, "x")[4]

        #             for f in range(ocp.N):
        #                 if x_sym[0] >= q_min and x_sym[0] <= q_max and x_sym[1] >= q_min and x_sym[1] <= q_max and x_sym[2] <= v_max and x_sym[2] >= v_min and x_sym[3] <= v_max and x_sym[3] >= v_min:
        #                     u_sym = ocp.ocp_solver.get(f, "u")     
        #                     sim.acados_integrator.set("u", u_sym)
        #                     sim.acados_integrator.set("x", x_sym)
        #                     sim.acados_integrator.set("T", dt_sym)
        #                     status = sim.acados_integrator.solve()
        #                     x_sym = sim.acados_integrator.get("x")
        #                     X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)

        #                     current_val = ocp.ocp_solver.get(f+1, "x")
        #                     X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)
                        
        #                     # print('first set, sym, status: ', status)
        #                     with torch.no_grad():
        #                         out_v = model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std).numpy()
        #                         out_uv = model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std).numpy()
        #                         if out_uv[0,0] < out_v[0,0] or np.argmax(out_uv, axis=1) == 1:
        #                             print('2 set, not at the limit, x: ', current_val[:4], out_v)
        #                             print('2 set, not at the limit, x_sym: ', x_sym, out_uv) 
        #                 else:
        #                     break

        #             # current_val = ocp.ocp_solver.get(ocp.N, "x")
        #             # X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)
                
        #             # print('second set, mintime, x: ', current_val[:4], model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std))
        #         else:
        #             X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
        #             X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)

        # # ---------------------------------------------------------

        # q_fin = q_min + random.random() * (q_max-q_min)

        # ran1 = random.choice([-1, 1]) * random.random() * multip
        # ran2 = random.choice([-1, 1]) * random.random() * multip
        # ran = random.choice([-1, 1]) * random.random()
        # p = np.array([ran1, ran, ran2, -1*random.random(), 0.])

        # ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 0.01))

        # ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_fin, q_max, 0., 0., 1.]))
        # ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_fin, q_max, 0., 0., 1.]))

        # ocp.ocp_solver.reset()

        # ocp.ocp_solver.set(ocp.N, 'x', np.array([q_fin, q_max, 0., 0., 1.]))
        # ocp.ocp_solver.set(ocp.N, 'p', p)

        # for i, tau in enumerate(np.linspace(0, 1, ocp.N, endpoint=False)):
        #     x_guess = np.array([q_fin, (1-tau)*q_min + tau*q_max, 0., (1-tau)*v_max, 1.])
        #     ocp.ocp_solver.set(i, 'x', x_guess)
        #     ocp.ocp_solver.set(i, 'p', p)
        #     ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1.])) 
        #     ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1.])) 

        # status = ocp.ocp_solver.solve()

        # if status == 0:
        #     x0 = ocp.ocp_solver.get(0, "x")
        #     x1_eps = x0[0] - eps * p[0] / (q_max - q_min)
        #     x2_eps = x0[1] - eps * p[1] / (q_max - q_min)
        #     x3_eps = x0[2] - eps * p[2] / (2*v_max)
        #     x4_eps = x0[3] - eps * p[3] / (2*v_max)
        #     x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

        #     if ran < 0:
        #         q_comp2 = q_max
        #     else:
        #         q_comp2 = q_min

        #     if ran1 < 0:
        #         q_comp1 = q_max
        #     else:
        #         q_comp1 = q_min

        #     if ran2 < 0:
        #         v_comp1 = v_max
        #     else:
        #         v_comp1 = v_min

        #     if abs(x0[0] - q_comp1) < 1e-4 or abs(x0[1] - q_comp2) < 1e-4 or abs(x0[2] - v_comp1) < 1e-4 or abs(x0[3] - v_max) < 1e-4: # if you are touching the limits in the cost direction
        #         X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
        #         X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)

        #         # print('third set, at the limit, x0: ', x0, model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std))
        #     else:
        #         x_sol = np.array([[0.,0.,0.,0.]])
        #         u_sol = np.array([[0.,0.]])

        #         for i in range(ocp.N):
        #             prev_sol = ocp.ocp_solver.get(i, "x")
        #             x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3]]], axis = 0)
        #             # X_save = np.append(X_save, [[prev_sol[0], prev_sol[1], 2]], axis = 0) # da togliere
        #             prev_sol = ocp.ocp_solver.get(i, "u")
        #             u_sol = np.append(u_sol, [[prev_sol[0], prev_sol[1]]], axis = 0)

        #         prev_sol = ocp.ocp_solver.get(ocp.N, "x")
        #         x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3]]], axis = 0)

        #         p = np.array([0., 0., 0., 0., 1.])

        #         ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

        #         ocp.ocp_solver.constraints_set(ocp.N, 'lbx', np.array([q_fin, q_max, 0., 0., 0.])) 
        #         ocp.ocp_solver.constraints_set(ocp.N, 'ubx', np.array([q_fin, q_max, 0., 0., 1e-2]))

        #         ocp.ocp_solver.reset() 

        #         ocp.ocp_solver.set(ocp.N, 'x', np.array([q_fin, q_max, 0., 0.,  1e-2]))
        #         ocp.ocp_solver.set(ocp.N, 'p', p)

        #         for i, tau in enumerate(np.linspace(0, 1, ocp.N)):
        #             prev_sol = x_sol[i+1]
        #             x_guess = np.array([prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3], 1e-2])
        #             # x_guess = (1-tau)*np.array([x0[0], x0[1], 2e-3]) + tau*np.array([q_max, 0., 2e-3])
        #             u_guess = u_sol[i+1]
        #             ocp.ocp_solver.set(i, 'x', x_guess)
        #             ocp.ocp_solver.set(i, 'u', u_guess)
        #             ocp.ocp_solver.set(i, 'p', p)
        #             ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min,  0.])) 
        #             ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max,  1e-2])) 

        #         ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x0[0], x0[1], x0[2], x0[3], 0.])) 
        #         ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x0[0], x0[1], x0[2], x0[3], 1e-2])) 

        #         status = ocp.ocp_solver.solve()

        #         # print('third set, minimum time, status: ', status)

        #         if status == 0:
        #             if x_sym[0] > q_max:
        #                 x_sym[0] = q_max
        #             if x_sym[0] < q_min:
        #                 x_sym[0] = q_min
        #             if x_sym[1] > q_max:
        #                 x_sym[1] = q_max
        #             if x_sym[1] < q_min:
        #                 x_sym[1] = q_min
        #             if x_sym[3] > v_max:
        #                 x_sym[3] = v_max
        #             if x_sym[2] > v_max:
        #                 x_sym[2] = v_max
        #             if x_sym[2] < v_min:
        #                 x_sym[2] = v_min

        #             X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)

        #             with torch.no_grad():
        #                 out_v = model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std).numpy()
        #                 out_uv = model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std).numpy()
        #                 if out_uv[0,0] < out_v[0,0] or np.argmax(out_uv, axis=1) == 1:
        #                     print('3 set, not at the limit, x0: ', x0, out_v)
        #                     print('3 set, not at the limit, x_sym: ', x_sym, out_uv)

        #             dt_sym = ocp.ocp_solver.get(0, "x")[4]

        #             for f in range(ocp.N):
        #                 if x_sym[0] >= q_min and x_sym[0] <= q_max and x_sym[1] >= q_min and x_sym[1] <= q_max and x_sym[2] <= v_max and x_sym[2] >= v_min and x_sym[3] <= v_max and x_sym[3] >= v_min:
        #                     u_sym = ocp.ocp_solver.get(f, "u")     
        #                     sim.acados_integrator.set("u", u_sym)
        #                     sim.acados_integrator.set("x", x_sym)
        #                     sim.acados_integrator.set("T", dt_sym)
        #                     status = sim.acados_integrator.solve()
        #                     x_sym = sim.acados_integrator.get("x")
        #                     X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)

        #                     current_val = ocp.ocp_solver.get(f+1, "x")
        #                     X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)
                    
        #                     # print('first set, sym, status: ', status)
        #                     with torch.no_grad():
        #                         out_v = model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std).numpy()
        #                         out_uv = model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std).numpy()
        #                         if out_uv[0,0] < out_v[0,0] or np.argmax(out_uv, axis=1) == 1:
        #                             print('3 set, not at the limit, x: ', current_val[:4], out_v)
        #                             print('3 set, not at the limit, x_sym: ', x_sym, out_uv) 
        #                 else:
        #                     break

        #             # current_val = ocp.ocp_solver.get(ocp.N, "x")
        #             # X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)
                
        #             # print('third set, mintime, x: ', current_val[:4], model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std))
        #         else:
        #             X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
        #             X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)

        # # ----------------------------------------------------------

        # q_fin = q_min + random.random() * (q_max-q_min)

        # ran1 = random.choice([-1, 1]) * random.random() * multip
        # ran2 = random.choice([-1, 1]) * random.random() * multip
        # ran = random.choice([-1, 1]) * random.random()
        # p = np.array([ran1, ran, ran2, random.random(), 0.])

        # ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 0.01))

        # ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_fin, q_min, 0., 0., 1.]))
        # ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_fin, q_min, 0., 0., 1.]))

        # ocp.ocp_solver.reset()

        # ocp.ocp_solver.set(ocp.N, 'x', np.array([q_fin, q_min, 0., 0., 1.]))
        # ocp.ocp_solver.set(ocp.N, 'p', p)

        # for i, tau in enumerate(np.linspace(0, 1, ocp.N, endpoint=False)):
        #     x_guess = np.array([q_fin, (1-tau)*q_max + tau*q_min, 0., (1-tau)*v_min, 1.])
        #     ocp.ocp_solver.set(i, 'x', x_guess)
        #     ocp.ocp_solver.set(i, 'p', p)
        #     ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1.])) 
        #     ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1.])) 

        # status = ocp.ocp_solver.solve()

        # if status == 0:
        #     x0 = ocp.ocp_solver.get(0, "x")
        #     x1_eps = x0[0] - eps * p[0] / (q_max - q_min)
        #     x2_eps = x0[1] - eps * p[1] / (q_max - q_min)
        #     x3_eps = x0[2] - eps * p[2] / (2*v_max)
        #     x4_eps = x0[3] - eps * p[3] / (2*v_max)
        #     x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

        #     if ran < 0:
        #         q_comp2 = q_max
        #     else:
        #         q_comp2 = q_min

        #     if ran1 < 0:
        #         q_comp1 = q_max
        #     else:
        #         q_comp1 = q_min

        #     if ran2 < 0:
        #         v_comp1 = v_max
        #     else:
        #         v_comp1 = v_min

        #     if abs(x0[0] - q_comp1) < 1e-4 or abs(x0[1] - q_comp2) < 1e-4 or abs(x0[2] - v_comp2) < 1e-4 or abs(x0[3] - v_min) < 1e-4: # if you are touching the limits in the cost direction
        #         X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
        #         X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)

        #         # print('fourth set, at the limit, x0: ', x0, model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std))
        #     else:
        #         x_sol = np.array([[0.,0.,0.,0.]])
        #         u_sol = np.array([[0.,0.]])

        #         for i in range(ocp.N):
        #             prev_sol = ocp.ocp_solver.get(i, "x")
        #             x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3]]], axis = 0)
        #             # X_save = np.append(X_save, [[prev_sol[0], prev_sol[1], 2]], axis = 0) # da togliere
        #             prev_sol = ocp.ocp_solver.get(i, "u")
        #             u_sol = np.append(u_sol, [[prev_sol[0], prev_sol[1]]], axis = 0)

        #         prev_sol = ocp.ocp_solver.get(ocp.N, "x")
        #         x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3]]], axis = 0)

        #         p = np.array([0., 0., 0., 0., 1.])

        #         ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

        #         ocp.ocp_solver.constraints_set(ocp.N, 'lbx', np.array([q_fin, q_min, 0., 0., 0.])) 
        #         ocp.ocp_solver.constraints_set(ocp.N, 'ubx', np.array([q_fin, q_min, 0., 0., 1e-2])) 

        #         ocp.ocp_solver.reset()

        #         ocp.ocp_solver.set(ocp.N, 'x', np.array([q_fin, q_min, 0., 0.,  1e-2]))
        #         ocp.ocp_solver.set(ocp.N, 'p', p)

        #         for i, tau in enumerate(np.linspace(0, 1, ocp.N)):
        #             prev_sol = x_sol[i+1]
        #             x_guess = np.array([prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3], 1e-2])
        #             # x_guess = (1-tau)*np.array([x0[0], x0[1], 2e-3]) + tau*np.array([q_max, 0., 2e-3])
        #             u_guess = u_sol[i+1]
        #             ocp.ocp_solver.set(i, 'x', x_guess)
        #             ocp.ocp_solver.set(i, 'u', u_guess)
        #             ocp.ocp_solver.set(i, 'p', p)
        #             ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min,  0.])) 
        #             ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max,  1e-2])) 

        #         ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x0[0], x0[1], x0[2], x0[3], 0.])) 
        #         ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x0[0], x0[1], x0[2], x0[3], 1e-2])) 

        #         status = ocp.ocp_solver.solve()

        #         # print('second set, minimum time, status: ', status)

        #         if status == 0:
        #             if x_sym[0] > q_max:
        #                 x_sym[0] = q_max
        #             if x_sym[0] < q_min:
        #                 x_sym[0] = q_min
        #             if x_sym[1] > q_max:
        #                 x_sym[1] = q_max
        #             if x_sym[1] < q_min:
        #                 x_sym[1] = q_min
        #             if x_sym[3] < v_min:
        #                 x_sym[3] = v_min
        #             if x_sym[2] > v_max:
        #                 x_sym[2] = v_max
        #             if x_sym[2] < v_min:
        #                 x_sym[2] = v_min

        #             X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)

        #             with torch.no_grad():
        #                 out_v = model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std).numpy()
        #                 out_uv = model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std).numpy()
        #                 if out_uv[0,0] < out_v[0,0] or np.argmax(out_uv, axis=1) == 1:
        #                     print('4 set, not at the limit, x0: ', x0, out_v)
        #                     print('4 set, not at the limit, x_sym: ', x_sym, out_uv)

        #             dt_sym = ocp.ocp_solver.get(0, "x")[4]

        #             for f in range(ocp.N):
        #                 if x_sym[0] >= q_min and x_sym[0] <= q_max and x_sym[1] >= q_min and x_sym[1] <= q_max and x_sym[2] <= v_max and x_sym[2] >= v_min and x_sym[3] <= v_max and x_sym[3] >= v_min:
        #                     u_sym = ocp.ocp_solver.get(f, "u")     
        #                     sim.acados_integrator.set("u", u_sym)
        #                     sim.acados_integrator.set("x", x_sym)
        #                     sim.acados_integrator.set("T", dt_sym)
        #                     status = sim.acados_integrator.solve()
        #                     x_sym = sim.acados_integrator.get("x")
        #                     X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)

        #                     current_val = ocp.ocp_solver.get(f+1, "x")
        #                     X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)
                    
        #                     # print('first set, sym, status: ', status)
        #                     with torch.no_grad():
        #                         out_v = model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std).numpy()
        #                         out_uv = model((torch.from_numpy(np.float32([x_sym[:4]])).to(device) - mean) / std).numpy()
        #                         if out_uv[0,0] < out_v[0,0] or np.argmax(out_uv, axis=1) == 1:
        #                             print('4 set, not at the limit, x: ', current_val[:4], out_v)
        #                             print('4 set, not at the limit, x_sym: ', x_sym, out_uv) 
        #                 else:
        #                     break

        #             # current_val = ocp.ocp_solver.get(ocp.N, "x")
        #             # X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)

        #             # print('second set, mintime, x: ', current_val[:4], model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std))
        #         else:
        #             X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
        #             X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)
    
    clf.fit(X_save[:,:4], X_save[:,4])

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
    #         norm(X_save[i][1] - (q_min + q_max) / 2) < 0.1
    #         and norm(X_save[i][3]) < 1.
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
