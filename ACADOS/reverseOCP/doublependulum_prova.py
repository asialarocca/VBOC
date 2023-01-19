import numpy as np
import random 
import matplotlib.pyplot as plt
from sklearn import svm
from numpy.linalg import norm as norm
import time
from double_pendulum_ocp_class import OCPdoublependulumRINIT, SYMdoublependulumINIT
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
        model.load_state_dict(torch.load('model_2pendulum'))

        mean, std = torch.tensor(1.9635), torch.tensor(2.8805)

    # Initialization of the SVM classifier:
    clf = svm.SVC(C=1e6, kernel='rbf', class_weight='balanced')

    eps = 1.
    multip = 1.
    num = 20
    
    X_save = np.array([[(q_min+q_max)/2, (q_min+q_max)/2, 0., 0., 1]])
    
    for _ in range(num):
               
        q_fin = q_min + random.random() * (q_max-q_min)

        ran1 = random.choice([-1, 1]) * random.random() * multip
        ran2 = random.choice([-1, 1]) * random.random() * multip
        ran = random.choice([-1, 1]) * random.random()
        p = np.array([ran, ran1, -1*random.random(), ran2, 0.])

        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 0.01))

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_max, q_fin, 0., 0., 1.]))
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_max, q_fin, 0., 0., 1.]))

        ocp.ocp_solver.reset()

        ocp.ocp_solver.set(ocp.N, 'x', np.array([q_max, q_fin, 0., 0., 1.]))
        ocp.ocp_solver.set(ocp.N, 'p', p)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N, endpoint=False)):
            x_guess = np.array([(1-tau)*q_min + tau*q_max, q_fin, (1-tau)*v_max, 0., 1.])
            ocp.ocp_solver.set(i, 'x', x_guess)
            ocp.ocp_solver.set(i, 'p', p)
            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1.])) 
            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1.])) 

        status = ocp.ocp_solver.solve()

        if status == 0:
            x0 = ocp.ocp_solver.get(0, "x")
            x1_eps = x0[0] - eps * p[0] / (q_max - q_min)
            x2_eps = x0[1] - eps * p[1] / (q_max - q_min)
            x3_eps = x0[2] - eps * p[2] / (2*v_max)
            x4_eps = x0[3] - eps * p[3] / (2*v_max)
            x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

            if ran < 0:
                q_comp1 = q_max
            else:
                q_comp1 = q_min

            if ran1 < 0:
                q_comp2 = q_max
            else:
                q_comp2 = q_min

            if ran2 < 0:
                v_comp2 = v_max
            else:
                v_comp2 = v_min

            if abs(x0[0] - q_comp1) < 1e-4 or abs(x0[1] - q_comp2) < 1e-4 or abs(x0[3] - v_comp2) < 1e-4 or abs(x0[2] - v_max) < 1e-4: # if you are touching the limits in the cost direction
                X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
                X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)

                print('first set, at the limit, x0: ', x0, model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std))
            else:
                x_sol = np.array([[0.,0.,0.,0.]])
                u_sol = np.array([[0.,0.]])

                for i in range(ocp.N):
                    prev_sol = ocp.ocp_solver.get(i, "x")
                    x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3]]], axis = 0)
                    # X_save = np.append(X_save, [[prev_sol[0], prev_sol[1], 2]], axis = 0) # da togliere
                    prev_sol = ocp.ocp_solver.get(i, "u")
                    u_sol = np.append(u_sol, [[prev_sol[0], prev_sol[1]]], axis = 0)

                prev_sol = ocp.ocp_solver.get(ocp.N, "x")
                x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3]]], axis = 0)

                p = np.array([0., 0., 0., 0., 1.])

                ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

                ocp.ocp_solver.constraints_set(ocp.N, 'lbx', np.array([q_max, q_fin, 0., 0., 0.])) 
                ocp.ocp_solver.constraints_set(ocp.N, 'ubx', np.array([q_max, q_fin, 0., 0., 1e-2])) 

                ocp.ocp_solver.reset()

                ocp.ocp_solver.set(ocp.N, 'x', np.array([q_max, q_fin, 0., 0.,  1e-2]))
                ocp.ocp_solver.set(ocp.N, 'p', p)

                for i, tau in enumerate(np.linspace(0, 1, ocp.N)):
                    prev_sol = x_sol[i+1]
                    x_guess = np.array([prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3], 1e-2])
                    # x_guess = (1-tau)*np.array([x0[0], x0[1], 2e-3]) + tau*np.array([q_max, 0., 2e-3])
                    u_guess = u_sol[i+1]
                    ocp.ocp_solver.set(i, 'x', x_guess)
                    ocp.ocp_solver.set(i, 'u', u_guess)
                    ocp.ocp_solver.set(i, 'p', p)
                    ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min,  0.])) 
                    ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max,  1e-2])) 

                ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x0[0], x0[1], x0[2], x0[3], 0.])) 
                ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x0[0], x0[1], x0[2], x0[3], 1e-2])) 

                status = ocp.ocp_solver.solve()

                print('first set, minimum time, status: ', status)

                if status == 0:
                    if x_sym[0] > q_max:
                        x_sym[0] = q_max
                    if x_sym[0] < q_min:
                        x_sym[0] = q_min
                    if x_sym[1] > q_max:
                        x_sym[1] = q_max
                    if x_sym[1] < q_min:
                        x_sym[1] = q_min
                    if x_sym[2] > v_max:
                        x_sym[2] = v_max
                    if x_sym[3] > v_max:
                        x_sym[3] = v_max
                    if x_sym[3] < v_min:
                        x_sym[3] = v_min

                    X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)

                    print('first set, not at the limit, x0: ', x0, model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std))

                    dt_sym = ocp.ocp_solver.get(0, "x")[4]

                    for f in range(ocp.N):
                        if x_sym[0] >= q_min and x_sym[0] <= q_max and x_sym[1] >= q_min and x_sym[1] <= q_max and x_sym[2] <= v_max and x_sym[2] >= v_min and x_sym[3] <= v_max and x_sym[3] >= v_min:
                            u_sym = ocp.ocp_solver.get(f, "u")     
                            sim.acados_integrator.set("u", u_sym)
                            sim.acados_integrator.set("x", x_sym)
                            sim.acados_integrator.set("T", dt_sym)
                            status = sim.acados_integrator.solve()
                            x_sym = sim.acados_integrator.get("x")
                            X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)

                            current_val = ocp.ocp_solver.get(f, "x")
                            X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)

                            print('first set, sym, status: ', status)
                            print('first set, sym, x: ', x_sym, model((torch.from_numpy(np.float32([x_sym])).to(device) - mean) / std))
                            print('first set, mintime, x: ', current_val[:4], model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std))
                        else:
                            break
                    
                    current_val = ocp.ocp_solver.get(ocp.N, "x")
                    X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)

                    print('first set, mintime, x: ', current_val[:4], model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std))
                else:
                    X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
                    X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)

        # ----------------------------------------------------------

        print('check')

        q_fin = q_min + random.random() * (q_max-q_min)

        ran1 = random.choice([-1, 1]) * random.random() * multip
        ran2 = random.choice([-1, 1]) * random.random() * multip
        ran = random.choice([-1, 1]) * random.random()
        p = np.array([ran, ran1, random.random(), ran2, 0.])

        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 0.01))

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_min, q_fin, 0., 0., 1.]))
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_min, q_fin, 0., 0., 1.]))

        ocp.ocp_solver.reset()

        ocp.ocp_solver.set(ocp.N, 'x', np.array([q_min, q_fin, 0., 0., 1.]))
        ocp.ocp_solver.set(ocp.N, 'p', p)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N, endpoint=False)):
            x_guess = np.array([(1-tau)*q_max + tau*q_min, q_fin, (1-tau)*v_min, 0., 1.])
            ocp.ocp_solver.set(i, 'x', x_guess)
            ocp.ocp_solver.set(i, 'p', p)
            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1.])) 
            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1.])) 

        status = ocp.ocp_solver.solve()

        if status == 0:
            x0 = ocp.ocp_solver.get(0, "x")
            x1_eps = x0[0] - eps * p[0] / (q_max - q_min)
            x2_eps = x0[1] - eps * p[1] / (q_max - q_min)
            x3_eps = x0[2] - eps * p[2] / (2*v_max)
            x4_eps = x0[3] - eps * p[3] / (2*v_max)
            x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

            if ran < 0:
                q_comp1 = q_max
            else:
                q_comp1 = q_min

            if ran1 < 0:
                q_comp2 = q_max
            else:
                q_comp2 = q_min

            if ran2 < 0:
                v_comp2 = v_max
            else:
                v_comp2 = v_min

            if abs(x0[0] - q_comp1) < 1e-4 or abs(x0[1] - q_comp2) < 1e-4 or abs(x0[3] - v_comp2) < 1e-4 or abs(x0[2] - v_min) < 1e-4: # if you are touching the limits in the cost direction
                X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
                X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)

                print('second set, at the limit, x0: ', x0, model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std))
            else:
                x_sol = np.array([[0.,0.,0.,0.]])
                u_sol = np.array([[0.,0.]])

                for i in range(ocp.N):
                    prev_sol = ocp.ocp_solver.get(i, "x")
                    x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3]]], axis = 0)
                    # X_save = np.append(X_save, [[prev_sol[0], prev_sol[1], 2]], axis = 0) # da togliere
                    prev_sol = ocp.ocp_solver.get(i, "u")
                    u_sol = np.append(u_sol, [[prev_sol[0], prev_sol[1]]], axis = 0)

                prev_sol = ocp.ocp_solver.get(ocp.N, "x")
                x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3]]], axis = 0)

                p = np.array([0., 0., 0., 0., 1.])

                ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

                ocp.ocp_solver.constraints_set(ocp.N, 'lbx', np.array([q_min, q_fin, 0., 0., 0.])) 
                ocp.ocp_solver.constraints_set(ocp.N, 'ubx', np.array([q_min, q_fin, 0., 0., 1e-2])) 

                ocp.ocp_solver.reset()

                ocp.ocp_solver.set(ocp.N, 'x', np.array([q_min, q_fin, 0., 0.,  1e-2]))
                ocp.ocp_solver.set(ocp.N, 'p', p)

                for i, tau in enumerate(np.linspace(0, 1, ocp.N)):
                    prev_sol = x_sol[i+1]
                    x_guess = np.array([prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3], 1e-2])
                    # x_guess = (1-tau)*np.array([x0[0], x0[1], 2e-3]) + tau*np.array([q_max, 0., 2e-3])
                    u_guess = u_sol[i+1]
                    ocp.ocp_solver.set(i, 'x', x_guess)
                    ocp.ocp_solver.set(i, 'u', u_guess)
                    ocp.ocp_solver.set(i, 'p', p)
                    ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min,  0.])) 
                    ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max,  1e-2])) 

                ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x0[0], x0[1], x0[2], x0[3], 0.])) 
                ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x0[0], x0[1], x0[2], x0[3], 1e-2])) 

                status = ocp.ocp_solver.solve()

                print('second set, minimum time, status: ', status)

                print(status)

                if status == 0:
                    if x_sym[0] > q_max:
                        x_sym[0] = q_max
                    if x_sym[0] < q_min:
                        x_sym[0] = q_min
                    if x_sym[1] > q_max:
                        x_sym[1] = q_max
                    if x_sym[1] < q_min:
                        x_sym[1] = q_min
                    if x_sym[2] < v_min:
                        x_sym[2] = v_min
                    if x_sym[3] > v_max:
                        x_sym[3] = v_max
                    if x_sym[3] < v_min:
                        x_sym[3] = v_min

                    X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)

                    print('second set, not at the limit, x0: ', x0, model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std))

                    dt_sym = ocp.ocp_solver.get(0, "x")[4]

                    for f in range(ocp.N):
                        if x_sym[0] >= q_min and x_sym[0] <= q_max and x_sym[1] >= q_min and x_sym[1] <= q_max and x_sym[2] <= v_max and x_sym[2] >= v_min and x_sym[3] <= v_max and x_sym[3] >= v_min:
                            u_sym = ocp.ocp_solver.get(f, "u")     
                            sim.acados_integrator.set("u", u_sym)
                            sim.acados_integrator.set("x", x_sym)
                            sim.acados_integrator.set("T", dt_sym)
                            status = sim.acados_integrator.solve()
                            x_sym = sim.acados_integrator.get("x")
                            X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)

                            current_val = ocp.ocp_solver.get(f, "x")
                            X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)
                        
                            print('second set, sym, status: ', status)
                            print('second set, sym, x: ', x_sym, model((torch.from_numpy(np.float32([x_sym])).to(device) - mean) / std))
                            print('second set, mintime, x: ', current_val[:4], model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std))
                        else:
                            break

                    current_val = ocp.ocp_solver.get(ocp.N, "x")
                    X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)
                
                    print('second set, mintime, x: ', current_val[:4], model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std))
                else:
                    X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
                    X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)

        # ---------------------------------------------------------

        print('check')

        q_fin = q_min + random.random() * (q_max-q_min)

        ran1 = random.choice([-1, 1]) * random.random() * multip
        ran2 = random.choice([-1, 1]) * random.random() * multip
        ran = random.choice([-1, 1]) * random.random()
        p = np.array([ran1, ran, ran2, -1*random.random(), 0.])

        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 0.01))

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_fin, q_max, 0., 0., 1.]))
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_fin, q_max, 0., 0., 1.]))

        ocp.ocp_solver.reset()

        ocp.ocp_solver.set(ocp.N, 'x', np.array([q_fin, q_max, 0., 0., 1.]))
        ocp.ocp_solver.set(ocp.N, 'p', p)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N, endpoint=False)):
            x_guess = np.array([q_fin, (1-tau)*q_min + tau*q_max, 0., (1-tau)*v_max, 1.])
            ocp.ocp_solver.set(i, 'x', x_guess)
            ocp.ocp_solver.set(i, 'p', p)
            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1.])) 
            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1.])) 

        status = ocp.ocp_solver.solve()

        if status == 0:
            x0 = ocp.ocp_solver.get(0, "x")
            x1_eps = x0[0] - eps * p[0] / (q_max - q_min)
            x2_eps = x0[1] - eps * p[1] / (q_max - q_min)
            x3_eps = x0[2] - eps * p[2] / (2*v_max)
            x4_eps = x0[3] - eps * p[3] / (2*v_max)
            x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

            if ran < 0:
                q_comp2 = q_max
            else:
                q_comp2 = q_min

            if ran1 < 0:
                q_comp1 = q_max
            else:
                q_comp1 = q_min

            if ran2 < 0:
                v_comp1 = v_max
            else:
                v_comp1 = v_min

            if abs(x0[0] - q_comp1) < 1e-4 or abs(x0[1] - q_comp2) < 1e-4 or abs(x0[2] - v_comp1) < 1e-4 or abs(x0[3] - v_max) < 1e-4: # if you are touching the limits in the cost direction
                X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
                X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)

                print('third set, at the limit, x0: ', x0, model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std))
            else:
                x_sol = np.array([[0.,0.,0.,0.]])
                u_sol = np.array([[0.,0.]])

                for i in range(ocp.N):
                    prev_sol = ocp.ocp_solver.get(i, "x")
                    x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3]]], axis = 0)
                    # X_save = np.append(X_save, [[prev_sol[0], prev_sol[1], 2]], axis = 0) # da togliere
                    prev_sol = ocp.ocp_solver.get(i, "u")
                    u_sol = np.append(u_sol, [[prev_sol[0], prev_sol[1]]], axis = 0)

                prev_sol = ocp.ocp_solver.get(ocp.N, "x")
                x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3]]], axis = 0)

                p = np.array([0., 0., 0., 0., 1.])

                ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

                ocp.ocp_solver.constraints_set(ocp.N, 'lbx', np.array([q_fin, q_max, 0., 0., 0.])) 
                ocp.ocp_solver.constraints_set(ocp.N, 'ubx', np.array([q_fin, q_max, 0., 0., 1e-2]))

                ocp.ocp_solver.reset() 

                ocp.ocp_solver.set(ocp.N, 'x', np.array([q_fin, q_max, 0., 0.,  1e-2]))
                ocp.ocp_solver.set(ocp.N, 'p', p)

                for i, tau in enumerate(np.linspace(0, 1, ocp.N)):
                    prev_sol = x_sol[i+1]
                    x_guess = np.array([prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3], 1e-2])
                    # x_guess = (1-tau)*np.array([x0[0], x0[1], 2e-3]) + tau*np.array([q_max, 0., 2e-3])
                    u_guess = u_sol[i+1]
                    ocp.ocp_solver.set(i, 'x', x_guess)
                    ocp.ocp_solver.set(i, 'u', u_guess)
                    ocp.ocp_solver.set(i, 'p', p)
                    ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min,  0.])) 
                    ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max,  1e-2])) 

                ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x0[0], x0[1], x0[2], x0[3], 0.])) 
                ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x0[0], x0[1], x0[2], x0[3], 1e-2])) 

                status = ocp.ocp_solver.solve()

                print('third set, minimum time, status: ', status)

                print(status)

                if status == 0:
                    if x_sym[0] > q_max:
                        x_sym[0] = q_max
                    if x_sym[0] < q_min:
                        x_sym[0] = q_min
                    if x_sym[1] > q_max:
                        x_sym[1] = q_max
                    if x_sym[1] < q_min:
                        x_sym[1] = q_min
                    if x_sym[3] > v_max:
                        x_sym[3] = v_max
                    if x_sym[2] > v_max:
                        x_sym[2] = v_max
                    if x_sym[2] < v_min:
                        x_sym[2] = v_min

                    X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)

                    print('third set, not at the limit, x0: ', x0, model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std))

                    dt_sym = ocp.ocp_solver.get(0, "x")[4]

                    for f in range(ocp.N):
                        if x_sym[0] >= q_min and x_sym[0] <= q_max and x_sym[1] >= q_min and x_sym[1] <= q_max and x_sym[2] <= v_max and x_sym[2] >= v_min and x_sym[3] <= v_max and x_sym[3] >= v_min:
                            u_sym = ocp.ocp_solver.get(f, "u")     
                            sim.acados_integrator.set("u", u_sym)
                            sim.acados_integrator.set("x", x_sym)
                            sim.acados_integrator.set("T", dt_sym)
                            status = sim.acados_integrator.solve()
                            x_sym = sim.acados_integrator.get("x")
                            X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)

                            current_val = ocp.ocp_solver.get(f, "x")
                            X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)
                    
                            print('third set, sym, status: ', status)
                            print('third set, sym, x: ', x_sym, model((torch.from_numpy(np.float32([x_sym])).to(device) - mean) / std))
                            print('third set, mintime, x: ', current_val[:4], model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std))
                        else:
                            break

                    current_val = ocp.ocp_solver.get(ocp.N, "x")
                    X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)
                
                    print('third set, mintime, x: ', current_val[:4], model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std))
                else:
                    X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
                    X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)

        print('check')

        # ----------------------------------------------------------

        q_fin = q_min + random.random() * (q_max-q_min)

        ran1 = random.choice([-1, 1]) * random.random() * multip
        ran2 = random.choice([-1, 1]) * random.random() * multip
        ran = random.choice([-1, 1]) * random.random()
        p = np.array([ran1, ran, ran2, random.random(), 0.])

        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 0.01))

        ocp.ocp_solver.constraints_set(ocp.N, "lbx", np.array([q_fin, q_min, 0., 0., 1.]))
        ocp.ocp_solver.constraints_set(ocp.N, "ubx", np.array([q_fin, q_min, 0., 0., 1.]))

        ocp.ocp_solver.reset()

        ocp.ocp_solver.set(ocp.N, 'x', np.array([q_fin, q_min, 0., 0., 1.]))
        ocp.ocp_solver.set(ocp.N, 'p', p)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N, endpoint=False)):
            x_guess = np.array([q_fin, (1-tau)*q_max + tau*q_min, 0., (1-tau)*v_min, 1.])
            ocp.ocp_solver.set(i, 'x', x_guess)
            ocp.ocp_solver.set(i, 'p', p)
            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min, 1.])) 
            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max, 1.])) 

        status = ocp.ocp_solver.solve()

        if status == 0:
            x0 = ocp.ocp_solver.get(0, "x")
            x1_eps = x0[0] - eps * p[0] / (q_max - q_min)
            x2_eps = x0[1] - eps * p[1] / (q_max - q_min)
            x3_eps = x0[2] - eps * p[2] / (2*v_max)
            x4_eps = x0[3] - eps * p[3] / (2*v_max)
            x_sym = np.array([x1_eps, x2_eps, x3_eps, x4_eps])

            print(x0)

            if ran < 0:
                q_comp2 = q_max
            else:
                q_comp2 = q_min

            if ran1 < 0:
                q_comp1 = q_max
            else:
                q_comp1 = q_min

            if ran2 < 0:
                v_comp1 = v_max
            else:
                v_comp1 = v_min

            if abs(x0[0] - q_comp1) < 1e-4 or abs(x0[1] - q_comp2) < 1e-4 or abs(x0[2] - v_comp2) < 1e-4 or abs(x0[3] - v_min) < 1e-4: # if you are touching the limits in the cost direction
                X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
                X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)

                print('fourth set, at the limit, x0: ', x0, model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std))
            else:
                x_sol = np.array([[0.,0.,0.,0.]])
                u_sol = np.array([[0.,0.]])

                for i in range(ocp.N):
                    prev_sol = ocp.ocp_solver.get(i, "x")
                    x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3]]], axis = 0)
                    # X_save = np.append(X_save, [[prev_sol[0], prev_sol[1], 2]], axis = 0) # da togliere
                    prev_sol = ocp.ocp_solver.get(i, "u")
                    u_sol = np.append(u_sol, [[prev_sol[0], prev_sol[1]]], axis = 0)

                prev_sol = ocp.ocp_solver.get(ocp.N, "x")
                x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3]]], axis = 0)

                p = np.array([0., 0., 0., 0., 1.])

                ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

                ocp.ocp_solver.constraints_set(ocp.N, 'lbx', np.array([q_fin, q_min, 0., 0., 0.])) 
                ocp.ocp_solver.constraints_set(ocp.N, 'ubx', np.array([q_fin, q_min, 0., 0., 1e-2])) 

                ocp.ocp_solver.reset()

                ocp.ocp_solver.set(ocp.N, 'x', np.array([q_fin, q_min, 0., 0.,  1e-2]))
                ocp.ocp_solver.set(ocp.N, 'p', p)

                for i, tau in enumerate(np.linspace(0, 1, ocp.N)):
                    prev_sol = x_sol[i+1]
                    x_guess = np.array([prev_sol[0], prev_sol[1], prev_sol[2], prev_sol[3], 1e-2])
                    # x_guess = (1-tau)*np.array([x0[0], x0[1], 2e-3]) + tau*np.array([q_max, 0., 2e-3])
                    u_guess = u_sol[i+1]
                    ocp.ocp_solver.set(i, 'x', x_guess)
                    ocp.ocp_solver.set(i, 'u', u_guess)
                    ocp.ocp_solver.set(i, 'p', p)
                    ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, v_min, v_min,  0.])) 
                    ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, v_max, v_max,  1e-2])) 

                ocp.ocp_solver.constraints_set(0, 'lbx', np.array([x0[0], x0[1], x0[2], x0[3], 0.])) 
                ocp.ocp_solver.constraints_set(0, 'ubx', np.array([x0[0], x0[1], x0[2], x0[3], 1e-2])) 

                status = ocp.ocp_solver.solve()

                print('second set, minimum time, status: ', status)

                print(status)

                if status == 0:
                    if x_sym[0] > q_max:
                        x_sym[0] = q_max
                    if x_sym[0] < q_min:
                        x_sym[0] = q_min
                    if x_sym[1] > q_max:
                        x_sym[1] = q_max
                    if x_sym[1] < q_min:
                        x_sym[1] = q_min
                    if x_sym[3] < v_min:
                        x_sym[3] = v_min
                    if x_sym[2] > v_max:
                        x_sym[2] = v_max
                    if x_sym[2] < v_min:
                        x_sym[2] = v_min

                    X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)

                    print('fourth set, not at the limit, x0: ', x0, model((torch.from_numpy(np.float32([x0[:4]])).to(device) - mean) / std))

                    dt_sym = ocp.ocp_solver.get(0, "x")[4]

                    for f in range(ocp.N):
                        if x_sym[0] >= q_min and x_sym[0] <= q_max and x_sym[1] >= q_min and x_sym[1] <= q_max and x_sym[2] <= v_max and x_sym[2] >= v_min and x_sym[3] <= v_max and x_sym[3] >= v_min:
                            u_sym = ocp.ocp_solver.get(f, "u")     
                            sim.acados_integrator.set("u", u_sym)
                            sim.acados_integrator.set("x", x_sym)
                            sim.acados_integrator.set("T", dt_sym)
                            status = sim.acados_integrator.solve()
                            x_sym = sim.acados_integrator.get("x")
                            X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)

                            current_val = ocp.ocp_solver.get(f, "x")
                            X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)
                    
                            print('fourth set, sym, status: ', status)
                            print('fourth set, sym, x: ', x_sym, model((torch.from_numpy(np.float32([x_sym])).to(device) - mean) / std))
                            print('fourth set, mintime, x: ', current_val[:4], model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std))
                        else:
                            break

                    current_val = ocp.ocp_solver.get(ocp.N, "x")
                    X_save = np.append(X_save, [[current_val[0], current_val[1], current_val[2], current_val[3], 1]], axis = 0)

                    print('second set, mintime, x: ', current_val[:4], model((torch.from_numpy(np.float32([current_val[:4]])).to(device) - mean) / std))
                else:
                    X_save = np.append(X_save, [[x_sym[0], x_sym[1], x_sym[2], x_sym[3], 0]], axis = 0)
                    X_save = np.append(X_save, [[x0[0], x0[1], x0[2], x0[3], 1]], axis = 0)

    print('check')
    
    clf.fit(X_save[:,:4], X_save[:,4])

    # Plot the results:
    plt.figure()
    h = 0.02
    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()
    Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
                    np.zeros(yrav.shape[0]), yrav])
    Z = Z.reshape(xx.shape)
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
            if X_save[i][4] < 0.5:
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
    Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
                    np.zeros(yrav.shape[0])])
    Z = Z.reshape(xx.shape)
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
            if X_save[i][4] < 0.5:
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

    # Plot the results:
    plt.figure()
    h = 0.02
    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()
    Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
                    np.zeros(yrav.shape[0]), yrav])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][0] - (q_min + q_max) / 2) < 0.2
            and norm(X_save[i][2]) < 2.
        ):
            xit.append(X_save[i][1])
            yit.append(X_save[i][3])
            if X_save[i][4] < 0.5:
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
    Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
                    np.zeros(yrav.shape[0])])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][1] - (q_min + q_max) / 2) < 0.2
            and norm(X_save[i][3]) < 2.
        ):
            xit.append(X_save[i][0])
            yit.append(X_save[i][2])
            if X_save[i][4] < 0.5:
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

    # Plot the results:
    plt.figure()
    h = 0.02
    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()
    Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
                    np.zeros(yrav.shape[0]), yrav])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][0] - (q_min + q_max) / 2) < 0.3
            and norm(X_save[i][2]) < 3.
        ):
            xit.append(X_save[i][1])
            yit.append(X_save[i][3])
            if X_save[i][4] < 0.5:
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
    Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
                    np.zeros(yrav.shape[0])])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][1] - (q_min + q_max) / 2) < 0.3
            and norm(X_save[i][3]) < 3.
        ):
            xit.append(X_save[i][0])
            yit.append(X_save[i][2])
            if X_save[i][4] < 0.5:
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

    # Plot the results:
    plt.figure()
    h = 0.02
    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()
    Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
                    np.zeros(yrav.shape[0]), yrav])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][0] - (q_min + q_max) / 2) < 0.4
            and norm(X_save[i][2]) < 4.
        ):
            xit.append(X_save[i][1])
            yit.append(X_save[i][3])
            if X_save[i][4] < 0.5:
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
    Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
                    np.zeros(yrav.shape[0])])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][1] - (q_min + q_max) / 2) < 0.4
            and norm(X_save[i][3]) < 4.
        ):
            xit.append(X_save[i][0])
            yit.append(X_save[i][2])
            if X_save[i][4] < 0.5:
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

    # Plot the results:
    h = 0.02
    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max, h))
    xrav = xx.ravel()
    yrav = yy.ravel()

    plt.figure()
    Z = clf.predict(np.c_[(q_min + q_max) / 2*np.ones(xrav.shape[0]), xrav,
                    np.zeros(yrav.shape[0]), yrav])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][0] - (q_min + q_max) / 2) < 0.5
            and norm(X_save[i][2]) < 5.
        ):
            xit.append(X_save[i][1])
            yit.append(X_save[i][3])
            if X_save[i][4] < 0.5:
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
    Z = clf.predict(np.c_[xrav, (q_min + q_max) / 2*np.ones(xrav.shape[0]), yrav,
                    np.zeros(yrav.shape[0])])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    cit = []
    for i in range(len(X_save)):
        if (
            norm(X_save[i][1] - (q_min + q_max) / 2) < 0.5
            and norm(X_save[i][3]) < 5.
        ):
            xit.append(X_save[i][0])
            yit.append(X_save[i][2])
            if X_save[i][4] < 0.5:
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

    plt.show()
