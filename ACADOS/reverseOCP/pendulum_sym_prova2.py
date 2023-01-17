import time
import scipy.linalg as lin
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
from acados_template import AcadosModel
from casadi import SX, vertcat, sin
import matplotlib.pyplot as plt
from sklearn import svm
from pendulum_ocp_class import OCPpendulumINIT, SYMpendulumINIT
import random
from sklearn import svm

if __name__ == "__main__":

    # Ocp and sym initialization:
    ocp = OCPpendulumINIT()
    sim = SYMpendulumINIT()

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = -ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    X_save = np.array([[(q_min+q_max)/2, 0., 1]])
    eps = 1e-2

    for iter in range(10):

        ocp.ocp_solver.reset()

        p = np.array([-1*random.random(), -1*random.random(), 0.])

        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 0.01))

        ocp.ocp_solver.set(ocp.N, 'lbx', np.array([q_max, 0.,  1.])) 
        ocp.ocp_solver.set(ocp.N, 'ubx', np.array([q_max, 0.,  1.])) 

        ocp.ocp_solver.set(ocp.N, 'x', np.array([q_max, 0.,  1.]))
        ocp.ocp_solver.set(ocp.N, 'p', p)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N)):
            x_guess = (1-tau)*np.array([q_min, v_max, 1.]) + tau*np.array([q_max, 0., 1.])
            ocp.ocp_solver.set(i, 'x', x_guess)
            ocp.ocp_solver.set(i, 'p', p)
            ocp.ocp_solver.set(i, 'lbx', np.array([q_min, v_min,  1.])) 
            ocp.ocp_solver.set(i, 'ubx', np.array([q_max, v_max,  1.])) 

        status = ocp.ocp_solver.solve()
        ocp.ocp_solver.print_statistics()
        print('std ocp')

        if status == 0:
            x0 = ocp.ocp_solver.get(0, "x")
            x1_eps = x0[0] - eps * p[0] / (q_max - q_min)
            x2_eps = x0[1] - eps * p[1] / v_max
            x_sym = np.array([x1_eps, x2_eps])

            x_sol = np.array([[0.,0.]])
            u_sol = np.array([[0.]])

            for i in range(ocp.N):
                prev_sol = ocp.ocp_solver.get(i, "x")
                x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1]]], axis = 0)
                # X_save = np.append(X_save, [[prev_sol[0], prev_sol[1], 2]], axis = 0) # da togliere
                u_sol = np.append(u_sol, [[ocp.ocp_solver.get(i, "x")[0]]], axis = 0)

            prev_sol = ocp.ocp_solver.get(ocp.N, "x")
            x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1]]], axis = 0)

            print(x0)

            if abs(x0[0] - q_max) < 1e-4 or abs(x0[1] - v_max) < 1e-4: # if you are touching the limits in the cost direction
                X_save = np.append(X_save, [[x_sym[0], x_sym[1], 0]], axis = 0)
                X_save = np.append(X_save, [[x0[0], x0[1], 1]], axis = 0)
            else:
                p = np.array([0., 0., 1.])

                ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

                ocp.ocp_solver.reset()

                ocp.ocp_solver.set(ocp.N, 'lbx', np.array([q_max, 0.,  0.])) 
                ocp.ocp_solver.set(ocp.N, 'ubx', np.array([q_max, 0.,  1e-2])) 

                ocp.ocp_solver.set(ocp.N, 'x', np.array([q_max, 0.,  1e-2]))
                ocp.ocp_solver.set(ocp.N, 'p', p)

                for i, tau in enumerate(np.linspace(0, 1, ocp.N)):
                    prev_sol = x_sol[i+1]
                    x_guess = np.array([prev_sol[0], prev_sol[1], 1e-2])
                    # x_guess = (1-tau)*np.array([x0[0], x0[1], 2e-3]) + tau*np.array([q_max, 0., 2e-3])
                    u_guess = u_sol[i+1]
                    ocp.ocp_solver.set(i, 'x', x_guess)
                    ocp.ocp_solver.set(i, 'u', u_guess)
                    ocp.ocp_solver.set(i, 'p', p)
                    ocp.ocp_solver.set(i, 'lbx', np.array([q_min, v_min,  0.])) 
                    ocp.ocp_solver.set(i, 'ubx', np.array([q_max, v_max,  1e-2])) 

                ocp.ocp_solver.set(0, 'lbx', np.array([x0[0], x0[1],  0.])) 
                ocp.ocp_solver.set(0, 'ubx', np.array([x0[0], x0[1],  1e-2])) 

                status = ocp.ocp_solver.solve()
                ocp.ocp_solver.print_statistics()
                print('min time')

                print(status)

                if status == 0:
                    if x_sym[0] > q_max:
                        x_sym[0] = q_max
                    if x_sym[1] > v_max:
                        x_sym[1] = v_max

                    X_save = np.append(X_save, [[x_sym[0], x_sym[1], 0]], axis = 0)

                    dt_sym = ocp.ocp_solver.get(0, "x")[2]

                    for f in range(ocp.N):
                        if x_sym[0] >= q_min and x_sym[0] <= q_max and x_sym[1] <= v_max and x_sym[1] >= v_min:
                            u_sym = ocp.ocp_solver.get(f, "u")     
                            sim.acados_integrator.set("u", u_sym)
                            sim.acados_integrator.set("x", x_sym)
                            sim.acados_integrator.set("T", dt_sym)
                            sim.acados_integrator.solve()
                            x_sym = sim.acados_integrator.get("x")
                            X_save = np.append(X_save, [[x_sym[0], x_sym[1], 0]], axis = 0)

                        current_val = ocp.ocp_solver.get(f, "x")
                        X_save = np.append(X_save, [[current_val[0], current_val[1], 1]], axis = 0)
                    
                    current_val = ocp.ocp_solver.get(ocp.N, "x")
                    X_save = np.append(X_save, [[current_val[0], current_val[1], 1]], axis = 0)
                else:
                    X_save = np.append(X_save, [[x_sym[0], x_sym[1], 0]], axis = 0)
                    X_save = np.append(X_save, [[x0[0], x0[1], 1]], axis = 0)

# --------------------------------------------------------------------------------------------

        ocp.ocp_solver.reset()

        p = np.array([random.random(), random.random(), 0.])

        ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 0.01))

        ocp.ocp_solver.set(ocp.N, 'lbx', np.array([q_min, 0.,  1.])) 
        ocp.ocp_solver.set(ocp.N, 'ubx', np.array([q_min, 0.,  1.])) 

        ocp.ocp_solver.set(ocp.N, 'x', np.array([q_min, 0.,  1.]))
        ocp.ocp_solver.set(ocp.N, 'p', p)

        for i, tau in enumerate(np.linspace(0, 1, ocp.N)):
            x_guess = (1-tau)*np.array([q_max, v_min, 1.]) + tau*np.array([q_min, 0., 1.])
            ocp.ocp_solver.set(i, 'x', x_guess)
            ocp.ocp_solver.set(i, 'p', p)
            ocp.ocp_solver.set(i, 'lbx', np.array([q_min, v_min,  1.])) 
            ocp.ocp_solver.set(i, 'ubx', np.array([q_max, v_max,  1.])) 

        status = ocp.ocp_solver.solve()
        ocp.ocp_solver.print_statistics()
        print('std ocp')

        if status == 0:
            x0 = ocp.ocp_solver.get(0, "x")
            x1_eps = x0[0] - eps * p[0] / (q_max - q_min)
            x2_eps = x0[1] - eps * p[1] / v_max
            x_sym = np.array([x1_eps, x2_eps])

            x_sol = np.array([[0.,0.]])
            u_sol = np.array([[0.]])

            for i in range(ocp.N):
                prev_sol = ocp.ocp_solver.get(i, "x")
                x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1]]], axis = 0)
                # X_save = np.append(X_save, [[prev_sol[0], prev_sol[1], 2]], axis = 0) # da togliere
                u_sol = np.append(u_sol, [[ocp.ocp_solver.get(i, "x")[0]]], axis = 0)

            prev_sol = ocp.ocp_solver.get(ocp.N, "x")
            x_sol = np.append(x_sol, [[prev_sol[0], prev_sol[1]]], axis = 0)

            print(x0)

            if abs(x0[0] - q_min) < 1e-4 or abs(x0[1] - v_min) < 1e-4: # if you are touching the limits in the cost direction
                X_save = np.append(X_save, [[x_sym[0], x_sym[1], 0]], axis = 0)
                X_save = np.append(X_save, [[x0[0], x0[1], 1]], axis = 0)
            else:
                p = np.array([0., 0., 1.])

                ocp.ocp_solver.set_new_time_steps(np.full((ocp.N,), 1.))

                ocp.ocp_solver.reset()

                ocp.ocp_solver.set(ocp.N, 'lbx', np.array([q_min, 0.,  0.])) 
                ocp.ocp_solver.set(ocp.N, 'ubx', np.array([q_min, 0.,  1e-2])) 

                ocp.ocp_solver.set(ocp.N, 'x', np.array([q_min, 0.,  1e-2]))
                ocp.ocp_solver.set(ocp.N, 'p', p)

                for i, tau in enumerate(np.linspace(0, 1, ocp.N)):
                    prev_sol = x_sol[i+1]
                    x_guess = np.array([prev_sol[0], prev_sol[1], 1e-2])
                    # x_guess = (1-tau)*np.array([x0[0], x0[1], 2e-3]) + tau*np.array([q_max, 0., 2e-3])
                    u_guess = u_sol[i+1]
                    ocp.ocp_solver.set(i, 'x', x_guess)
                    ocp.ocp_solver.set(i, 'u', u_guess)
                    ocp.ocp_solver.set(i, 'p', p)
                    ocp.ocp_solver.set(i, 'lbx', np.array([q_min, v_min,  0.])) 
                    ocp.ocp_solver.set(i, 'ubx', np.array([q_max, v_max,  1e-2])) 

                ocp.ocp_solver.set(0, 'lbx', np.array([x0[0], x0[1],  0.])) 
                ocp.ocp_solver.set(0, 'ubx', np.array([x0[0], x0[1],  1e-2])) 

                status = ocp.ocp_solver.solve()
                ocp.ocp_solver.print_statistics()
                print('min time')

                print(status)

                if status == 0:
                    if x_sym[0] < q_min:
                        x_sym[0] = q_min
                    if x_sym[1] < v_min:
                        x_sym[1] = v_min

                    X_save = np.append(X_save, [[x_sym[0], x_sym[1], 0]], axis = 0)

                    dt_sym = ocp.ocp_solver.get(0, "x")[2]

                    for f in range(ocp.N):
                        if x_sym[0] >= q_min and x_sym[0] <= q_max and x_sym[1] <= v_max and x_sym[1] >= v_min:
                            u_sym = ocp.ocp_solver.get(f, "u")     
                            sim.acados_integrator.set("u", u_sym)
                            sim.acados_integrator.set("x", x_sym)
                            sim.acados_integrator.set("T", dt_sym)
                            sim.acados_integrator.solve()
                            x_sym = sim.acados_integrator.get("x")
                            X_save = np.append(X_save, [[x_sym[0], x_sym[1], 0]], axis = 0)

                        current_val = ocp.ocp_solver.get(f, "x")
                        X_save = np.append(X_save, [[current_val[0], current_val[1], 1]], axis = 0)
                    
                    current_val = ocp.ocp_solver.get(ocp.N, "x")
                    X_save = np.append(X_save, [[current_val[0], current_val[1], 1]], axis = 0)
                else:
                    X_save = np.append(X_save, [[x_sym[0], x_sym[1], 0]], axis = 0)
                    X_save = np.append(X_save, [[x0[0], x0[1], 1]], axis = 0)

    # Initialization of the SVM classifier:
    clf = svm.SVC(C=1e6, kernel='rbf')
    clf.fit(X_save[:,:2], X_save[:,2])
            
    plt.figure()
    plt.scatter(
        X_save[:,0], X_save[:,1], c =X_save[:,2], marker=".", cmap=plt.cm.Paired
    )
    h = 0.01
    x_min, x_max = q_min-(q_max-q_min)/100, q_max+(q_max-q_min)/100
    y_min, y_max = v_min-(v_max-v_min)/100, v_max+(v_max-v_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    out = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    out = out.reshape(xx.shape)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.xlabel("Initial position [rad]")
    plt.ylabel("Initial velocity [rad/s]")
    plt.grid(True)

    plt.show()