import time
from utils import plot_pendulum
import scipy.linalg as lin
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template import AcadosModel
from casadi import SX, vertcat, sin
import matplotlib.pyplot as plt
from sklearn import svm

if __name__ == "__main__":

    # constants
    m = 0.5  # mass of the ball [kg]
    g = 9.81  # gravity constant [m/s^2]
    d = 0.3  # length of the rod [m]
    b = 0.01  # damping

    # states
    theta = SX.sym("theta")
    dtheta = SX.sym("dtheta")
    x = vertcat(theta, dtheta)

    # controls
    F = SX.sym("F")
    u = vertcat(F)

    # parameters
    # weight = SX.sym("weight") 
    # p = vertcat(weight)

    # dynamics
    f_expl = vertcat(dtheta, (m * g * d * sin(theta) + F - b * dtheta) / (d * d * m))

    model = AcadosModel()

    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    #model.p = p
    model.name = "pendulum_time_opt"

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # times
    Tf = 1.
    Tf = Tf
    N = int(100 * Tf)

    # prediction horizon
    ocp.solver_options.tf = Tf

    # dimensions
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N

    # ocp model
    ocp.model = model

    # set cost
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    ocp.model.cost_expr_ext_cost_0 = - dtheta # weight * x[1]
    ocp.model.cost_expr_ext_cost = - dtheta # weight * x[1]
    ocp.model.cost_expr_ext_cost_e = 0
    #ocp.parameter_values = np.array([1.])

    # set constraints
    Fmax = 3
    thetamax = np.pi / 2
    thetamin = 0.0
    dthetamax = 10.0

    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbx = np.array([thetamin, -dthetamax])
    ocp.constraints.ubx = np.array([thetamax, dthetamax])
    ocp.constraints.idxbx = np.array([0, 1])
    ocp.constraints.lbx_e = np.array([thetamax, 0.])
    ocp.constraints.ubx_e = np.array([thetamax, 0.])
    ocp.constraints.idxbx_e = np.array([0, 1])
    ocp.constraints.lbx_0 = np.array([thetamin, -dthetamax])
    ocp.constraints.ubx_0 = np.array([thetamin, dthetamax])
    ocp.constraints.idxbx_0 = np.array([0, 1])

    # options
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.exact_hess_constr = 0
    ocp.solver_options.exact_hess_cost = 0
    ocp.solver_options.exact_hess_dyn = 0
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.nlp_solver_max_iter = 1000
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.alpha_reduction = 0.3
    ocp.solver_options.alpha_min = 1e-2
    ocp.solver_options.levenberg_marquardt = 1e-2
    # ocp.solver_options.sim_method_num_steps = 4
    # ocp.solver_options.line_search_use_sufficient_descent = 1

    # Solver
    ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    X_save = np.array([[(thetamin+thetamax)/2, 0., 1]])
    eps = 1e-1

    ocp_solver.reset()

    gr = np.load("goodres.npy")

    x_guess = np.array([thetamax, dthetamax])

    for i, tau in enumerate(np.linspace(0, 1, N)):
        # x_guess = np.array([(1-tau)*thetamin + tau*thetamax, dthetamax])
        # x_guess = (1-tau)*np.array([thetamin, dthetamax]) + tau*np.array([thetamax, 0.])
        ocp_solver.set(i, 'x', x_guess)
        #ocp_solver.set(i, 'u', np.array([-tau*Fmax]))
        #ocp_solver.set(i, 'p', np.array([-1.]))

    status = ocp_solver.solve()
    ocp_solver.print_statistics()
    print(ocp_solver.get_cost())

    if status == 0:
        for f in range(N+1):
            current_val = ocp_solver.get(f, "x")
            X_save = np.append(X_save, [[current_val[0], current_val[1], 1]], axis = 0)
            X_save = np.append(X_save, [[current_val[0], current_val[1] + eps, 0]], axis = 0)

    ocp_solver.reset()

    for i in range(N+1):
        ocp_solver.set(i, 'x', gr[N+1 + i,:2])

    print(ocp_solver.get_cost())

    del ocp, ocp_solver

    # ---------------------------------------------------

    # constants
    m = 0.5  # mass of the ball [kg]
    g = 9.81  # gravity constant [m/s^2]
    d = 0.3  # length of the rod [m]
    b = 0.01  # damping

    # states
    theta = SX.sym("theta")
    dtheta = SX.sym("dtheta")
    x = vertcat(theta, dtheta)

    # controls
    F = SX.sym("F")
    u = vertcat(F)

    # xdot
    theta_dot = SX.sym("theta_dot")
    dtheta_dot = SX.sym("dtheta_dot")
    xdot = vertcat(theta_dot, dtheta_dot)

    # parameters
    # weight = SX.sym("weight") 
    # p = vertcat(weight)

    # dynamics
    f_expl = vertcat(dtheta, (m * g * d * sin(theta) + F - b * dtheta) / (d * d * m))
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.xdot = xdot
    model.x = x
    model.u = u
    #model.p = p
    model.name = "pendulum_time_opt"

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # times
    Tf = 1.
    Tf = Tf
    N = int(100 * Tf)

    # prediction horizon
    ocp.solver_options.tf = Tf

    # dimensions
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N

    # ocp model
    ocp.model = model

    # set cost
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    ocp.model.cost_expr_ext_cost_0 = dtheta # weight * x[1]
    ocp.model.cost_expr_ext_cost = dtheta # weight * x[1]
    ocp.model.cost_expr_ext_cost_e = 0
    #ocp.parameter_values = np.array([1.])

    # set constraints
    Fmax = 3
    thetamax = np.pi / 2
    thetamin = 0.0
    dthetamax = 10.0

    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbx = np.array([thetamin, -dthetamax])
    ocp.constraints.ubx = np.array([thetamax, dthetamax])
    ocp.constraints.idxbx = np.array([0, 1])
    ocp.constraints.lbx_e = np.array([thetamin, 0.])
    ocp.constraints.ubx_e = np.array([thetamin, 0.])
    ocp.constraints.idxbx_e = np.array([0, 1])
    ocp.constraints.lbx_0 = np.array([thetamax, -dthetamax])
    ocp.constraints.ubx_0 = np.array([thetamax, dthetamax])
    ocp.constraints.idxbx_0 = np.array([0, 1])

    # options
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.exact_hess_constr = 0
    ocp.solver_options.exact_hess_dyn = 0
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.nlp_solver_max_iter = 1000
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.alpha_reduction = 0.3
    ocp.solver_options.alpha_min = 1e-2
    ocp.solver_options.levenberg_marquardt = 1e-2
    # ocp.solver_options.sim_method_num_steps = 4
    # ocp.solver_options.line_search_use_sufficient_descent = 1

    # Solver
    ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    ocp_solver.reset()

    x_guess = np.array([thetamin, -dthetamax])

    for i, tau in enumerate(np.linspace(0, 1, N)):
        # x_guess = np.array([tau*thetamin + (1-tau)*thetamax, -dthetamax])
        # x_guess = (1-tau)*np.array([thetamax, -dthetamax]) + tau*np.array([thetamin, 0.])
        ocp_solver.set(i, 'x', x_guess)
        #ocp_solver.set(i, 'u', np.array([0.]))

    status = ocp_solver.solve()
    ocp_solver.print_statistics()
    print(ocp_solver.get_cost())

    if status == 0:
        for f in range(N+1):
            current_val = ocp_solver.get(f, "x")
            X_save = np.append(X_save, [[current_val[0], current_val[1], 1]], axis = 0)
            X_save = np.append(X_save, [[current_val[0], current_val[1] - eps, 0]], axis = 0)

    ocp_solver.reset()

    for i in range(N+1):
        ocp_solver.set(i, 'x', gr[i,:2])

    print(ocp_solver.get_cost())


    clf = svm.SVC(C=1e6, kernel='rbf')
    clf.fit(X_save[:,:2], X_save[:,2])

    v_max = dthetamax
    v_min = -dthetamax
    q_max = thetamax
    q_min = thetamin
            
    plt.figure()
    plt.scatter(
        X_save[:,0], X_save[:,1], c =X_save[:,2], marker=".", cmap=plt.cm.Paired
    )
    h = 0.01
    x_min, x_max = q_min-(q_max-q_min)/100, q_max+(q_max-q_min)/100
    y_min, y_max = v_min-(v_max-v_min)/100, v_max+(v_max-v_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #inp = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    #out = model(inp)
    #y_pred = np.argmax(out.detach().numpy(), axis=1)
    #Z = y_pred.reshape(xx.shape)
    #plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    out = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    out = out.reshape(xx.shape)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.xlabel("Initial position [rad]")
    plt.ylabel("Initial velocity [rad/s]")
    plt.grid(True)

    plt.show()