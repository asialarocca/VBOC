import time
import scipy.linalg as lin
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
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
    w1 = SX.sym("w1") 
    w2 = SX.sym("w2") 
    p = vertcat(w1, w2)

    # dynamics
    f_expl = vertcat(dtheta, (m * g * d * sin(theta) + F - b * dtheta) / (d * d * m))

    model = AcadosModel()

    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.p = p
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

    # set constraints
    Fmax = 3
    thetamax = np.pi / 2
    thetamin = 0.0
    dthetamax = 10.0

    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbx = np.array([thetamin, 0.])
    ocp.constraints.ubx = np.array([thetamax, dthetamax])
    ocp.constraints.idxbx = np.array([0, 1])
    ocp.constraints.lbx_e = np.array([thetamax, 0.])
    ocp.constraints.ubx_e = np.array([thetamax, 0.])
    ocp.constraints.idxbx_e = np.array([0, 1])
    ocp.constraints.lbx_0 = np.array([thetamin, 0.])
    ocp.constraints.ubx_0 = np.array([thetamax, dthetamax])
    ocp.constraints.idxbx_0 = np.array([0, 1])

    # set cost
    ocp.cost.cost_type_0 = 'EXTERNAL'
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    ocp.model.cost_expr_ext_cost_0 = w1 / (thetamax - thetamin) * theta + w2 / dthetamax * dtheta
    ocp.model.cost_expr_ext_cost = 0.
    ocp.model.cost_expr_ext_cost_e = 0.
    ocp.parameter_values = np.array([-1., -10.])

    # options
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.exact_hess_constr = 0
    ocp.solver_options.exact_hess_cost = 0
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

    for i, tau in enumerate(np.linspace(0, 1, N)):
        x_guess = np.array([(1-tau)*thetamin + tau*thetamax, dthetamax])
        # x_guess = (1-tau)*np.array([thetamin, dthetamax]) + tau*np.array([thetamax, 0.])
        ocp_solver.set(i, 'x', x_guess)

    status = ocp_solver.solve()
    ocp_solver.print_statistics()

    if status == 0:
        for f in range(N+1):
            current_val = ocp_solver.get(f, "x")
            X_save = np.append(X_save, [[current_val[0], current_val[1], 1]], axis = 0)

    sim = AcadosSim()
    model.p = []
    sim.model = model
    sim.solver_options.T = 1e-2
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 3
    acados_integrator = AcadosSimSolver(sim)

    current_val = ocp_solver.get(0, "x")

    x1_eps = current_val[0] - eps * ocp.parameter_values[0] / (thetamax - thetamin)
    x2_eps = current_val[1] - eps * ocp.parameter_values[1] / dthetamax

    x_sym = np.array([x1_eps, x2_eps])
    X_save = np.append(X_save, [[x_sym[0], x_sym[1], 0]], axis = 0)

    if not (abs(current_val[0] - thetamax) < 1e-4 or abs(current_val[1] - dthetamax) < 1e-4):
        for f in range(N):
            u_sym = ocp_solver.get(f, "u")     
            acados_integrator.set("u", u_sym)
            acados_integrator.set("x", x_sym)
            acados_integrator.solve()
            x_sym = acados_integrator.get("x")
            X_save = np.append(X_save, [[x_sym[0], x_sym[1], 0]], axis = 0)

            if x_sym[0] <= thetamin or x_sym[0] >= thetamax or x_sym[1] <= -dthetamax or x_sym[1] >= dthetamax:
                break

    v_max = dthetamax
    v_min = -dthetamax
    q_max = thetamax
    q_min = thetamin
            
    plt.figure()

    plt.scatter(
        X_save[:,0], X_save[:,1], c =X_save[:,2], marker=".", cmap=plt.cm.Paired
    )

    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.xlabel("Initial position [rad]")
    plt.ylabel("Initial velocity [rad/s]")
    plt.grid(True)

    plt.show()