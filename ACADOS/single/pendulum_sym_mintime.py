import time
from utils import plot_pendulum
import scipy.linalg as lin
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template import AcadosModel
from casadi import SX, vertcat, sin
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # constants
    m = 0.5  # mass of the ball [kg]
    g = 9.81  # gravity constant [m/s^2]
    d = 0.3  # length of the rod [m]

    # states
    theta = SX.sym("theta")
    dtheta = SX.sym("dtheta")
    x = vertcat(theta, dtheta)

    # controls
    F = SX.sym("F")
    dt = SX.sym('dt')
    u = vertcat(F, dt)

    # dynamics
    f_expl = dt*vertcat(dtheta, (m * g * d * sin(theta) + F) / (d * d * m))

    model = AcadosModel()

    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.name = "pendulum_time_opt"

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set dimensions
    N = 100
    ocp.dims.N = N
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx

    # ocp model
    ocp.model = model

    # set prediction horizon
    ocp.solver_options.tf = N

    # set cost
    # ocp.cost.W_0 = 2 * np.diag([0., 1., 0.])
    # ocp.cost.W = 2 * np.diag([0., 1., 0.])
    # ocp.cost.W_e = 2 * np.diag([0., 0.])

    # ocp.cost.cost_type = "LINEAR_LS"
    # ocp.cost.cost_type_e = "LINEAR_LS"

    # ocp.cost.Vx = np.zeros((ny, nx))
    # ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    # ocp.cost.Vu = np.zeros((ny, nu))
    # ocp.cost.Vu[2, 0] = 1.0
    # ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    ocp.model.cost_expr_ext_cost_0 = dt - x[1]
    ocp.model.cost_expr_ext_cost = dt - x[1]
    ocp.model.cost_expr_ext_cost_e = 0

    # set constraints
    Fmax = 3
    thetamax = np.pi / 2
    thetamin = 0.0
    dthetamax = 10.0

    ocp.constraints.lbu = np.array([-Fmax, 1e-3])
    ocp.constraints.ubu = np.array([+Fmax, 1e-2])
    ocp.constraints.idxbu = np.array([0, 1.])
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
    ocp.solver_options.exact_hess_dyn = 0
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_tol = 1e-6
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

    for i, tau in enumerate(np.linspace(0, 1, N)):
        x_guess = np.array([(1-tau)*thetamin + tau*thetamax, dthetamax])
        # x_guess = (1-tau)*x0 + tau*xe
        ocp_solver.set(i, 'x', x_guess)
        ocp_solver.set(i, 'u', np.array([-tau*Fmax, 1e-2]))
        
    start_time = time.time()

    status = ocp_solver.solve()
    ocp_solver.print_statistics()

    print("Execution time: %s seconds" % (time.time() - start_time))

    if status == 0:
        # get solution
        simX = np.ndarray((N + 1, nx))
        simU = np.ndarray((N, nu))

        for i in range(N):
            simX[i, :] = ocp_solver.get(i, "x")
            simU[i, :] = ocp_solver.get(i, "u")
        simX[N, :] = ocp_solver.get(N, "x")

        tot_time = np.sum(simU[:, 1])

        print(tot_time, simU[:, 1])

        plot_pendulum(np.linspace(0, tot_time, N + 1), Fmax, simU[:,0], simX, latexify=False)