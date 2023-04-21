import scipy.linalg as lin
from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, cos, sin, exp, norm_2, fmax, tanh
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.optimize import fsolve
import torch


class OCPtriplependulum:
    def __init__(self, mean, std, params):

        # --------------------SET MODEL--------------------
        # -------------------------------------------------
        model_name = "double_pendulum_ode"

        # constants
        self.m1 = 0.4  # mass of the first link [kself.g]
        self.m2 = 0.4  # mass of the second link [kself.g]
        self.m3 = 0.4  # mass of the third link [kself.g]
        self.g = 9.81  # self.gravity constant [m/s^2]
        self.l1 = 0.8  # lenself.gth of the first link [m]
        self.l2 = 0.8  # lenself.gth of the second link [m]
        self.l3 = 0.8  # lenself.gth of the second link [m]

        # states
        theta1 = SX.sym("theta1")
        theta2 = SX.sym("theta2")
        theta3 = SX.sym("theta3")
        dtheta1 = SX.sym("dtheta1")
        dtheta2 = SX.sym("dtheta2")
        dtheta3 = SX.sym("dtheta3")
        self.x = vertcat(theta1, theta2, theta3, dtheta1, dtheta2, dtheta3)

        # controls
        C1 = SX.sym("C1")
        C2 = SX.sym("C2")
        C3 = SX.sym("C3")
        u = vertcat(C1, C2, C3)

        # parameters
        p = []

        # dynamics
        f_expl = vertcat(
            dtheta1,
            dtheta2,
            dtheta3,
            (-self.g * self.l1 * self.l2 * self.l3 * self.m1 * self.m3 * sin(-2 * theta3 + 2 * theta2 + theta1) - self.g * self.l1 * self.l2 * self.l3 * self.m1 * self.m3 * sin(2 * theta3 - 2 * theta2 + theta1) + 2 * C1 * self.l2 * self.l3 * self.m3 * cos(-2 * theta3 + 2 * theta2) + 2 * dtheta1 ** 2 * self.l1 ** 2 * self.l2 * self.l3 * self.m2 * (self.m2 + self.m3) * sin(-2 * theta2 + 2 * theta1) - 2 * C3 * self.l1 * self.l2 * (self.m2 + self.m3) * cos(-2 * theta2 + theta1 + theta3) - 2 * C2 * self.l1 * self.l3 * self.m3 * cos(-2 * theta3 + theta2 + theta1) + 2 * self.l1 * self.l2 * self.l3 ** 2 * self.m2 * self.m3 * dtheta3 ** 2 * sin(-2 * theta2 + theta1 + theta3) + 2 * C3 * self.l1 * self.l2 * (self.m2 + self.m3) *
             cos(theta1 - theta3) + 2 * (C2 * self.l1 * (self.m3 + 2 * self.m2) * cos(-theta2 + theta1) + (self.g * self.l1 * self.m2 * (self.m2 + self.m3) * sin(-2 * theta2 + theta1) + 2 * dtheta2 ** 2 * self.l1 * self.l2 * self.m2 * (self.m2 + self.m3) * sin(-theta2 + theta1) + self.m3 * dtheta3 ** 2 * sin(theta1 - theta3) * self.l1 * self.l3 * self.m2 + self.g * self.l1 * (self.m2 ** 2 + (self.m3 + 2 * self.m1) * self.m2 + self.m1 * self.m3) * sin(theta1) - C1 * (self.m3 + 2 * self.m2)) * self.l2) * self.l3) / self.l1 ** 2 / self.l3 / (self.m2 * (self.m2 + self.m3) * cos(-2 * theta2 + 2 * theta1) + self.m1 * self.m3 * cos(-2 * theta3 + 2 * theta2) - self.m2 ** 2 + (-self.m3 - 2 * self.m1) * self.m2 - self.m1 * self.m3) / self.l2 / 2,
            (-2 * C3 * self.l1 * self.l2 * (self.m2 + self.m3) * cos(2 * theta1 - theta3 - theta2) - 2 * self.l1 * self.l2 * self.l3 ** 2 * self.m2 * self.m3 * dtheta3 ** 2 * sin(2 * theta1 - theta3 - theta2) + self.g * self.l1 * self.l2 * self.l3 * self.m1 * self.m3 * sin(theta2 + 2 * theta1 - 2 * theta3) - self.g * self.l1 * self.l3 * ((self.m1 + 2 * self.m2) * self.m3 + 2 * self.m2 * (self.m1 + self.m2)) * self.l2 * sin(-theta2 + 2 * theta1) - 2 * dtheta2 ** 2 * self.l1 * self.l2 ** 2 * self.l3 * self.m2 * (self.m2 + self.m3) * sin(-2 * theta2 + 2 * theta1) + 2 * C2 * self.l1 * self.l3 * self.m3 * cos(-2 * theta3 + 2 * theta1) + 2 * self.l1 * self.l2 ** 2 * self.l3 * self.m1 * self.m3 * dtheta2 ** 2 * sin(-2 * theta3 + 2 * theta2) - 2 * C1 * self.l2 * self.l3 * self.m3 * cos(-2 * theta3 + theta2 + theta1) + 2 * self.l1 ** 2 * self.l2 * self.l3 * self.m1 * self.m3 * dtheta1 ** 2 * sin(-2 * theta3 +
             theta2 + theta1) - 2 * self.l1 ** 2 * self.l3 * dtheta1 ** 2 * ((self.m1 + 2 * self.m2) * self.m3 + 2 * self.m2 * (self.m1 + self.m2)) * self.l2 * sin(-theta2 + theta1) + 2 * C3 * self.l1 * self.l2 * (self.m3 + 2 * self.m1 + self.m2) * cos(-theta3 + theta2) + (2 * C1 * self.l2 * (self.m3 + 2 * self.m2) * cos(-theta2 + theta1) + self.l1 * (4 * dtheta3 ** 2 * self.m3 * self.l3 * (self.m1 + self.m2 / 2) * self.l2 * sin(-theta3 + theta2) + self.g * self.m3 * self.l2 * self.m1 * sin(-2 * theta3 + theta2) + self.g * ((self.m1 + 2 * self.m2) * self.m3 + 2 * self.m2 * (self.m1 + self.m2)) * self.l2 * sin(theta2) - 2 * C2 * (self.m3 + 2 * self.m1 + 2 * self.m2))) * self.l3) / (self.m2 * (self.m2 + self.m3) * cos(-2 * theta2 + 2 * theta1) + self.m1 * self.m3 * cos(-2 * theta3 + 2 * theta2) + (-self.m1 - self.m2) * self.m3 - 2 * self.m1 * self.m2 - self.m2 ** 2) / self.l1 / self.l3 / self.l2 ** 2 / 2,
            (-2 * self.m3 * C2 * self.l1 * self.l3 * (self.m2 + self.m3) * cos(2 * theta1 - theta3 - theta2) + self.g * self.m3 * self.l1 * self.l2 * self.l3 * self.m1 * (self.m2 + self.m3) * sin(2 * theta1 + theta3 - 2 * theta2) + 2 * C3 * self.l1 * self.l2 * (self.m2 + self.m3) ** 2 * cos(-2 * theta2 + 2 * theta1) - self.g * self.m3 * self.l1 * self.l2 * self.l3 * self.m1 * (self.m2 + self.m3) * sin(2 * theta1 - theta3) - self.g * self.m3 * self.l1 * self.l2 * self.l3 * self.m1 * (self.m2 + self.m3) * sin(-theta3 + 2 * theta2) - 2 * self.l1 * self.l2 * self.l3 ** 2 * self.m1 * self.m3 ** 2 * dtheta3 ** 2 * sin(-2 * theta3 + 2 * theta2) - 2 * C1 * self.l2 * self.l3 * self.m3 * (self.m2 + self.m3) * cos(-2 * theta2 + theta1 + theta3) + 2 * self.m3 * dtheta1 ** 2 * self.l1 ** 2 *
             self.l2 * self.l3 * self.m1 * (self.m2 + self.m3) * sin(-2 * theta2 + theta1 + theta3) + 2 * self.m3 * C2 * self.l1 * self.l3 * (self.m3 + 2 * self.m1 + self.m2) * cos(-theta3 + theta2) + (self.m2 + self.m3) * (2 * C1 * self.l3 * self.m3 * cos(theta1 - theta3) + self.l1 * (-2 * self.m3 * dtheta1 ** 2 * self.l1 * self.l3 * self.m1 * sin(theta1 - theta3) - 4 * self.m3 * dtheta2 ** 2 * sin(-theta3 + theta2) * self.l2 * self.l3 * self.m1 + self.g * self.m3 * sin(theta3) * self.l3 * self.m1 - 2 * C3 * (self.m3 + 2 * self.m1 + self.m2))) * self.l2) / self.m3 / (self.m2 * (self.m2 + self.m3) * cos(-2 * theta2 + 2 * theta1) + self.m1 * self.m3 * cos(-2 * theta3 + 2 * theta2) + (-self.m1 - self.m2) * self.m3 - 2 * self.m1 * self.m2 - self.m2 ** 2) / self.l1 / self.l3 ** 2 / self.l2 / 2,
        )

        self.model = AcadosModel()

        self.model.f_expl_expr = f_expl
        self.model.x = self.x
        self.model.u = u
        self.model.p = p
        self.model.name = model_name
        # -------------------------------------------------

        # ---------------------SET OCP---------------------
        # -------------------------------------------------
        self.ocp = AcadosOcp()

        # time
        self.N = 1
        self.Tf = 1e-2 * self.N
        self.ocp.solver_options.tf = self.Tf

        # dimensions
        self.nx = self.model.x.size()[0]
        nu = self.model.u.size()[0]
        ny = self.nx + nu
        ny_e = self.nx

        self.ocp.dims.N = self.N

        # ocp model
        self.ocp.model = self.model

        # set constraints
        self.Cmax = 10.
        self.thetamax = np.pi / 4 + np.pi
        self.thetamin = - np.pi / 4 + np.pi
        self.dthetamax = 10.

        self.ocp.constraints.lbu = np.array([-self.Cmax, -self.Cmax, -self.Cmax])
        self.ocp.constraints.ubu = np.array([self.Cmax, self.Cmax, self.Cmax])
        self.ocp.constraints.idxbu = np.array([0, 1, 2])
        self.ocp.constraints.lbu_0 = np.array([-self.Cmax, -self.Cmax, -self.Cmax])
        self.ocp.constraints.ubu_0 = np.array([self.Cmax, self.Cmax, self.Cmax])
        self.ocp.constraints.idxbu_0 = np.array([0, 1, 2])
        self.ocp.constraints.lbx_0 = np.array(
            [self.thetamin, self.thetamin, self.thetamin, -
                self.dthetamax, -self.dthetamax, -self.dthetamax]
        )
        self.ocp.constraints.ubx_0 = np.array(
            [self.thetamax, self.thetamax, self.thetamax, self.dthetamax, self.dthetamax, self.dthetamax]
        )
        self.ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5])

        self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.hessian_approx = 'EXACT'
        self.ocp.solver_options.exact_hess_constr = 0
        # self.ocp.solver_options.exact_hess_cost = 0
        self.ocp.solver_options.exact_hess_dyn = 0
        # self.ocp.solver_options.nlp_solver_tol_stat = 1e-4
        # self.ocp.solver_options.nlp_solver_tol_comp = 1e-2
        # self.ocp.solver_options.tol = 1e-2
        self.ocp.solver_options.qp_solver_iter_max = 100
        self.ocp.solver_options.nlp_solver_max_iter = 1000
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        self.ocp.solver_options.alpha_reduction = 0.3
        self.ocp.solver_options.alpha_min = 1e-2
        self.ocp.solver_options.levenberg_marquardt = 1e-5

        # cost
        self.ocp.cost.cost_type_e = 'EXTERNAL'
        self.ocp.model.cost_expr_ext_cost_e = self.nn_decisionfunction(params, mean, std, self.x)

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

        # -------------------------------------------------

    def compute_problem(self, x0):

        self.ocp_solver.reset()

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        x_guess = np.array([x0[0]+x0[3]*1e-2, x0[1]+x0[4]*1e-2, x0[2]+x0[5]*1e-2, x0[3]*0.9, x0[4]*0.9, x0[5]*0.9])

        self.ocp_solver.set(0, "x", x0)
        self.ocp_solver.set(self.N, "x", x_guess)

        status = self.ocp_solver.solve()

        if status == 0:
            return 1
        else:
            return 0
        
    def nn_decisionfunction(self, params, mean, std, x):

        out = (x - mean) / std
        it = 2

        for param in params:
            param = SX(param.tolist())
            if it % 2 == 0:
                out = param @ out
            else:
                out = param + out

                if it == 3:
                    out = fmax(0.0, out)
                elif it == 5:
                    out = fmax(0.0, out)
                # else:
                #    out = 1 / (1 + exp(-out))

            it += 1

        return vertcat(out[0])
        # return vertcat(out[1]-out[0])
        