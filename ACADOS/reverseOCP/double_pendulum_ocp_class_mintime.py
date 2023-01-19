import scipy.linalg as lin
from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, cos, sin, exp, norm_2, fmax, tanh
import matplotlib.pyplot as plt
import time

class OCPdoublependulumR:
    def __init__(self):

        # --------------------SET MODEL--------------------
        # -------------------------------------------------
        model_name = "double_pendulum_ode"

        # constants
        self.m1 = 0.4  # mass of the first link [kself.g]
        self.m2 = 0.4  # mass of the second link [kself.g]
        self.g = 9.81  # self.gravity constant [m/s^2]
        self.l1 = 0.8  # lenself.gth of the first link [m]
        self.l2 = 0.8  # lenself.gth of the second link [m]

        # states
        theta1 = SX.sym("theta1")
        theta2 = SX.sym("theta2")
        dtheta1 = SX.sym("dtheta1")
        dtheta2 = SX.sym("dtheta2")
        dt = SX.sym('dt')
        self.x = vertcat(theta1, theta2, dtheta1, dtheta2, dt)

        # controls
        C1 = SX.sym("C1")
        C2 = SX.sym("C2")
        u = vertcat(C1, C2)

        # parameters
        w1 = SX.sym("w1") 
        w2 = SX.sym("w2") 
        p = vertcat(w1, w2)

        # dynamics
        f_expl = dt*vertcat(
            dtheta1,
            dtheta2,
            (
                self.l1**2
                * self.l2
                * self.m2
                * dtheta1**2
                * sin(-2 * theta2 + 2 * theta1)
                + 2 * C2 * cos(-theta2 + theta1) * self.l1
                + 2
                * (
                    self.g * sin(-2 * theta2 + theta1) * self.l1 * self.m2 / 2
                    + sin(-theta2 + theta1) * dtheta2**2 * self.l1 * self.l2 * self.m2
                    + self.g * self.l1 * (self.m1 + self.m2 / 2) * sin(theta1)
                    - C1
                )
                * self.l2
            )
            / self.l1**2
            / self.l2
            / (self.m2 * cos(-2 * theta2 + 2 * theta1) - 2 * self.m1 - self.m2),
            (
                -self.g
                * self.l1
                * self.l2
                * self.m2
                * (self.m1 + self.m2)
                * sin(-theta2 + 2 * theta1)
                - self.l1
                * self.l2**2
                * self.m2**2
                * dtheta2**2
                * sin(-2 * theta2 + 2 * theta1)
                - 2
                * dtheta1**2
                * self.l1**2
                * self.l2
                * self.m2
                * (self.m1 + self.m2)
                * sin(-theta2 + theta1)
                + 2 * C1 * cos(-theta2 + theta1) * self.l2 * self.m2
                + self.l1
                * (self.m1 + self.m2)
                * (sin(theta2) * self.g * self.l2 * self.m2 - 2 * C2)
            )
            / self.l2**2
            / self.l1
            / self.m2
            / (self.m2 * cos(-2 * theta2 + 2 * theta1) - 2 * self.m1 - self.m2),
            0.
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

        # set dimensions
        self.N = 100
        self.ocp.dims.N = self.N
        self.ocp.solver_options.tf = self.N
        self.nx = self.model.x.size()[0]
        nu = self.model.u.size()[0]
        ny = self.nx + nu
        ny_e = self.nx

        # ocp model
        self.ocp.model = self.model

        # cost
        self.ocp.cost.cost_type_0 = 'EXTERNAL'
        self.ocp.cost.cost_type = 'EXTERNAL'
        self.ocp.cost.cost_type_e = 'EXTERNAL'

        self.ocp.model.cost_expr_ext_cost_0 = dt + w1 * dtheta1 + w2 * dtheta2 
        self.ocp.model.cost_expr_ext_cost = dt + w1 * dtheta1 + w2 * dtheta2 
        self.ocp.model.cost_expr_ext_cost_e = 0.
        self.ocp.parameter_values = np.array([0., 0.])

        # set constraints
        self.Cmax = 10
        self.thetamax = np.pi / 2 + np.pi
        self.thetamin = np.pi
        self.dthetamax = 5.0

        self.ocp.constraints.lbu = np.array([-self.Cmax, -self.Cmax])
        self.ocp.constraints.ubu = np.array([self.Cmax, self.Cmax])
        self.ocp.constraints.idxbu = np.array([0, 1])
        self.ocp.constraints.lbx = np.array(
            [self.thetamin, self.thetamin, -self.dthetamax, -self.dthetamax, 0.]
        )
        self.ocp.constraints.ubx = np.array(
            [self.thetamax, self.thetamax, self.dthetamax, self.dthetamax, 1e-2]
        )
        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4])
        self.ocp.constraints.lbx_e = np.array(
            [self.thetamin, self.thetamin, -self.dthetamax, -self.dthetamax, 0.]
        ) # not necessary
        self.ocp.constraints.ubx_e = np.array(
            [self.thetamax, self.thetamax, self.dthetamax, self.dthetamax, 1e-2]
        ) # not necessary
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2, 3, 4])
        self.ocp.constraints.lbx_0 = np.array([self.thetamin, self.thetamin, -self.dthetamax, -self.dthetamax, 0.]) # not necessary
        self.ocp.constraints.ubx_0 = np.array([self.thetamax, self.thetamax, self.dthetamax, self.dthetamax, 1e-2]) # not necessary
        self.ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4])

        # -------------------------------------------------

        self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.hessian_approx = 'EXACT'
        # self.ocp.solver_options.exact_hess_constr = 0
        # self.ocp.solver_options.exact_hess_dyn = 0
        self.ocp.solver_options.tol = 1e-4
        self.ocp.solver_options.qp_tol = 1e-6
        self.ocp.solver_options.qp_solver_iter_max = 1000
        self.ocp.solver_options.nlp_solver_max_iter = 10000
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        self.ocp.solver_options.alpha_reduction = 0.3
        self.ocp.solver_options.alpha_min = 1e-2
        self.ocp.solver_options.levenberg_marquardt = 1e-2
        # self.ocp.solver_options.sim_method_num_steps = 4
        # self.ocp.solver_options.regularize_method = "PROJECT"
        # self.ocp.solver_options.nlp_solver_step_length = 0.01

class OCPdoublependulumRINIT(OCPdoublependulumR):
    def __init__(self):

        # inherit initialization
        super().__init__()

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")
