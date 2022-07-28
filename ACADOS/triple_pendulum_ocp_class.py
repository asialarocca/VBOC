import scipy.linalg as lin
from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, cos, sin, exp, norm_2, fmax, tanh
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.optimize import fsolve


class OCPtriplependulum:
    def __init__(self):

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

        # xdot
        theta1_dot = SX.sym("theta1_dot")
        theta2_dot = SX.sym("theta1_dot")
        theta3_dot = SX.sym("theta3_dot")
        dtheta1_dot = SX.sym("dtheta2_dot")
        dtheta2_dot = SX.sym("dtheta2_dot")
        dtheta3_dot = SX.sym("dtheta3_dot")
        xdot = vertcat(theta1_dot, theta2_dot, theta3_dot, dtheta1_dot, dtheta2_dot, dtheta3_dot)

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

        f_impl = xdot - f_expl

        self.model = AcadosModel()

        self.model.f_impl_expr = f_impl
        self.model.f_expl_expr = f_expl
        self.model.x = self.x
        self.model.xdot = xdot
        self.model.u = u
        self.model.p = p
        self.model.name = model_name
        # -------------------------------------------------

        # ---------------------SET OCP---------------------
        # -------------------------------------------------
        self.ocp = AcadosOcp()

        # dimensions
        self.Tf = 0.1
        self.ocp.solver_options.tf = self.Tf  # prediction horizon

        self.N = int(100 * self.Tf)
        self.ocp.dims.N = self.N

        self.nx = self.model.x.size()[0]
        self.nu = self.model.u.size()[0]
        ny = self.nx + self.nu
        ny_e = self.nx

        # # cost
        # Q = 2 * np.diag([0.0, 0.0, 0.0, 0.0])
        # # Q = 2 * np.diag([0.0, 0.0, 1e-2, 1e-2])
        # R = 2 * np.diag([0.0, 0.0])

        # self.ocp.cost.W_e = Q
        # self.ocp.cost.W = lin.block_diag(Q, R)

        # self.ocp.cost.cost_type = "LINEAR_LS"
        # self.ocp.cost.cost_type_e = "LINEAR_LS"

        # self.ocp.cost.Vx = np.zeros((ny, self.nx))
        # self.ocp.cost.Vx[: self.nx, : self.nx] = np.eye(self.nx)
        # self.ocp.cost.Vu = np.zeros((ny, nu))
        # self.ocp.cost.Vu[self.nx :, :nu] = np.eye(nu)
        # self.ocp.cost.Vx_e = np.eye(self.nx)

        # # reference
        # self.ocp.cost.yref = np.zeros((ny,))
        # self.ocp.cost.yref_e = np.zeros((ny_e,))

        # set constraints
        self.Cmax = 12
        self.thetamax = np.pi / 2 + np.pi
        self.thetamin = np.pi
        self.dthetamax = 5.0

        self.ocp.constraints.lbu = np.array([-self.Cmax, -self.Cmax, -self.Cmax])
        self.ocp.constraints.ubu = np.array([self.Cmax, self.Cmax, self.Cmax])
        self.ocp.constraints.idxbu = np.array([0, 1, 2])
        self.ocp.constraints.lbx = np.array(
            [self.thetamin, self.thetamin, self.thetamin, -
                self.dthetamax, -self.dthetamax, -self.dthetamax]
        )
        self.ocp.constraints.ubx = np.array(
            [self.thetamax, self.thetamax, self.thetamax, self.dthetamax, self.dthetamax, self.dthetamax]
        )
        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5])
        self.ocp.constraints.lbx_e = np.array(
            [self.thetamin, self.thetamin, self.thetamin, -
                self.dthetamax, -self.dthetamax, -self.dthetamax]
        )
        self.ocp.constraints.ubx_e = np.array(
            [self.thetamax, self.thetamax, self.thetamax, self.dthetamax, self.dthetamax, self.dthetamax]
        )
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2, 3, 4, 5])

        self.ocp.constraints.x0 = np.array(
            [self.thetamin, self.thetamin, self.thetamin, 0.0, 0.0, 0.0])

        # -------------------------------------------------

    def compute_problem(self, q0, v0):

        self.ocp_solver.reset()

        x0 = np.array([q0[0], q0[1], q0[2], v0[0], v0[1], v0[2]])

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        x_guess = np.array([q0[0], q0[1], q0[2], 0.0, 0.0, 0.0])

        for i in range(self.N + 1):
            self.ocp_solver.set(i, "x", x_guess)

        status = self.ocp_solver.solve()

        if status == 0:

            return 1
        elif status == 4:
            return 0
        else:
            return 2


class OCPtriplependulumINIT(OCPtriplependulum):
    def __init__(self):

        # inherit initialization
        super().__init__()

        # linear terminal constraints (zero final velocity)
        self.ocp.constraints.lbx_e[3] = 0.0
        self.ocp.constraints.lbx_e[4] = 0.0
        self.ocp.constraints.lbx_e[5] = 0.0
        self.ocp.constraints.ubx_e[3] = 0.0
        self.ocp.constraints.ubx_e[4] = 0.0
        self.ocp.constraints.ubx_e[5] = 0.0

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")


class OCPtriplependulumNN(OCPtriplependulum):
    def __init__(self, nn, mean, std):

        # inherit initialization
        super().__init__()

        # nonlinear terminal constraints
        self.model.con_h_expr_e = self.nn_decisionfunction(nn, self.x, mean, std)
        self.ocp.constraints.lh_e = np.array([0.5])
        self.ocp.constraints.uh_e = np.array([1.1])

        self.ocp.constraints.lbx_e = np.array(
            [self.thetamin, self.thetamin,  self.thetamin, -
                self.dthetamax, -self.dthetamax, -self.dthetamax]
        )
        self.ocp.constraints.ubx_e = np.array(
            [self.thetamax, self.thetamax, self.thetamax, self.dthetamax, self.dthetamax, self.dthetamax]
        )

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

    def nn_decisionfunction(self, nn, x, mean, std):

        out = (x-mean.item())/std.item()
        it = 2

        for param in nn.parameters():
            param = SX(param.tolist())
            if it % 2 == 0:
                out = param @ out
            else:
                out = param + out

                if it == 3:
                    out = fmax(0.0, out)
                elif it == 5:
                    out = tanh(out)
                else:
                    out = 1 / (1 + exp(-out))
            it += 1

        return out[1]
