import scipy.linalg as lin
from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, cos, sin, exp, norm_2, fmax, tanh
from scipy.integrate import odeint
from scipy.optimize import fsolve
import torch


class OCPdoublependulum:
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
        self.x = vertcat(theta1, theta2, dtheta1, dtheta2)

        # controls
        C1 = SX.sym("C1")
        C2 = SX.sym("C2")
        u = vertcat(C1, C2)

        # xdot
        theta1_dot = SX.sym("theta1_dot")
        theta2_dot = SX.sym("theta1_dot")
        dtheta1_dot = SX.sym("dtheta2_dot")
        dtheta2_dot = SX.sym("dtheta2_dot")
        xdot = vertcat(theta1_dot, theta2_dot, dtheta1_dot, dtheta2_dot)

        # parameters
        p = []

        # dynamics
        f_expl = vertcat(
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
        self.Tf = 1.
        self.ocp.solver_options.tf = self.Tf  # prediction horizon

        self.N = int(100 * self.Tf)
        self.ocp.dims.N = self.N

        self.nx = self.model.x.size()[0]
        self.nu = self.model.u.size()[0]
        ny = self.nx + self.nu
        ny_e = self.nx

        # cost
        Q = 2 * np.diag([0., 0., 1., 1.])
        R = 2 * np.diag([0.0, 0.0])

        self.ocp.cost.W_e = Q
        self.ocp.cost.W = lin.block_diag(Q, R)

        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        self.ocp.cost.Vx = np.zeros((ny, self.nx))
        self.ocp.cost.Vx[: self.nx, : self.nx] = np.eye(self.nx)
        self.ocp.cost.Vu = np.zeros((ny, self.nu))
        self.ocp.cost.Vu[self.nx:, :self.nu] = np.eye(self.nu)
        self.ocp.cost.Vx_e = np.eye(self.nx)

        # reference
        self.ocp.cost.yref = np.zeros((ny,))
        self.ocp.cost.yref_e = np.zeros((ny_e,))

        # set constraints
        self.Cmax = 10
        self.thetamax = np.pi / 4 + np.pi
        self.thetamin = - np.pi / 4 + np.pi
        self.dthetamax = 10.

        self.normal = np.array(
            [
                self.thetamax - self.thetamin,
                self.thetamax - self.thetamin,
                2 * self.dthetamax,
                2 * self.dthetamax,
            ]
        )

        self.ocp.constraints.lbu = np.array([-self.Cmax, -self.Cmax])
        self.ocp.constraints.ubu = np.array([self.Cmax, self.Cmax])
        self.ocp.constraints.idxbu = np.array([0, 1])
        self.ocp.constraints.lbx = np.array(
            [self.thetamin, self.thetamin, -self.dthetamax, -self.dthetamax]
        )
        self.ocp.constraints.ubx = np.array(
            [self.thetamax, self.thetamax, self.dthetamax, self.dthetamax]
        )
        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3])
        self.ocp.constraints.lbx_e = np.array(
            [self.thetamin, self.thetamin, -self.dthetamax, -self.dthetamax]
        )
        self.ocp.constraints.ubx_e = np.array(
            [self.thetamax, self.thetamax, self.dthetamax, self.dthetamax]
        )
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2, 3])
        self.ocp.constraints.lbu_0 = np.array([-self.Cmax, -self.Cmax])
        self.ocp.constraints.ubu_0 = np.array([self.Cmax, self.Cmax])
        self.ocp.constraints.idxbu_0 = np.array([0, 1])

        self.ocp.constraints.x0 = np.array([self.thetamin, self.thetamin, 0.0, 0.0])

        # -------------------------------------------------

    def set_bounds(self, q_min, q_max):
        self.ocp.constraints.lbx[0] = q_min[0]
        self.ocp.constraints.ubx[0] = q_max[0]
        self.ocp.constraints.lbx_e[0] = q_min[0]
        self.ocp.constraints.ubx_e[0] = q_max[0]
        self.ocp.constraints.lbx[1] = q_min[1]
        self.ocp.constraints.ubx[1] = q_max[1]
        self.ocp.constraints.lbx_e[1] = q_min[1]
        self.ocp.constraints.ubx_e[1] = q_max[1]

        for i in range(self.N):
            self.ocp_solver.constraints_set(i, "lbx", self.ocp.constraints.lbx)
            self.ocp_solver.constraints_set(i, "ubx", self.ocp.constraints.ubx)

        self.ocp_solver.constraints_set(self.N, "lbx", self.ocp.constraints.lbx_e)
        self.ocp_solver.constraints_set(self.N, "ubx", self.ocp.constraints.ubx_e)

    def compute_problem(self, q0, v0):

        self.ocp_solver.reset()

        x0 = np.array([q0[0], q0[1], v0[0], v0[1]])

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        x_guess = np.array([q0[0], q0[1], 0.0, 0.0])

        for i in range(self.N + 1):
            self.ocp_solver.set(i, "x", x_guess)

        status = self.ocp_solver.solve()

        if status == 0:
            return 1
        elif status == 4:
            return 0
        else:
            return 2

    def compute_problem_nnguess(self, q0, v0, model, mean, std):

        self.ocp_solver.reset()

        x0 = np.array([q0[0], q0[1], v0[0], v0[1]])

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        with torch.no_grad():
            inp = torch.Tensor([[q0[0], q0[1], v0[0], v0[1]]])
            inp = (inp - mean) / std
            out = model(inp)
            out = out * std + mean
            out = out.numpy()
        
        out = np.reshape(out, (self.N,self.nx))

        self.ocp_solver.set(0, "x", x0)

        for i in range(self.N):
            self.ocp_solver.set(i+1, "x", out[i])

        status = self.ocp_solver.solve()

        if status == 0:
            return 1
        elif status == 4:
            return 0
        else:
            return 2

    def compute_problem_withGUESS(self, q0, v0, simX_vec, simU_vec):

        self.ocp_solver.reset()

        x0 = np.array([q0[0], q0[1], v0[0], v0[1]])

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        x_guess = np.array([q0[0], q0[1], 0.0, 0.0])

        for i in range(self.N + 1):
            self.ocp_solver.set(i, "x", x_guess)

        status = self.ocp_solver.solve()

        if status == 0:
            return 1
        else:
            self.ocp_solver.reset()

            x0 = np.array([q0[0], q0[1], v0[0], v0[1]])

            self.ocp_solver.constraints_set(0, "lbx", x0)
            self.ocp_solver.constraints_set(0, "ubx", x0)

            dist_min = 1e3

            for k in range(simX_vec.shape[0]):
                # if np.siself.gn(simX_vec[k, 0, 2]) == np.siself.gn(v0[0]) and np.siself.gn(simX_vec[k, 0, 3]) == np.siself.gn(v0[1]):
                dist = np.linalg.norm(x0 / self.normal - simX_vec[k, 0, :] / self.normal)
                if dist < dist_min:
                    dist_min = dist
                    index = k

            for i in range(self.N):
                self.ocp_solver.set(i, "x", simX_vec[index, i, :])
                self.ocp_solver.set(i, "u", simU_vec[index, i, :])

            self.ocp_solver.set(self.N, "x", simX_vec[index, self.N, :])

            status = self.ocp_solver.solve()

            if status == 0:
                return 1
            elif status == 4:
                return 0
            else:
                return 2

    def compute_problem_withGUESSPID(self, q0, v0):

        self.ocp_solver.reset()

        x0 = np.array([q0[0], q0[1], v0[0], v0[1]])

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        times = np.linspace(0, self.Tf, self.N + 1)
        simX = odeint(self.model_pid, x0, times)

        simX[:, 0] = [
            self.thetamax if simX[i, 0] > self.thetamax else simX[i, 0]
            for i in range(self.N + 1)
        ]
        simX[:, 0] = [
            self.thetamin if simX[i, 0] < self.thetamin else simX[i, 0]
            for i in range(self.N + 1)
        ]
        simX[:, 1] = [
            self.thetamax if simX[i, 1] > self.thetamax else simX[i, 1]
            for i in range(self.N + 1)
        ]
        simX[:, 1] = [
            self.thetamin if simX[i, 1] < self.thetamin else simX[i, 1]
            for i in range(self.N + 1)
        ]
        simX[:, 2] = [
            np.sign(simX[i, 2]) * self.dthetamax
            if abs(simX[i, 2]) > self.dthetamax
            else simX[i, 2]
            for i in range(self.N + 1)
        ]
        simX[:, 3] = [
            np.sign(simX[i, 3]) * self.dthetamax
            if abs(simX[i, 3]) > self.dthetamax
            else simX[i, 3]
            for i in range(self.N + 1)
        ]

        for i in range(self.N + 1):
            self.ocp_solver.set(i, "x", simX[i, :])

        status = self.ocp_solver.solve()

        if status == 0:
            return 1
        elif status == 4:
            return 0
        else:
            return 2

    def model_pid(self, x, t):
        # c1, c2 = fsolve(self.gravity_comp, (0.0, 0.0), args=x)

        y1 = x[0]
        y2 = x[1]
        dy1dt = x[2]
        dy2dt = x[3]
        u1 = -100 * dy1dt  # + c1
        u2 = -10 * dy2dt  # + c2

        if u1 >= self.Cmax:
            u1 = self.Cmax
        elif u1 <= -self.Cmax:
            u1 = -self.Cmax
        if u2 >= self.Cmax:
            u2 = self.Cmax
        elif u2 <= -self.Cmax:
            u2 = -self.Cmax

        dy12dt2 = (
            (
                self.l1**2 * self.l2 * self.m2 * dy1dt**2 * sin(-2 * y2 + 2 * y1)
                + 2 * u2 * cos(-y2 + y1) * self.l1
                + 2
                * (
                    self.g * sin(-2 * y2 + y1) * self.l1 * self.m2 / 2
                    + sin(-y2 + y1) * dy2dt**2 * self.l1 * self.l2 * self.m2
                    + self.g * self.l1 * (self.m1 + self.m2 / 2) * sin(y1)
                    - u1
                )
                * self.l2
            )
            / self.l1**2
            / self.l2
            / (self.m2 * cos(-2 * y2 + 2 * y1) - 2 * self.m1 - self.m2)
        )
        dy22dt2 = (
            (
                -self.g
                * self.l1
                * self.l2
                * self.m2
                * (self.m1 + self.m2)
                * sin(-y2 + 2 * y1)
                - self.l1
                * self.l2**2
                * self.m2**2
                * dy2dt**2
                * sin(-2 * y2 + 2 * y1)
                - 2
                * dy1dt**2
                * self.l1**2
                * self.l2
                * self.m2
                * (self.m1 + self.m2)
                * sin(-y2 + y1)
                + 2 * u1 * cos(-y2 + y1) * self.l2 * self.m2
                + self.l1
                * (self.m1 + self.m2)
                * (sin(y2) * self.g * self.l2 * self.m2 - 2 * u2)
            )
            / self.l2**2
            / self.l1
            / self.m2
            / (self.m2 * cos(-2 * y2 + 2 * y1) - 2 * self.m1 - self.m2)
        )
        return [dy1dt, dy2dt, dy12dt2, dy22dt2]

    def gravity_comp(self, C, x0):
        y1 = x0[0]
        y2 = x0[1]
        u1, u2 = C

        return (
            (
                2 * u2 * cos(-y2 + y1) * self.l1
                + 2
                * (
                    self.g * sin(-2 * y2 + y1) * self.l1 * self.m2 / 2
                    + self.g * self.l1 * (self.m1 + self.m2 / 2) * sin(y1)
                    - u1
                )
                * self.l2
            )
            / self.l1**2
            / self.l2
            / (self.m2 * cos(-2 * y2 + 2 * y1) - 2 * self.m1 - self.m2),
            (
                -self.g
                * self.l1
                * self.l2
                * self.m2
                * (self.m1 + self.m2)
                * sin(-y2 + 2 * y1)
                + 2 * u1 * cos(-y2 + y1) * self.l2 * self.m2
                + self.l1
                * (self.m1 + self.m2)
                * (sin(y2) * self.g * self.l2 * self.m2 - 2 * u2)
            )
            / self.l2**2
            / self.l1
            / self.m2
            / (self.m2 * cos(-2 * y2 + 2 * y1) - 2 * self.m1 - self.m2),
        )


class OCPdoublependulumINIT(OCPdoublependulum):
    def __init__(self):

        # inherit initialization
        super().__init__()

        # linear terminal constraints (zero final velocity)
        self.ocp.constraints.lbx_e[2] = 0.0
        self.ocp.constraints.lbx_e[3] = 0.0
        self.ocp.constraints.ubx_e[2] = 0.0
        self.ocp.constraints.ubx_e[3] = 0.0

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")


class OCPdoublependulumSVM(OCPdoublependulum):
    def __init__(self, clf, X_iter):

        # inherit initialization
        super().__init__()

        # nonlinear terminal constraints (svm)
        self.model.con_h_expr_e = self.clf_decisionfunction(clf, X_iter, self.x)
        self.ocp.constraints.lh_e = np.array([0.0])
        self.ocp.constraints.uh_e = np.array([1e10])

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

    def clf_decisionfunction(self, clf, X_iter, x):

        dual_coef = clf.dual_coef_
        sup_vec = clf.support_vectors_
        const = clf.intercept_
        output = 0
        for i in range(sup_vec.shape[0]):
            output += dual_coef[0, i] * exp(
                -(norm_2(x - sup_vec[i]) ** 2) / (2 * X_iter.var())
            )
        output += const

        return vertcat(output)


class OCPdoublependulumNN(OCPdoublependulum):
    def __init__(self, nn, mean, std):

        # inherit initialization
        super().__init__()

        # nonlinear terminal constraints
        self.model.con_h_expr_e = self.nn_decisionfunction(
            nn, mean, std, self.x)
        self.ocp.constraints.lh_e = np.array([0.5])
        self.ocp.constraints.uh_e = np.array([1.1])

        # ocp model
        self.ocp.model = self.model

        # options
        self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.tol = 1e-3
        self.ocp.solver_options.qp_tol = 1e-3
        self.ocp.solver_options.nlp_solver_max_iter = 1000
        self.ocp.solver_options.qp_solver_iter_max = 1000

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

    def nn_decisionfunction(self, nn, mean, std, x):

        out = (x - mean) / std
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
