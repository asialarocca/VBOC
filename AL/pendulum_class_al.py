import scipy.linalg as lin
from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, exp, norm_2, fmax, tanh
from scipy.integrate import odeint
import torch

class OCPpendulum:
    def __init__(self):

        # --------------------SET MODEL--------------------
        # -------------------------------------------------
        model_name = "pendulum_ode"

        # constants
        self.m = 0.5  # mass of the ball [kg]
        self.g = 9.81  # gravity constant [m/s^2]
        self.d = 0.3  # length of the rod [m]
        self.b = 0.01  # damping

        # states
        theta = SX.sym("theta")
        dtheta = SX.sym("dtheta")
        self.x = vertcat(theta, dtheta)

        # controls
        F = SX.sym("F")
        u = vertcat(F)

        # xdot
        theta_dot = SX.sym("theta_dot")
        dtheta_dot = SX.sym("dtheta_dot")
        xdot = vertcat(theta_dot, dtheta_dot)

        # parameters
        p = []

        # dynamics
        f_expl = vertcat(
            dtheta,
            (self.m * self.g * self.d * sin(theta) + F - self.b * dtheta)
            / (self.d * self.d * self.m),
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

        # times
        Tf = 1.
        self.Tf = Tf
        self.N = int(100 * Tf)

        # prediction horizon
        self.ocp.solver_options.tf = self.Tf

        # dimensions
        self.nx = self.model.x.size()[0]
        nu = self.model.u.size()[0]
        ny = self.nx + nu
        ny_e = self.nx

        self.ocp.dims.N = self.N

        # cost
        Q = 2 * np.diag([0.0, 1.])
        R = 2 * np.diag([0.0])

        self.ocp.cost.W_e = Q
        self.ocp.cost.W = lin.block_diag(Q, R)

        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        self.ocp.cost.Vx = np.zeros((ny, self.nx))
        self.ocp.cost.Vx[: self.nx, : self.nx] = np.eye(self.nx)
        Vu = np.zeros((ny, nu))
        Vu[2, 0] = 1.0
        self.ocp.cost.Vu = Vu
        self.ocp.cost.Vx_e = np.eye(self.nx)

        # reference
        self.ocp.cost.yref = np.zeros((ny,))
        self.ocp.cost.yref_e = np.zeros((ny_e,))

        # constraints
        self.Fmax = 3
        self.thetamax = np.pi / 4 + np.pi
        self.thetamin = - np.pi / 4 + np.pi
        self.dthetamax = 10.0

        self.ocp.constraints.lbu = np.array([-self.Fmax])
        self.ocp.constraints.ubu = np.array([+self.Fmax])
        self.ocp.constraints.idxbu = np.array([0])
        self.ocp.constraints.lbx = np.array([self.thetamin, -self.dthetamax])
        self.ocp.constraints.ubx = np.array([self.thetamax, self.dthetamax])
        self.ocp.constraints.idxbx = np.array([0, 1])
        self.ocp.constraints.lbx_e = np.array([self.thetamin, -self.dthetamax])
        self.ocp.constraints.ubx_e = np.array([self.thetamax, self.dthetamax])
        self.ocp.constraints.idxbx_e = np.array([0, 1])

        self.ocp.constraints.idxbx_0 = np.array([0, 1])
        self.ocp.constraints.idxbxe_0 = np.array([0, 1])
        self.ocp.constraints.lbx_0 = np.array([0.0, 0.0])
        self.ocp.constraints.ubx_0 = np.array([0.0, 0.0])
        self.ocp.constraints.lbu_0 = np.array([-self.Fmax])
        self.ocp.constraints.ubu_0 = np.array([+self.Fmax])
        self.ocp.constraints.idxbu_0 = np.array([0])

        # # options
        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        # -------------------------------------------------

    def set_bounds(self, q_min, q_max):
        self.thetamax = q_max
        self.thetamin = q_min

        self.ocp.constraints.lbx[0] = q_min
        self.ocp.constraints.ubx[0] = q_max
        self.ocp.constraints.lbx_e[0] = q_min
        self.ocp.constraints.ubx_e[0] = q_max

        for i in range(self.N):
            self.ocp_solver.constraints_set(i, "lbx", self.ocp.constraints.lbx)
            self.ocp_solver.constraints_set(i, "ubx", self.ocp.constraints.ubx)

        self.ocp_solver.constraints_set(self.N, "lbx", self.ocp.constraints.lbx_e)
        self.ocp_solver.constraints_set(self.N, "ubx", self.ocp.constraints.ubx_e)

    def compute_problem(self, q0, v0):

        self.ocp_solver.reset()

        x0 = np.array([q0, v0])

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        x_guess = np.array([q0, 0.0])

        for i in range(self.N+1):
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

        x0 = np.array([q0, v0])

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        with torch.no_grad():
            inp = torch.Tensor([[[q0, v0]]])
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

        x0 = np.array([q0, v0])

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        normal = np.array([self.thetamax - self.thetamin, 2 * self.dthetamax])
        dist_min = 1e3
        index = 0

        for k in range(simX_vec.shape[0]):
            if np.sign(simX_vec[k, 0, 1]) == np.sign(x0[1]):
                dist = np.linalg.norm(x0 / normal - simX_vec[k, 0, :] / normal)
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

        x0 = np.array([q0, v0])

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        times = np.linspace(0, self.Tf, self.N + 1)
        simX = odeint(self.model_pid, [q0, v0], times)

        simX[:, 0] = [
            np.sign(simX[i, 0]) * self.thetamax
            if abs(simX[i, 0]) > self.thetamax
            else simX[i, 0]
            for i in range(self.N + 1)
        ]
        simX[:, 1] = [
            np.sign(simX[i, 1]) * self.dthetamax
            if abs(simX[i, 1]) > self.dthetamax
            else simX[i, 1]
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
        y = x[0]
        dydt = x[1]
        u = -10 * dydt

        if u >= self.Fmax:
            u = self.Fmax
        elif u <= -self.Fmax:
            u = -self.Fmax

        dy2dt2 = (self.m * self.g * self.d * sin(y) + u - self.b * dydt) / (
            self.d * self.d * self.m
        )
        return [dydt, dy2dt2]


class OCPpendulumINIT(OCPpendulum):
    def __init__(self):

        # inherit initialization
        super().__init__()

        # linear terminal constraints (zero final velocity)
        self.ocp.constraints.lbx_e[1] = 0.0
        self.ocp.constraints.ubx_e[1] = 0.0

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")


class OCPpendulumSVM(OCPpendulum):
    def __init__(self, clf, X_iter):

        # inherit initialization
        super().__init__()

        # nonlinear terminal constraints (svm)
        self.model.con_h_expr_e = self.clf_decisionfunction(clf, X_iter, self.x)
        self.ocp.constraints.lh_e = np.array([0.0])
        self.ocp.constraints.uh_e = np.array([1e10])  # prova con inf o nan

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


class OCPpendulumNN(OCPpendulum):
    def __init__(self, nn, mean, std):

        # inherit initialization
        super().__init__()

        # nonlinear terminal constraints
        self.model.con_h_expr_e = self.nn_decisionfunction(
            nn, mean, std, self.x)
        self.ocp.constraints.lh_e = np.array([-0.0])
        self.ocp.constraints.uh_e = np.array([1.1])

        self.ocp.constraints.lbx_e = np.array([self.thetamin, -self.dthetamax])
        self.ocp.constraints.ubx_e = np.array([self.thetamax, self.dthetamax])

        # ocp model
        self.ocp.model = self.model

        # options
        self.ocp.solver_options.nlp_solver_type = "SQP"

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
                    return vertcat(out[1]-out[0])

            it += 1
