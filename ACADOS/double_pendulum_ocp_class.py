import scipy.linalg as lin
from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, cos, sin, exp, norm_2
import matplotlib.pyplot as plt


class OCPdoublependulum:
    def __init__(self):

        # --------------------SET MODEL--------------------
        # -------------------------------------------------
        model_name = "double_pendulum_ode"

        # constants
        m1 = 0.4  # mass of the first link [kg]
        m2 = 0.4  # mass of the second link [kg]
        g = 9.81  # gravity constant [m/s^2]
        l1 = 0.8  # length of the first link [m]
        l2 = 0.8  # length of the second link [m]

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
                l1**2 * l2 * m2 * dtheta1**2 * sin(-2 * theta2 + 2 * theta1)
                + 2 * C2 * cos(-theta2 + theta1) * l1
                + 2
                * (
                    g * sin(-2 * theta2 + theta1) * l1 * m2 / 2
                    + sin(-theta2 + theta1) * dtheta2**2 * l1 * l2 * m2
                    + g * l1 * (m1 + m2 / 2) * sin(theta1)
                    - C1
                )
                * l2
            )
            / l1**2
            / l2
            / (m2 * cos(-2 * theta2 + 2 * theta1) - 2 * m1 - m2),
            (
                -g * l1 * l2 * m2 * (m1 + m2) * sin(-theta2 + 2 * theta1)
                - l1 * l2**2 * m2**2 * dtheta2**2 * sin(-2 * theta2 + 2 * theta1)
                - 2
                * dtheta1**2
                * l1**2
                * l2
                * m2
                * (m1 + m2)
                * sin(-theta2 + theta1)
                + 2 * C1 * cos(-theta2 + theta1) * l2 * m2
                + l1 * (m1 + m2) * (sin(theta2) * g * l2 * m2 - 2 * C2)
            )
            / l2**2
            / l1
            / m2
            / (m2 * cos(-2 * theta2 + 2 * theta1) - 2 * m1 - m2),
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
        self.Tf = 0.05
        self.ocp.solver_options.tf = self.Tf  # prediction horizon

        self.N = int(100 * self.Tf)
        self.ocp.dims.N = self.N

        self.nx = self.model.x.size()[0]
        nu = self.model.u.size()[0]
        ny = self.nx + nu
        ny_e = self.nx

        # cost
        Q = 2 * np.diag([0.0, 0.0, 1e-2, 1e-2])
        R = 2 * np.diag([0.0, 0.0])

        self.ocp.cost.W_e = Q
        self.ocp.cost.W = lin.block_diag(Q, R)

        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        self.ocp.cost.Vx = np.zeros((ny, self.nx))
        self.ocp.cost.Vx[: self.nx, : self.nx] = np.eye(self.nx)
        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.cost.Vu[self.nx :, :nu] = np.eye(nu)
        self.ocp.cost.Vx_e = np.eye(self.nx)

        # reference
        self.ocp.cost.yref = np.zeros((ny,))
        self.ocp.cost.yref_e = np.zeros((ny_e,))

        # set constraints
        self.Cmax = 10
        self.thetamax = np.pi / 2 + np.pi
        self.thetamin = np.pi
        self.dthetamax = 5.0

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

        self.ocp.constraints.x0 = np.array([self.thetamin, self.thetamin, 0.0, 0.0])

        # options
        self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp.solver_options.nlp_solver_max_iter = 1000
        self.ocp.solver_options.qp_solver_iter_max = 1000
        self.ocp.solver_options.tol = 1e-3
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        # -------------------------------------------------

    def compute_problem(self, q0, v0):

        self.ocp_solver.reset()

        x0 = np.array([q0[0], q0[1], v0[0], v0[1]])

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        x_guess = np.array([q0[0], q0[1], 0, 0])
        u_guess = np.array([-v0[0], -v0[1]])

        for i in range(self.N + 1):
            self.ocp_solver.set(i, "x", x_guess)

        for i in range(self.N):
            self.ocp_solver.set(i, "u", u_guess)

        status = self.ocp_solver.solve()

        if status == 2:
            return 2
        elif status != 0:
            return 0
        else:
            # # get solution
            # simX = np.ndarray((self.N+1, self.nx))
            # simU = np.ndarray((self.N, 2))

            # for i in range(self.N):
            #     simX[i, :] = self.ocp_solver.get(i, "x")
            #     simU[i, :] = self.ocp_solver.get(i, "u")
            # simX[self.N, :] = self.ocp_solver.get(self.N, "x")

            # t = np.linspace(0, self.Tf, self.N+1)

            # plt.figure()
            # plt.subplot(2, 1, 1)
            # line, = plt.step(t, np.append([simU[0, 0]], simU[:, 0]))
            # plt.ylabel('$C1$')
            # plt.xlabel('$t$')
            # plt.hlines(self.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.hlines(-self.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.ylim([-1.2*self.Cmax, 1.2*self.Cmax])
            # plt.title('Controls')
            # plt.grid()
            # plt.subplot(2, 1, 2)
            # line, = plt.step(t, np.append([simU[0, 1]], simU[:, 1]))
            # plt.ylabel('$C2$')
            # plt.xlabel('$t$')
            # plt.hlines(self.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.hlines(-self.Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
            # plt.ylim([-1.2*self.Cmax, 1.2*self.Cmax])
            # plt.grid()

            # plt.figure()
            # plt.subplot(4, 1, 1)
            # line, = plt.plot(t, simX[:, 0])
            # plt.ylabel('$theta1$')
            # plt.xlabel('$t$')
            # plt.title('States')
            # plt.grid()
            # plt.subplot(4, 1, 2)
            # line, = plt.plot(t, simX[:, 1])
            # plt.ylabel('$theta2$')
            # plt.xlabel('$t$')
            # plt.grid()
            # plt.subplot(4, 1, 3)
            # line, = plt.plot(t, simX[:, 2])
            # plt.ylabel('$dtheta1$')
            # plt.xlabel('$t$')
            # plt.grid()
            # plt.subplot(4, 1, 4)
            # line, = plt.plot(t, simX[:, 3])
            # plt.ylabel('$dtheta2$')
            # plt.xlabel('$t$')
            # plt.grid()
            # plt.show()

            return 1


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
