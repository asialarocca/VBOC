from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, cos, sin, fmax, norm_2
import scipy.linalg as lin

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

        # xdot
        theta1_dot = SX.sym("theta1_dot")
        theta2_dot = SX.sym("theta1_dot")
        dtheta1_dot = SX.sym("dtheta2_dot")
        dtheta2_dot = SX.sym("dtheta2_dot")
        xdot = vertcat(theta1_dot, theta2_dot, dtheta1_dot, dtheta2_dot)

        # controls
        C1 = SX.sym("C1")
        C2 = SX.sym("C2")
        u = vertcat(C1, C2)

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

        self.model = AcadosModel()

        f_impl = xdot - f_expl

        self.model.f_expl_expr = f_expl
        self.model.f_impl_expr = f_impl
        self.model.x = self.x
        self.model.xdot = xdot
        self.model.u = u
        self.model.p = p
        self.model.name = model_name
        

class OCPdoublependulumINIT(OCPdoublependulum):
    def __init__(self, regenerate, nn_params, mean, std, safety_margin):

        # inherit initialization
        super().__init__()

        # ---------------------SET OCP---------------------
        # -------------------------------------------------
        self.ocp = AcadosOcp()

        # times
        self.Tf = 0.05
        self.N = int(100 * self.Tf)
        self.ocp.solver_options.tf = self.Tf
        self.ocp.dims.N = self.N

        self.nx = self.model.x.size()[0]
        self.nu = self.model.u.size()[0]
        self.ny = self.nx + self.nu
        self.ny_e = self.nx

        # cost
        Q = np.diag([1e2, 1e2, 1e-4, 1e-4])
        R = np.diag([1e-4, 1e-4])

        self.ocp.cost.W_e = Q
        self.ocp.cost.W = lin.block_diag(Q, R)

        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        self.ocp.cost.Vx = np.zeros((self.ny, self.nx))
        self.ocp.cost.Vx[: self.nx, :self.nx] = np.eye(self.nx)
        self.ocp.cost.Vu = np.zeros((self.ny, self.nu))
        self.ocp.cost.Vu[self.nx:, :self.nu] = np.eye(self.nu)
        self.ocp.cost.Vx_e = np.eye(self.nx)

        # reference
        self.ocp.cost.yref = np.array([np.pi, np.pi, 0., 0., 0., 0.])
        self.ocp.cost.yref_e = np.array([np.pi, np.pi, 0., 0.])

        # set constraints
        self.Cmax = 10.
        self.thetamax = np.pi / 4 + np.pi
        self.thetamin = - np.pi / 4 + np.pi
        self.dthetamax = 10.

        Cmax_limits = np.array([self.Cmax, self.Cmax])
        Xmax_limits = np.array([self.thetamax, self.thetamax, self.dthetamax, self.dthetamax])
        Xmin_limits = np.array([self.thetamin, self.thetamin, -self.dthetamax, -self.dthetamax])

        self.ocp.constraints.lbu = -Cmax_limits
        self.ocp.constraints.ubu = Cmax_limits
        self.ocp.constraints.idxbu = np.array([0, 1])
        self.ocp.constraints.lbx = Xmin_limits
        self.ocp.constraints.ubx = Xmax_limits
        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3])

        self.ocp.constraints.lbx_e = Xmin_limits
        self.ocp.constraints.ubx_e = Xmax_limits
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2, 3])

        self.ocp.constraints.lbx_0 = Xmin_limits
        self.ocp.constraints.ubx_0 = Xmax_limits
        self.ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])

        # nonlinear constraints
        self.model.con_h_expr_e = self.nn_decisionfunction(nn_params, mean, std, safety_margin, self.x)
        
        self.ocp.constraints.lh_e = np.array([0.])
        self.ocp.constraints.uh_e = np.array([1e6])

        # -------------------------------------------------

        self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        self.ocp.solver_options.tol = 1e-2
        self.ocp.solver_options.qp_solver_iter_max = 100
        self.ocp.solver_options.nlp_solver_max_iter = 1000
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        self.ocp.solver_options.alpha_reduction = 0.3
        self.ocp.solver_options.alpha_min = 1e-2
        self.ocp.solver_options.levenberg_marquardt = 1.

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, build=regenerate)

    def OCP_solve(self, x0, q_ref, x_sol_guess, u_sol_guess):

        # Reset current iterate:
        self.ocp_solver.reset()

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        # Set parameters, guesses and constraints:
        for i in range(self.N):
            self.ocp_solver.set(i, 'x', x_sol_guess[i])
            self.ocp_solver.set(i, 'u', u_sol_guess[i])
            self.ocp_solver.cost_set(i, 'y_ref', np.array([q_ref[0],q_ref[1],0.,0.,0.,0.]))

        self.ocp_solver.set(self.N, 'x', x_sol_guess[self.N])
        self.ocp_solver.cost_set(self.N, 'y_ref', np.array([q_ref[0],q_ref[1],0.,0.]))

        # Solve the OCP:
        status = self.ocp_solver.solve()

        return status
    
    def nn_decisionfunction(self, params, mean, std, safety_margin, x):

        vel_norm = fmax(norm_2(x[2:]), 1e-3)

        mean = vertcat(mean,mean,0.,0.)
        std = vertcat(std,std,vel_norm,vel_norm)

        out = (x - mean) / std
        it = 0

        for param in params:

            param = SX(param.tolist())

            if it % 2 == 0:
                out = param @ out
            else:
                out = param + out

                if it == 1 or it == 3:
                    out = fmax(0., out)

            it += 1

        return out - vel_norm 


class SYMdoublependulumINIT(OCPdoublependulum):
    def __init__(self, regenerate):

        # inherit initialization
        super().__init__()

        sim = AcadosSim()
        sim.model = self.model
        sim.solver_options.T = 1e-2
        sim.solver_options.num_stages = 4
        self.acados_integrator = AcadosSimSolver(sim, build=regenerate)
