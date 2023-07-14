from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, cos, sin, sqrt

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
        dt = SX.sym('dt')
        self.x = vertcat(theta1, theta2, dtheta1, dtheta2, dt)

        # controls
        C1 = SX.sym("C1")
        C2 = SX.sym("C2")
        u = vertcat(C1, C2)

        # parameters
        w1 = SX.sym("w1") 
        w2 = SX.sym("w2") 
        wt = SX.sym("wt") 
        p = vertcat(w1, w2, wt)

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

        # times
        self.N = 100
        self.ocp.solver_options.tf = self.N
        self.ocp.dims.N = self.N

        # ocp model
        self.ocp.model = self.model

        # set constraints
        self.Cmax = 10.
        self.thetamax =  np.pi / 4 + np.pi # 2*np.pi
        self.thetamin = - np.pi / 4 + np.pi # np.pi
        self.dthetamax = 10.

        # cost
        self.ocp.cost.cost_type_0 = 'EXTERNAL'
        self.ocp.cost.cost_type = 'EXTERNAL'

        self.ocp.model.cost_expr_ext_cost_0 = w1 * dtheta1 + w2 * dtheta2 + wt * dt 
        self.ocp.model.cost_expr_ext_cost = wt * dt 
        self.ocp.parameter_values = np.array([0., 0., 0.])

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
        ) 
        self.ocp.constraints.ubx_e = np.array(
            [self.thetamax, self.thetamax, self.dthetamax, self.dthetamax, 1e-2]
        )
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2, 3, 4])
        self.ocp.constraints.lbx_0 = np.array([self.thetamin, self.thetamin, -self.dthetamax, -self.dthetamax, 0.]) 
        self.ocp.constraints.ubx_0 = np.array([self.thetamax, self.thetamax, self.dthetamax, self.dthetamax, 1e-2]) 
        self.ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4])

        self.ocp.constraints.C = np.zeros((2,5))
        self.ocp.constraints.D = np.zeros((2,2))
        self.ocp.constraints.lg = np.zeros((2))
        self.ocp.constraints.ug = np.zeros((2))

        self.model.con_h_expr = (self.l1*sin(theta1) + self.l2*sin(theta2))**2 + (self.l1*cos(theta1) + self.l2*cos(theta2) + self.l1 + self.l2/2)**2
        self.ocp.constraints.lh = np.array([(self.l2/2)**2])
        self.ocp.constraints.uh = np.array([1e6])

        # -------------------------------------------------

        self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.hessian_approx = 'EXACT'
        self.ocp.solver_options.exact_hess_constr = 0
        # self.ocp.solver_options.exact_hess_cost = 0
        self.ocp.solver_options.exact_hess_dyn = 0
        self.ocp.solver_options.nlp_solver_tol_stat = 1e-4
        self.ocp.solver_options.qp_solver_iter_max = 100
        self.ocp.solver_options.nlp_solver_max_iter = 1000
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        self.ocp.solver_options.alpha_reduction = 0.3
        self.ocp.solver_options.alpha_min = 1e-2
        self.ocp.solver_options.levenberg_marquardt = 1e-5

class OCPdoublependulumINIT(OCPdoublependulum):
    def __init__(self):

        # inherit initialization
        super().__init__()

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

    def OCP_solve(self, x_sol_guess, u_sol_guess, p, q_lb, q_ub, u_lb, u_ub, q_init_lb, q_init_ub, q_fin_lb, q_fin_ub):

        # Reset current iterate:
        self.ocp_solver.reset()

        # Set parameters, guesses and constraints:
        for i in range(self.N):
            self.ocp_solver.set(i, 'x', x_sol_guess[i])
            self.ocp_solver.set(i, 'u', u_sol_guess[i])
            self.ocp_solver.set(i, 'p', p)
            self.ocp_solver.constraints_set(i, 'lbx', q_lb) 
            self.ocp_solver.constraints_set(i, 'ubx', q_ub) 
            self.ocp_solver.constraints_set(i, 'lbu', u_lb)
            self.ocp_solver.constraints_set(i, 'ubu', u_ub)
            self.ocp_solver.constraints_set(i, 'C', np.zeros((2,5)))
            self.ocp_solver.constraints_set(i, 'D', np.zeros((2,2)))
            self.ocp_solver.constraints_set(i, 'lg', np.zeros((2)))
            self.ocp_solver.constraints_set(i, 'ug', np.zeros((2)))

        C = np.zeros((2,5))
        d = np.array([p[:2].tolist()])
        dt = np.transpose(d)
        C[:,2:4] = np.identity(2)-np.matmul(dt,d)
        self.ocp_solver.constraints_set(0, "C", C, api='new') 

        self.ocp_solver.constraints_set(0, "lbx", q_init_lb)
        self.ocp_solver.constraints_set(0, "ubx", q_init_ub)

        self.ocp_solver.constraints_set(self.N, "lbx", q_fin_lb)
        self.ocp_solver.constraints_set(self.N, "ubx", q_fin_ub)
        self.ocp_solver.set(self.N, 'x', x_sol_guess[-1])
        self.ocp_solver.set(self.N, 'p', p)

        # Solve the OCP:
        status = self.ocp_solver.solve()

        return status


class SYMdoublependulumINIT(OCPdoublependulum):
    def __init__(self):

        # inherit initialization
        super().__init__()

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

        self.model.f_expl_expr = f_expl
        self.model.x = self.x
        self.model.p = p
        self.model.u = u

        sim = AcadosSim()
        sim.model = self.model
        sim.solver_options.T = 1e-2
        sim.solver_options.num_stages = 4
        self.acados_integrator = AcadosSimSolver(sim)
