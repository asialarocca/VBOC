import scipy.linalg as lin
from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, exp, norm_2, fmax, tanh
import time
from scipy.integrate import odeint
import torch
from sklearn import svm

class OCPpendulumR:
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
        Tf = 0.4
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
        Q = 2 * np.diag([0.0, -1.])
        R = 2 * np.diag([0.0])

        self.ocp.cost.W_e = np.diag([0.0, 0.]) 
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
        self.thetamax = np.pi / 2
        self.thetamin = 0.0
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
        self.ocp.constraints.lbx_0 = np.array([self.thetamin, -self.dthetamax])
        self.ocp.constraints.ubx_0 = np.array([self.thetamax, self.dthetamax])
        self.ocp.constraints.idxbx_0 = np.array([0, 1])

        # options
        self.ocp.solver_options.nlp_solver_type = "SQP"
        # -------------------------------------------------

    def compute_problem(self, qe, ve, yref):

        self.ocp_solver.reset()

        xe = np.array([qe, ve])

        self.ocp_solver.constraints_set(self.N, "lbx", xe)
        self.ocp_solver.constraints_set(self.N, "ubx", xe)

        # for i in range(self.N):
        #     self.ocp_solver.set(i, "x", yref[:2])
        #     self.ocp_solver.cost_set(i, "yref", yref)

        # self.ocp_solver.set(self.N, "x", xe)
        
        if qe > (self.thetamin + self.thetamax)/2:
        
            x_guess = np.array([self.thetamin, self.dthetamax])
            
            lb = np.array([self.thetamin, 0.])
            ub = np.array([self.thetamax, self.dthetamax])
            
            for i in range(self.N):
                self.ocp_solver.set(i, "x", x_guess)
                self.ocp_solver.constraints_set(i, "lbx", lb)
                self.ocp_solver.constraints_set(i, "ubx", ub)
                
            self.ocp_solver.set(self.N, "x", x_guess)
            
        else:
        
            x_guess = np.array([self.thetamax, -self.dthetamax])
            
            lb = np.array([self.thetamin, -self.dthetamax])
            ub = np.array([self.thetamax, 0.])
            
            for i in range(self.N):
                self.ocp_solver.set(i, "x", x_guess)
                self.ocp_solver.constraints_set(i, "lbx", lb)
                self.ocp_solver.constraints_set(i, "ubx", ub)
                
            self.ocp_solver.set(self.N, "x", x_guess)

        status = self.ocp_solver.solve()

        if status == 0:
            return 1
        else:
            return 0

class OCPpendulumRINIT(OCPpendulumR):
    def __init__(self):

        # inherit initialization
        super().__init__()

        # ocp model
        self.ocp.model = self.model
        
        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")


