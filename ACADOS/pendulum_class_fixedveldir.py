import scipy.linalg as lin
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, exp, norm_2, fmax, tanh
import time
from scipy.integrate import odeint
import torch
from sklearn import svm

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
        dt = SX.sym('dt')
        self.x = vertcat(theta, dtheta, dt)

        # controls
        F = SX.sym("F")
        u = vertcat(F)

        # parameters
        w1 = SX.sym("w1") 
        wt = SX.sym("wt") 
        p = vertcat(w1, wt)

        # dynamics
        f_expl = dt*vertcat(
            dtheta,
            (self.m * self.g * self.d * sin(theta) + F - self.b * dtheta)
            / (self.d * self.d * self.m), 0.,
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
        Tf = 100
        self.N = Tf
        self.ocp.solver_options.tf = Tf
        self.ocp.dims.N = self.N

        # ocp model
        self.ocp.model = self.model

        # constraints
        self.Fmax = 3
        self.thetamax = np.pi / 4 + np.pi
        self.thetamin = - np.pi / 4 + np.pi
        self.dthetamax = 10.0

        # cost
        self.ocp.cost.cost_type_0 = 'EXTERNAL'
        self.ocp.cost.cost_type = 'EXTERNAL'

        self.ocp.model.cost_expr_ext_cost_0 = w1 * dtheta + wt * dt 
        self.ocp.model.cost_expr_ext_cost = wt * dt 
        self.ocp.parameter_values = np.array([0., 1.])

        self.ocp.constraints.lbu = np.array([-self.Fmax])
        self.ocp.constraints.ubu = np.array([+self.Fmax])
        self.ocp.constraints.idxbu = np.array([0])
        self.ocp.constraints.lbx = np.array([self.thetamin, -self.dthetamax, 0.])
        self.ocp.constraints.ubx = np.array([self.thetamax, self.dthetamax, 1e-2])
        self.ocp.constraints.idxbx = np.array([0, 1, 2])
        self.ocp.constraints.lbx_e = np.array([self.thetamin, -self.dthetamax, 0.]) 
        self.ocp.constraints.ubx_e = np.array([self.thetamax, self.dthetamax, 1e-2])
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2])
        self.ocp.constraints.lbx_0 = np.array([self.thetamin, -self.dthetamax, 0.])
        self.ocp.constraints.ubx_0 = np.array([self.thetamax, self.dthetamax, 1e-2]) 
        self.ocp.constraints.idxbx_0 = np.array([0, 1, 2])

        # options
        self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.hessian_approx = 'EXACT'
        self.ocp.solver_options.exact_hess_constr = 0
        # self.ocp.solver_options.exact_hess_cost = 0
        self.ocp.solver_options.exact_hess_dyn = 0
        self.ocp.solver_options.nlp_solver_tol_stat = 1e-4
        # self.ocp.solver_options.qp_solver_iter_max = 100
        self.ocp.solver_options.nlp_solver_max_iter = 1000
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        self.ocp.solver_options.alpha_reduction = 0.3
        self.ocp.solver_options.alpha_min = 1e-2
        self.ocp.solver_options.levenberg_marquardt = 1e-5
        # -------------------------------------------------

    def compute_problem(self, xe, x_guess):

        self.ocp_solver.reset()

        self.ocp_solver.constraints_set(self.N, "lbx", np.array([xe[0], xe[1], 0.]))
        self.ocp_solver.constraints_set(self.N, "ubx", np.array([xe[0], xe[1], 1e-2]))

        self.ocp_solver.constraints_set(0, "lbx", np.array([x_guess[0,0], -self.dthetamax, 0.]))
        self.ocp_solver.constraints_set(0, "ubx", np.array([x_guess[0,0], self.dthetamax, 1e-2]))

        for i in range(self.N):
            self.ocp_solver.set(i, "x", np.array([x_guess[i,0],x_guess[i,1], 1e-2]))

        self.ocp_solver.set(self.N, "x", np.array([xe[0], xe[1], 1e-2]))
        
        if xe[0] > (self.thetamin + self.thetamax)/2:
            
            lb = np.array([self.thetamin, 0.,0.])
            ub = np.array([self.thetamax, self.dthetamax,1e-2])
            
            for i in range(1,self.N):
                self.ocp_solver.constraints_set(i, "lbx", lb)
                self.ocp_solver.constraints_set(i, "ubx", ub)
                self.ocp_solver.set(i, 'p', np.array([-1., 1.]))
            
        else:
            
            lb = np.array([self.thetamin, -self.dthetamax,0.])
            ub = np.array([self.thetamax, 0.,1e-2])
            
            for i in range(1,self.N):
                self.ocp_solver.constraints_set(i, "lbx", lb)
                self.ocp_solver.constraints_set(i, "ubx", ub)
                self.ocp_solver.set(i, 'p', np.array([1., 1.]))

        status = self.ocp_solver.solve()

        if status == 0:
            return 1
        else:
            return 0

class OCPpendulumINIT(OCPpendulum):
    def __init__(self):

        # inherit initialization
        super().__init__()
        
        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")
