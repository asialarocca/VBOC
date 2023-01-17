import scipy.linalg as lin
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, exp, norm_2, fmax, tanh
import time
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
        dt = SX.sym('dt')
        self.x = vertcat(theta, dtheta, dt)

        # controls
        F = SX.sym("F")
        u = vertcat(F)

        # parameters
        w1 = SX.sym("w1") 
        w2 = SX.sym("w2") 
        wt = SX.sym("wt") 
        p = vertcat(w1, w2, wt)

        # dynamics
        f_expl = dt*vertcat(
            dtheta,
            (self.m * self.g * self.d * sin(theta) + F - self.b * dtheta)
            / (self.d * self.d * self.m), 0.
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

        # ocp model
        self.ocp.model = self.model

        # constraints
        self.Fmax = 3
        self.thetamax = np.pi / 2
        self.thetamin = 0.0
        self.dthetamax = 10.0

        # times
        Tf = 1.
        self.N = int(100 * Tf)
        self.ocp.solver_options.tf = Tf
        self.ocp.dims.N = self.N

        # set cost
        self.ocp.cost.cost_type_0 = 'EXTERNAL'
        self.ocp.cost.cost_type = 'EXTERNAL'

        self.ocp.model.cost_expr_ext_cost_0 = w1 / (self.thetamax - self.thetamin) * theta + w2 / self.dthetamax * dtheta
        self.ocp.model.cost_expr_ext_cost = wt * dt # + 1e-6*(wt / (self.thetamax - self.thetamin) * theta + wt / self.dthetamax * dtheta)

        self.ocp.parameter_values = np.array([0., 0., 0.])

        self.ocp.constraints.lbu = np.array([-self.Fmax])
        self.ocp.constraints.ubu = np.array([+self.Fmax])
        self.ocp.constraints.idxbu = np.array([0])
        self.ocp.constraints.lbx = np.array([self.thetamin, -self.dthetamax, 1.])
        self.ocp.constraints.ubx = np.array([self.thetamax, self.dthetamax,  1.])
        self.ocp.constraints.idxbx = np.array([0, 1, 2])
        self.ocp.constraints.lbx_e = np.array([self.thetamin, -self.dthetamax,  1.])
        self.ocp.constraints.ubx_e = np.array([self.thetamax, self.dthetamax,  1.])
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2])
        self.ocp.constraints.lbx_0 = np.array([self.thetamin, -self.dthetamax,  1.])
        self.ocp.constraints.ubx_0 = np.array([self.thetamax, self.dthetamax,  1.])
        self.ocp.constraints.idxbx_0 = np.array([0, 1, 2])

        # options
        self.ocp.solver_options.nlp_solver_type = 'SQP'
        self.ocp.solver_options.hessian_approx = 'EXACT'
        self.ocp.solver_options.exact_hess_constr = 0
        self.ocp.solver_options.exact_hess_cost = 0
        self.ocp.solver_options.exact_hess_dyn = 0
        self.ocp.solver_options.tol = 1e-6
        self.ocp.solver_options.qp_solver_iter_max = 100
        self.ocp.solver_options.nlp_solver_max_iter = 1000
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        self.ocp.solver_options.alpha_reduction = 0.3
        self.ocp.solver_options.alpha_min = 1e-2
        self.ocp.solver_options.levenberg_marquardt = 1e-6

        # -------------------------------------------------

class OCPpendulumINIT(OCPpendulum):
    def __init__(self):

        # inherit initialization
        super().__init__()

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

class SYMpendulumINIT(OCPpendulum):
    def __init__(self):

        # inherit initialization
        super().__init__()

        theta = SX.sym("theta")
        dtheta = SX.sym("dtheta")
        self.x = vertcat(theta, dtheta)

        # controls
        F = SX.sym("F")
        u = vertcat(F)
        
        p = []

        # dynamics
        f_expl = vertcat(
            dtheta,
            (self.m * self.g * self.d * sin(theta) + F - self.b * dtheta)
            / (self.d * self.d * self.m),
        )

        self.model.f_expl_expr = f_expl
        self.model.x = self.x
        self.model.p = p
        self.model.u = u

        sim = AcadosSim()
        sim.model = self.model
        sim.solver_options.T = 1e-2
        sim.solver_options.integrator_type = 'ERK'
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = 3
        self.acados_integrator = AcadosSimSolver(sim)
