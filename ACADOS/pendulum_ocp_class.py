from utils import plot_pendulum
import scipy.linalg as lin
from pendulum_model import export_pendulum_ode_model
from acados_template import AcadosOcp, AcadosOcpSolver
from numpy.linalg import norm as norm
import numpy as np
from numpy import nan
import time
import sys
sys.path.insert(0, '../common')


class OCPpendulum:

    def __init__(self):
        self.Tf = 1.0
        self.N = 20
        self.Fmax = 10
        self.thetamax = np.pi/2
        self.thetamin = 0.0
        self.dthetamax = 10.
        self.Q = 2*np.diag([0.0, 1e-2])
        self.R = 2*np.diag([0.0])
        self.simX = np.ndarray((self.N+1, 2))
        self.simU = np.ndarray((self.N, 1))

    def compute_problem(self, q0, v0):
        # create ocp object to formulate the OCP
        ocp = AcadosOcp()

        # set model
        model = export_pendulum_ode_model()
        ocp.model = model

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        ny_e = nx

        # set dimensions
        ocp.dims.N = self.N

        # set cost
        ocp.cost.W_e = self.Q
        ocp.cost.W = lin.block_diag(self.Q, self.R)

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        Vu = np.zeros((ny, nu))
        Vu[2, 0] = 1.0
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        # set constraints
        ocp.constraints.lbu = np.array([-self.Fmax])
        ocp.constraints.ubu = np.array([+self.Fmax])
        ocp.constraints.idxbu = np.array([0])

        ocp.constraints.lbx = np.array([self.thetamin, -self.dthetamax])
        ocp.constraints.ubx = np.array([self.thetamax, self.dthetamax])
        ocp.constraints.idxbx = np.array([0, 1])
        ocp.constraints.lbx_e = np.array([self.thetamin, -self.dthetamax])
        ocp.constraints.ubx_e = np.array([self.thetamax, self.dthetamax])
        ocp.constraints.idxbx_e = np.array([0, 1])

        # set options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP'

        # set prediction horizon
        ocp.solver_options.tf = self.Tf

        # Initial cnditions
        ocp.constraints.x0 = np.array([q0, v0])

        # Solver
        ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

        status = ocp_solver.solve()

        if status != 0:
            return 0

        # get solution
        for i in range(self.N):
            self.simX[i, :] = ocp_solver.get(i, "x")
            self.simU[i, :] = ocp_solver.get(i, "u")
        self.simX[self.N, :] = ocp_solver.get(self.N, "x")

        if norm(self.simX[self.N, 1]) < 0.01:
            return 1
        else:
            return 0
