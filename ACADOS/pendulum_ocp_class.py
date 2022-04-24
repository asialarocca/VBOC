import scipy.linalg as lin
from pendulum_model import export_pendulum_ode_model
from acados_template import AcadosOcp, AcadosOcpSolver
from numpy.linalg import norm as norm
import numpy as np
import sys
sys.path.insert(0, '../common')


class OCPpendulum:

    def __init__(self):
        # create ocp object to formulate the OCP
        self.ocp = AcadosOcp()

        # set model
        model = export_pendulum_ode_model()
        self.ocp.model = model

        self.nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = self.nx + nu
        ny_e = self.nx

        Tf = .1
        self.Tf = Tf
        self.N = 10  # int(100*Tf)

        # set prediction horizon
        self.ocp.solver_options.tf = self.Tf

        # set dimensions
        self.ocp.dims.N = self.N

        # set cost
        Q = 2*np.diag([0.0, 1e-2])
        R = 2*np.diag([0.0])

        self.ocp.cost.W_e = Q
        self.ocp.cost.W = lin.block_diag(Q, R)

        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        self.ocp.cost.Vx = np.zeros((ny, self.nx))
        self.ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)
        Vu = np.zeros((ny, nu))
        Vu[2, 0] = 1.0
        self.ocp.cost.Vu = Vu
        self.ocp.cost.Vx_e = np.eye(self.nx)

        self.ocp.cost.yref = np.zeros((ny, ))
        self.ocp.cost.yref_e = np.zeros((ny_e, ))

        # set constraints
        self.Fmax = 10
        self.thetamax = np.pi/2
        self.thetamin = 0.0
        self.dthetamax = 10.

        self.ocp.constraints.lbu = np.array([-self.Fmax])
        self.ocp.constraints.ubu = np.array([+self.Fmax])
        self.ocp.constraints.idxbu = np.array([0])
        self.ocp.constraints.lbx = np.array([self.thetamin, -self.dthetamax])
        self.ocp.constraints.ubx = np.array([self.thetamax, self.dthetamax])
        self.ocp.constraints.idxbx = np.array([0, 1])
        self.ocp.constraints.lbx_e = np.array([self.thetamin, 0.])
        self.ocp.constraints.ubx_e = np.array([self.thetamax, 0.])
        self.ocp.constraints.idxbx_e = np.array([0, 1])

        self.ocp.constraints.idxbx_0 = np.array([0, 1])
        self.ocp.constraints.idxbxe_0 = np.array([0, 1])
        self.ocp.constraints.lbx_0 = np.array([0., 0.])
        self.ocp.constraints.ubx_0 = np.array([0., 0.])

        # set options
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.nlp_solver_type = 'SQP'

        # Solver
        self.ocp_solver = AcadosOcpSolver(
            self.ocp, json_file='acados_ocp.json')

    def compute_problem(self, q0, v0):

        self.ocp_solver.reset()

        x0 = np.array([q0, v0])

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        status = self.ocp_solver.solve()

        if status != 0:
            return 0
        else:
            return 1
