import scipy.linalg as lin
from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, exp, norm_2


class OCPpendulum:

    def __init__(self):

        # --------------------SET MODEL--------------------
        # -------------------------------------------------
        model_name = 'pendulum_ode'

        # constants
        m = 0.5  # mass of the ball [kg]
        g = 9.81  # gravity constant [m/s^2]
        d = 0.3  # length of the rod [m]
        b = 0.01  # damping

        # states
        theta = SX.sym('theta')
        dtheta = SX.sym('dtheta')
        self.x = vertcat(theta, dtheta)

        # controls
        F = SX.sym('F')
        u = vertcat(F)

        # xdot
        theta_dot = SX.sym('theta_dot')
        dtheta_dot = SX.sym('dtheta_dot')
        xdot = vertcat(theta_dot, dtheta_dot)

        # parameters
        p = []

        # dynamics
        f_expl = vertcat(dtheta, (m*g*d*sin(theta)+F-b*dtheta)/(d*d*m))
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
        self.N = int(100*Tf)

        # prediction horizon
        self.ocp.solver_options.tf = self.Tf

        # dimensions
        self.nx = self.model.x.size()[0]
        nu = self.model.u.size()[0]
        ny = self.nx + nu
        ny_e = self.nx

        self.ocp.dims.N = self.N

        # cost
        Q = 2*np.diag([0.0, 1e-1])
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

        # reference
        self.ocp.cost.yref = np.zeros((ny, ))
        self.ocp.cost.yref_e = np.zeros((ny_e, ))

        # constraints
        self.Fmax = 3
        self.thetamax = np.pi/2
        self.thetamin = 0.0
        self.dthetamax = 10.

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
        self.ocp.constraints.lbx_0 = np.array([0., 0.])
        self.ocp.constraints.ubx_0 = np.array([0., 0.])

        # options
        self.ocp.solver_options.nlp_solver_type = 'SQP'
        # -------------------------------------------------

    def compute_problem(self, q0, v0):

        self.ocp_solver.reset()

        x0 = np.array([q0, v0])

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        for i in range(self.N+1):
            self.ocp_solver.set(i, 'x', x0)

        status = self.ocp_solver.solve()

        # if status == 0:
        #     return 1
        # elif status == 4:
        #     return 0
        # else:
        #     return 2

        if status != 0:
            return 0
        else:
            return 1


class OCPpendulumINIT(OCPpendulum):

    def __init__(self):

        # inherit initialization
        super().__init__()

        # linear terminal constraints (zero final velocity)
        self.ocp.constraints.lbx_e[1] = 0.
        self.ocp.constraints.ubx_e[1] = 0.

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(
            self.ocp, json_file='acados_ocp.json')


class OCPpendulumSVM(OCPpendulum):

    def __init__(self, clf, X_iter):

        # inherit initialization
        super().__init__()

        # nonlinear terminal constraints (svm)
        self.model.con_h_expr_e = self.clf_decisionfunction(clf, X_iter, self.x)
        self.ocp.constraints.lh_e = np.array([0.])
        self.ocp.constraints.uh_e = np.array([1e10])  # prova con inf o nan

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(
            self.ocp, json_file='acados_ocp.json')

    def clf_decisionfunction(self, clf, X_iter, x):

        dual_coef = clf.dual_coef_
        sup_vec = clf.support_vectors_
        const = clf.intercept_
        output = 0
        for i in range(sup_vec.shape[0]):
            output += dual_coef[0, i] * \
                exp(- (norm_2(x - sup_vec[i])**2)/(2*X_iter.var()))
        output += const

        return vertcat(output)
