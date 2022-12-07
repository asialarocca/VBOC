import scipy.linalg as lin
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template import AcadosModel
from casadi import SX, vertcat, exp, fmax, tanh, Function
import matplotlib.pyplot as plt
import os
import casadi as cs
import urdf2casadi.urdfparser as u2c


class OCPdoublependulum:
    def __init__(self):

        # --------------------SET MODEL--------------------
        # -------------------------------------------------
        model_name = "double_pendulum_ode"

        gravity = [0, 0, -9.81]
        root = "base_link"
        tip = "tool0"

        ur5 = u2c.URDFparser()
        path_to_urdf = absPath = os.path.dirname(
            os.path.abspath(__file__)) + '/ur5_fix.urdf'
        ur5.from_file(path_to_urdf)

        n_joints = ur5.get_n_joints(root, tip)

        # ERROR_MSG = 'You should set the environment variable LOCOSIM_DIR"\n'
        # path = os.environ.get('LOCOSIM_DIR', ERROR_MSG)
        # srdf = path + "/robot_urdf/" + "ur5.srdf"
        # urdf = path + "/robot_urdf/generated_urdf/" + "ur5_fix.urdf"
        # self.robot = RobotWrapper.BuildFromURDF(urdf, [path, srdf])
        # self.frame_name = conf.robot_params['ur5']["ee_frame"]

        # states
        q = cs.SX.sym("qs", n_joints)
        qdot = cs.SX.sym("qsdot", n_joints)
        q_dot = cs.SX.sym("qs_dot", n_joints)
        qdot_dot = cs.SX.sym("qsdot_dot", n_joints)
        self.x = vertcat(q, qdot)
        xdot = vertcat(q_dot, qdot_dot)

        # controls
        self.u = cs.SX.sym("C", n_joints)

        # parameters
        p = []

        # dynamics

        func = ur5.get_forward_dynamics_aba(root, tip, gravity=gravity)
        f_expl = vertcat(qdot, func(q, qdot, self.u))
        f_impl = xdot - f_expl

        self.model = AcadosModel()

        self.model.f_impl_expr = f_impl
        self.model.f_expl_expr = f_expl
        self.model.x = self.x
        self.model.xdot = xdot
        self.model.u = self.u
        self.model.p = p
        self.model.name = model_name
        # -------------------------------------------------

        # ---------------------SET OCP---------------------
        # -------------------------------------------------
        self.ocp = AcadosOcp()

        # dimensions
        self.Tf = 0.1
        self.ocp.solver_options.tf = self.Tf  # prediction horizon

        self.N = int(100 * self.Tf)
        self.ocp.dims.N = self.N

        self.nx = self.model.x.size()[0]
        self.nu = self.model.u.size()[0]
        ny = self.nx + self.nu
        ny_e = self.nx

        # cost
        Q = np.diag([1e2, 1e2, 1e-2, 1e-2])
        R = np.diag([1e-4, 1e-4])

        self.ocp.cost.W_e = Q
        self.ocp.cost.W = lin.block_diag(Q, R)

        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        self.ocp.cost.Vx = np.zeros((ny, self.nx))
        self.ocp.cost.Vx[: self.nx, : self.nx] = np.eye(self.nx)
        self.ocp.cost.Vu = np.zeros((ny, self.nu))
        self.ocp.cost.Vu[self.nx:, : self.nu] = np.eye(self.nu)
        self.ocp.cost.Vx_e = np.eye(self.nx)

        # reference
        self.ocp.cost.yref = np.array([-1., -2.1, 0., 0., 0., 0.])
        self.ocp.cost.yref_e = np.array([-1., -2.1, 0., 0.])

        self.x_guess = self.ocp.cost.yref[:self.nx]

        # # set constraints
        self.Cmax = np.array([55., 20.])
        self.Cmin = - self.Cmax
        self.xmax = np.array([-0.5, -2., 3.14, 3.14])
        self.xmin = - self.xmax
        self.xmin[0] = -1.5
        self.xmin[1] = -3.

        self.ocp.constraints.lbu = self.Cmin
        self.ocp.constraints.ubu = self.Cmax
        self.ocp.constraints.idxbu = np.array([0, 1])
        self.ocp.constraints.lbx = self.xmin
        self.ocp.constraints.ubx = self.xmax
        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3])
        self.ocp.constraints.lbx_e = self.xmin
        self.ocp.constraints.ubx_e = self.xmax
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2, 3])

        self.ocp.constraints.x0 = np.zeros((self.nx,))

        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.integrator_type = "ERK"
        #self.ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        # self.ocp.solver_options.qp_tol = 1e-4
        # self.ocp.solver_options.tol = 1e-4

        # -------------------------------------------------

    def compute_problem(self, q0, v0):  # , x_guess):
    
        print(q0,v0)

        self.ocp_solver.reset()

        x0 = np.array([q0[0], q0[1], v0[0], v0[1]])

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        for i in range(self.N + 1):
            self.ocp_solver.set(i, "x", self.x_guess)

        status = self.ocp_solver.solve()

        if status == 0:
            # get solution
            simX = np.ndarray((self.N+1, self.nx))
            simU = np.ndarray((self.N, self.nu))

            for i in range(self.N):
                simX[i, :] = self.ocp_solver.get(i, "x")
                simU[i, :] = self.ocp_solver.get(i, "u")
            simX[self.N, :] = self.ocp_solver.get(self.N, "x")

            t = np.linspace(0, self.Tf, self.N+1)

            plt.figure()
            plt.subplot(2, 1, 1)
            line, = plt.step(t, np.append([simU[0, 0]], simU[:, 0]))
            plt.ylabel('$C1$')
            plt.xlabel('$t$')
            plt.hlines(self.Cmax[0], t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.hlines(self.Cmin[0], t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.ylim([1.2*self.Cmin[0], 1.2*self.Cmax[0]])
            plt.title('Controls')
            plt.grid()
            plt.subplot(2, 1, 2)
            line, = plt.step(t, np.append([simU[0, 1]], simU[:, 1]))
            plt.ylabel('$C2$')
            plt.xlabel('$t$')
            plt.hlines(self.Cmax[1], t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.hlines(self.Cmin[1], t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.ylim([1.2*self.Cmin[1], 1.2*self.Cmax[1]])
            plt.grid()

            plt.figure()
            plt.subplot(4, 1, 1)
            line, = plt.plot(t, simX[:, 0])
            plt.ylabel('$theta1$')
            plt.xlabel('$t$')
            plt.title('States')
            plt.grid()
            plt.subplot(4, 1, 2)
            line, = plt.plot(t, simX[:, 1])
            plt.ylabel('$theta2$')
            plt.xlabel('$t$')
            plt.grid()
            plt.subplot(4, 1, 3)
            line, = plt.plot(t, simX[:, 2])
            plt.ylabel('$dtheta1$')
            plt.xlabel('$t$')
            plt.grid()
            plt.subplot(4, 1, 4)
            line, = plt.plot(t, simX[:, 3])
            plt.ylabel('$dtheta2$')
            plt.xlabel('$t$')
            plt.grid()

            plt.show()
            return 1
        else:
            return 0


class OCPdoublependulumINIT(OCPdoublependulum):
    def __init__(self):

        # inherit initialization
        super().__init__()

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")


class OCPdoublependulumNN(OCPdoublependulum):
    def __init__(self, nn, mean, std):

        # inherit initialization
        super().__init__()

        # nonlinear terminal constraints (svm)
        self.model.con_h_expr_e = self.nn_decisionfunction(nn, self.x, mean, std)
        self.ocp.constraints.lh_e = np.array([0.5])
        self.ocp.constraints.uh_e = np.array([1.1])

        self.ocp.constraints.lbx_e = self.xmin
        self.ocp.constraints.ubx_e = self.xmax
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2, 3])

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

    def nn_decisionfunction(self, nn, x, mean, std):

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
            it += 1

        return out[1]
