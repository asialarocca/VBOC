from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, cos, sin, dot
import urdf2casadi.urdfparser as u2c
import os


class OCPtriplependulum:
    def __init__(self):

        # --------------------SET MODEL--------------------
        # -------------------------------------------------
        model_name = "ur5_model"

        self.gravity = [0, 0, -9.81]
        self.root = "world"
        self.tip = "tool0"

        self.ur5 = u2c.URDFparser()
        path_to_urdf = os.path.dirname(
            os.path.abspath(__file__)) + '/ur5.urdf'
        self.ur5.from_file(path_to_urdf)
        n_joints = self.ur5.get_n_joints(self.root, self.tip)

        # states
        q = SX.sym("qs", n_joints)
        qdot = SX.sym("qsdot", n_joints)
        q_dot = SX.sym("qs_dot", n_joints)
        qdot_dot = SX.sym("qsdot_dot", n_joints)
        self.x = vertcat(q, qdot)
        xdot = vertcat(q_dot, qdot_dot)

        # controls
        self.u = SX.sym("C", n_joints)

        # parameters
        w1 = SX.sym("w1") 
        w2 = SX.sym("w2")
        w3 = SX.sym("w3") 
        p = vertcat(w1, w2, w3)

        # dynamics
        func = self.ur5.get_forward_dynamics_aba(self.root, self.tip, gravity=self.gravity)
        f_expl = vertcat(qdot, func(q, qdot, self.u))
        f_impl = xdot - f_expl

        self.inv_dyn = self.ur5.get_inverse_dynamics_rnea(self.root, self.tip, gravity=self.gravity)

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

        # times
        self.Tf = 1.
        self.ocp.solver_options.tf = self.Tf  # prediction horizon
        self.N = int(100 * self.Tf)
        self.ocp.dims.N = self.N

        # ocp model
        self.ocp.model = self.model
        
        # cost
        self.ocp.cost.cost_type_0 = 'EXTERNAL'

        self.ocp.model.cost_expr_ext_cost_0 = dot(p,qdot)
        self.ocp.parameter_values = np.array([0., 0., 0.])

        # set constraints
        self.Cmax = np.array([150., 150., 150.]) # np.array([28., 28., 28.])
        self.Cmin = - self.Cmax
        self.xmax = np.array([3.14, 3.14, 3.14, 3.15, 3.15, 3.15]) # np.array([3.14, 3.14, 3.14, 3.2, 3.2, 3.2]) 
        self.xmin = - self.xmax

        self.dthetamax = self.xmax[3]
        self.dthetamin = self.xmin[3]
        self.thetamax = self.xmax[0]
        self.thetamin = self.xmin[0]

        self.ocp.constraints.lbu = np.copy(self.Cmin)
        self.ocp.constraints.ubu = np.copy(self.Cmax)
        self.ocp.constraints.idxbu = np.array([0, 1, 2])
        self.ocp.constraints.lbx = np.copy(self.xmin)
        self.ocp.constraints.ubx = np.copy(self.xmax)
        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5])
        self.ocp.constraints.lbx_e = np.copy(self.xmin)
        self.ocp.constraints.ubx_e = np.copy(self.xmax)
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2, 3, 4, 5])
        self.ocp.constraints.lbx_0 = np.copy(self.xmin)
        self.ocp.constraints.ubx_0 = np.copy(self.xmax)
        self.ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5])

        self.ocp.constraints.C = np.zeros((3,6))
        self.ocp.constraints.D = np.zeros((3,3))
        self.ocp.constraints.lg = np.zeros((3))
        self.ocp.constraints.ug = np.zeros((3))

        # Cartesian constraints:
        # radius = 0.1
        # center = [0.2,0.2,0.2]
        # self.kine = self.get_kinematics(q)
        # self.model.con_h_expr = (self.kine[0]-center[0])**2 + (self.kine[1]-center[1])**2 + (self.kine[2]-center[2])**2 - radius**2
        # self.ocp.constraints.lh = np.array([0.])
        # self.ocp.constraints.uh = np.array([1e6])
        
        # options
        self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.hessian_approx = "EXACT"
        self.ocp.solver_options.exact_hess_constr = 0
        # self.ocp.solver_options.exact_hess_cost = 0
        self.ocp.solver_options.exact_hess_dyn = 0
        self.ocp.solver_options.nlp_solver_tol_stat = 1e-3
        self.ocp.solver_options.qp_solver_iter_max = 100
        self.ocp.solver_options.nlp_solver_max_iter = 1000
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        self.ocp.solver_options.alpha_reduction = 0.3
        self.ocp.solver_options.alpha_min = 1e-2
        self.ocp.solver_options.levenberg_marquardt = 1e-2

        # -------------------------------------------------

    def get_inverse_dynamics(self, q, qdot):

        return self.inv_dyn(q, qdot, np.zeros((3,))).toarray().reshape(3,)
    
    def get_kinematics(self, q):

        dual_quaternion = self.ur5.get_forward_kinematics(self.root, self.tip)["dual_quaternion_fk"](q)
        return 2*dual_quaternion[4:7]*dual_quaternion[:3]


class OCPtriplependulumINIT(OCPtriplependulum):
    def __init__(self):

        # inherit initialization
        super().__init__()

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json", build=True)

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
            self.ocp_solver.constraints_set(i, 'C', np.zeros((3,6)))
            self.ocp_solver.constraints_set(i, 'D', np.zeros((3,3)))
            self.ocp_solver.constraints_set(i, 'lg', np.zeros((3)))
            self.ocp_solver.constraints_set(i, 'ug', np.zeros((3)))

        C = np.zeros((3,6))
        d = np.array([p.tolist()])
        dt = np.transpose(d)
        C[:,3:] = np.identity(3)-np.matmul(dt,d) 
        self.ocp_solver.constraints_set(0, "C", C, api='new') 

        self.ocp_solver.constraints_set(0, "lbx", q_init_lb)
        self.ocp_solver.constraints_set(0, "ubx", q_init_ub)

        self.ocp_solver.constraints_set(self.N, "lbx", q_fin_lb)
        self.ocp_solver.constraints_set(self.N, "ubx", q_fin_ub)
        self.ocp_solver.set(self.N, 'x', x_sol_guess[-1])
        self.ocp_solver.set(self.N, 'p', p)

        # Solve the OCP:
        status = self.ocp_solver.solve()

        # print(status)
        # print(self.ocp_solver.get_stats('residuals'))
        
        return status


class SYMtriplependulumINIT(OCPtriplependulum):
    def __init__(self):

        # inherit initialization
        super().__init__()

        self.model.p = []

        sim = AcadosSim()
        sim.model = self.model
        sim.solver_options.T = 1e-2
        sim.solver_options.num_stages = 4
        self.acados_integrator = AcadosSimSolver(sim)
