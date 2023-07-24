from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, dot
import urdf2casadi.urdfparser as u2c
import os


class OCPUR5:
    def __init__(self):

        # --------------------SET MODEL--------------------
        # -------------------------------------------------
        model_name = "ur5_model"

        self.gravity = [0, 0, -9.81]
        self.root = "base_link"
        self.tip = "tool0"

        self.ur5 = u2c.URDFparser()
        path_to_urdf = os.path.dirname(
            os.path.abspath(__file__)) + '/ur5.urdf'
        self.ur5.from_file(path_to_urdf)
        self.n_joints = self.ur5.get_n_joints(self.root, self.tip)

        # states
        q = SX.sym("qs", self.n_joints)
        qdot = SX.sym("qsdot", self.n_joints)
        q_dot = SX.sym("qs_dot", self.n_joints)
        qdot_dot = SX.sym("qsdot_dot", self.n_joints)
        self.x = vertcat(q, qdot)
        xdot = vertcat(q_dot, qdot_dot)

        # controls
        self.u = SX.sym("C", self.n_joints)

        # parameters
        p = SX.sym("w", self.n_joints)

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
        self.ocp.parameter_values = np.zeros((self.n_joints))

        # set constraints
        u_limits = np.array([100., 80., 60., 1., 0.8, 0.6]) # np.array([150., 150., 150., 28., 28., 28.])
        x_limits = np.array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]) # np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14])

        self.Cmax = u_limits[:self.n_joints]
        self.Cmin = - self.Cmax
        self.xmax = np.concatenate((x_limits[:self.n_joints],x_limits[self.n_joints*2:self.n_joints*3]))
        self.xmin = - self.xmax

        self.xmax[1] = 0.

        self.ocp.constraints.lbu = np.copy(self.Cmin)
        self.ocp.constraints.ubu = np.copy(self.Cmax)
        self.ocp.constraints.idxbu = np.array(range(self.n_joints))
        self.ocp.constraints.lbx = np.copy(self.xmin)
        self.ocp.constraints.ubx = np.copy(self.xmax)
        self.ocp.constraints.idxbx = np.array(range(self.n_joints*2))
        self.ocp.constraints.lbx_e = np.copy(self.xmin)
        self.ocp.constraints.ubx_e = np.copy(self.xmax)
        self.ocp.constraints.idxbx_e = np.array(range(self.n_joints*2))
        self.ocp.constraints.lbx_0 = np.copy(self.xmin)
        self.ocp.constraints.ubx_0 = np.copy(self.xmax)
        self.ocp.constraints.idxbx_0 = np.array(range(self.n_joints*2))

        self.ocp.constraints.C = np.zeros((self.n_joints,self.n_joints*2))
        self.ocp.constraints.D = np.zeros((self.n_joints,self.n_joints))
        self.ocp.constraints.lg = np.zeros((self.n_joints))
        self.ocp.constraints.ug = np.zeros((self.n_joints))

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
        self.ocp.solver_options.qp_solver_tol_stat = 1e-3
        self.ocp.solver_options.qp_solver_iter_max = 100
        self.ocp.solver_options.nlp_solver_max_iter = 1000
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        self.ocp.solver_options.alpha_reduction = 0.3
        self.ocp.solver_options.alpha_min = 1e-2
        self.ocp.solver_options.levenberg_marquardt = 1e-2

        # -------------------------------------------------

    def get_inverse_dynamics(self, q, qdot):

        return self.inv_dyn(q, qdot, np.zeros((self.n_joints,))).toarray().reshape(self.n_joints,)
    
    def get_kinematics(self, q):

        dual_quaternion = self.ur5.get_forward_kinematics(self.root, self.tip)["dual_quaternion_fk"](q)
        return 2*dual_quaternion[4:7]*dual_quaternion[:3]


class OCPUR5INIT(OCPUR5):
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
            self.ocp_solver.constraints_set(i, 'C', np.zeros((self.n_joints,self.n_joints*2)))
            self.ocp_solver.constraints_set(i, 'D', np.zeros((self.n_joints,self.n_joints)))
            self.ocp_solver.constraints_set(i, 'lg', np.zeros((self.n_joints)))
            self.ocp_solver.constraints_set(i, 'ug', np.zeros((self.n_joints)))

        C = np.zeros((self.n_joints,self.n_joints*2))
        d = np.array([p.tolist()])
        dt = np.transpose(d)
        C[:,self.n_joints:] = np.identity(self.n_joints)-np.matmul(dt,d) 
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


class SYMUR5INIT(OCPUR5):
    def __init__(self):

        # inherit initialization
        super().__init__()

        self.model.p = []

        sim = AcadosSim()
        sim.model = self.model
        sim.solver_options.T = 1e-2
        sim.solver_options.num_stages = 4
        self.acados_integrator = AcadosSimSolver(sim)
