import scipy.linalg as lin
from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, MX, vertcat, sin, exp, fmax, tanh, norm_2

class OCPpendulum:
    def __init__(self, mean, std, params): #(self, clf, X_iter):

        # --------------------SET MODEL--------------------
        # -------------------------------------------------
        model_name = "pendulum_ode"

        # constants
        self.m = 0.5  # mass of the ball [kg]
        self.g = 9.81  # gravity constant [m/s^2]
        self.d = 0.3  # length of the rod [m]

        # states
        theta = SX.sym("theta")
        dtheta = SX.sym("dtheta")
        self.x = vertcat(theta, dtheta)

        # controls
        F = SX.sym("F")
        u = vertcat(F)

        # parameters
        p = []

        # dynamics
        f_expl = vertcat(
            dtheta,
            (self.m * self.g * self.d * sin(theta) + F)
            / (self.d * self.d * self.m),
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

        # time
        self.N = 1
        self.Tf = 1e-2 * self.N
        self.ocp.solver_options.tf = self.Tf

        # dimensions
        self.nx = self.model.x.size()[0]
        nu = self.model.u.size()[0]
        ny = self.nx + nu
        ny_e = self.nx

        self.ocp.dims.N = self.N

        # # nonlinear terminal constraints
        # self.model.con_h_expr_e = self.nn_decisionfunction(
        #     nn, mean, std, self.x)
        # self.ocp.constraints.lh_e = np.array([-0.0])
        # self.ocp.constraints.uh_e = np.array([1.1])

        # ocp model
        self.ocp.model = self.model

        # constraints
        self.Fmax = 3
        self.thetamax = np.pi / 4 + np.pi
        self.thetamin = - np.pi / 4 + np.pi
        self.dthetamax = 10.0

        self.ocp.constraints.lbu = np.array([-self.Fmax])
        self.ocp.constraints.ubu = np.array([+self.Fmax])
        self.ocp.constraints.idxbu = np.array([0])
        self.ocp.constraints.lbu_0 = np.array([-self.Fmax])
        self.ocp.constraints.ubu_0 = np.array([+self.Fmax])
        self.ocp.constraints.idxbu_0 = np.array([0])

        self.ocp.constraints.lbx_0 = np.array([self.thetamin, -self.dthetamax])
        self.ocp.constraints.ubx_0 = np.array([self.thetamax, self.dthetamax])
        self.ocp.constraints.idxbx_0 = np.array([0, 1])

        self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.hessian_approx = 'EXACT'
        self.ocp.solver_options.exact_hess_constr = 0
        # self.ocp.solver_options.exact_hess_cost = 0
        self.ocp.solver_options.exact_hess_dyn = 0
        # self.ocp.solver_options.nlp_solver_tol_stat = 1e-2
        # self.ocp.solver_options.nlp_solver_tol_comp = 1e-2
        # self.ocp.solver_options.tol = 1e-2
        self.ocp.solver_options.qp_solver_iter_max = 100
        self.ocp.solver_options.nlp_solver_max_iter = 1000
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        self.ocp.solver_options.alpha_reduction = 0.3
        self.ocp.solver_options.alpha_min = 1e-2
        self.ocp.solver_options.levenberg_marquardt = 1e-2

        # cost
        self.ocp.cost.cost_type_e = 'EXTERNAL'
        self.ocp.model.cost_expr_ext_cost_e = self.nn_decisionfunction(params, mean, std, self.x)
        # self.ocp.cost.cost_type_e = 'EXTERNAL'
        # self.ocp.model.cost_expr_ext_cost_e = self.svm_decisionfunction(clf, X_iter, self.x)

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

        # -------------------------------------------------

    def compute_problem(self, q0, v0):

        self.ocp_solver.reset()

        x0 = np.array([q0, v0])

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        x_guess = np.array([q0+v0*1e-2, v0*0.9])
        u_guess = np.array([-self.m * self.g * self.d * sin(q0)])

        # for i in range(1,self.N):
        #     self.ocp_solver.set(i, "x", x_guess)
        #     self.ocp_solver.set(i, "u", u_guess)

        self.ocp_solver.set(0, "u", u_guess)
        self.ocp_solver.set(0, "x", x0)
        self.ocp_solver.set(self.N, "x", x_guess)

        status = self.ocp_solver.solve()

        if status == 0:
            return 1
        # elif status == 4:
        #     return 0
        else:
            return 0

    def nn_decisionfunction(self, params, mean, std, x):

        out = (x - mean) / std
        it = 2

        for param in params:
            param = SX(param.tolist())
            if it % 2 == 0:
                out = param @ out
            else:
                out = param + out

                if it == 3:
                    out = fmax(0.0, out)
                elif it == 5:
                    out = fmax(0.0, out) #tanh(out)
                # else:
                #    out = 1 / (1 + exp(-out))

            it += 1

        return vertcat(out[0])
        # return vertcat(out[1]-out[0])
    
    def svm_decisionfunction(self, svm, X_iter, x):

        dual_coef = svm.dual_coef_
        sup_vec = svm.support_vectors_
        const = svm.intercept_
        output = 0
        for i in range(sup_vec.shape[0]):
            output += dual_coef[0, i] * exp(
                -(norm_2(x - sup_vec[i]) ** 2) / (2 * X_iter.var())
            )
        output += const

        return vertcat(-output)
