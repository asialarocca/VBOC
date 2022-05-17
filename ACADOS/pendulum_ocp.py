import time
from utils import plot_pendulum
import scipy.linalg as lin
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, exp, norm_2

start_time = time.time()

model_name = 'pendulum_ode'

# constants
m = 0.4  # mass of the ball [kg]
g = 9.81  # gravity constant [m/s^2]
d = 0.8  # length of the rod [m]
b = 0.  # damping

# states
theta = SX.sym('theta')
dtheta = SX.sym('dtheta')
x = vertcat(theta, dtheta)

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
f_expl = vertcat(dtheta, (g*sin(theta)+F/m)/(d*d))  # -b*dtheta/m
f_impl = xdot - f_expl

model = AcadosModel()

model.f_impl_expr = f_impl
model.f_expl_expr = f_expl
model.x = x
model.xdot = xdot
model.u = u
model.p = p
model.name = model_name

# create ocp object to formulate the OCP
ocp = AcadosOcp()

# ocp model
ocp.model = model

Tf = 0.1
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
ny_e = nx
N = 10

# set dimensions
ocp.dims.N = N

# set prediction horizon
ocp.solver_options.tf = Tf

# set cost
Q = 2*np.diag([0.0, 1e-2])
R = 2*np.diag([0.0])
ocp.cost.W_e = Q
ocp.cost.W = lin.block_diag(Q, R)

ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.cost_type_e = 'LINEAR_LS'

ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx, :nx] = np.eye(nx)
ocp.cost.Vu = np.zeros((ny, nu))
ocp.cost.Vu[nx:, :nu] = np.eye(nu)
ocp.cost.Vx_e = np.eye(nx)

ocp.cost.yref = np.zeros((ny, ))
ocp.cost.yref_e = np.zeros((ny_e, ))

# set constraints
Fmax = 10
thetamax = np.pi/2
thetamin = 0.0
dthetamax = 10.

# Calculate max torque to sustain the gravity force:
# print(sin(thetamax)*9.81*d)

ocp.constraints.lbu = np.array([-Fmax])
ocp.constraints.ubu = np.array([+Fmax])
ocp.constraints.idxbu = np.array([0])
ocp.constraints.lbx = np.array([thetamin, -dthetamax])
ocp.constraints.ubx = np.array([thetamax, dthetamax])
ocp.constraints.idxbx = np.array([0, 1])
ocp.constraints.lbx_e = np.array([thetamin, -dthetamax])
ocp.constraints.ubx_e = np.array([thetamax, dthetamax])
ocp.constraints.idxbx_e = np.array([0, 1])
#ocp.constraints.lh_e = np.array([-0.01])
#ocp.constraints.uh_e = np.array([0.01])

ocp.solver_options.nlp_solver_type = 'SQP'

# Initial cnditions
ocp.constraints.x0 = np.array([0.1, 0.001])

# Solver
ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

ocp_solver.reset()

ocp_solver.constraints_set(0, "lbu", 0.)
ocp_solver.constraints_set(0, "ubu", 0.)

status = ocp_solver.solve()

if status != 0:
    ocp_solver.print_statistics()
    raise Exception(f'acados returned status {status}.')

# get solution
simX = np.ndarray((N+1, nx))
simU = np.ndarray((N, nu))

for i in range(N):
    simX[i, :] = ocp_solver.get(i, "x")
    simU[i, :] = ocp_solver.get(i, "u")
simX[N, :] = ocp_solver.get(N, "x")

ocp_solver.print_statistics()
print("Execution time: %s seconds" % (time.time() - start_time))

plot_pendulum(np.linspace(0, Tf, N+1), Fmax, simU, simX, latexify=False)
