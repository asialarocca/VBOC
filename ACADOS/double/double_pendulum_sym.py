import time
import scipy.linalg as lin
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos
import matplotlib.pyplot as plt

start_time = time.time()

model_name = 'double_pendulum_ode'

# constants
m1 = 0.4  # mass of the first link [kg]
m2 = 0.4  # mass of the second link [kg]
g = 9.81  # gravity constant [m/s^2]
l1 = 0.8  # length of the first link [m]
l2 = 0.8  # length of the second link [m]

# states
theta1 = SX.sym('theta1')
theta2 = SX.sym('theta2')
dtheta1 = SX.sym('dtheta1')
dtheta2 = SX.sym('dtheta2')
x = vertcat(theta1, theta2, dtheta1, dtheta2)

# controls
C1 = SX.sym('C1')
C2 = SX.sym('C2')
u = vertcat(C1, C2)

# xdot
theta1_dot = SX.sym('theta1_dot')
theta2_dot = SX.sym('theta1_dot')
dtheta1_dot = SX.sym('dtheta2_dot')
dtheta2_dot = SX.sym('dtheta2_dot')
xdot = vertcat(theta1_dot, theta2_dot, dtheta1_dot, dtheta2_dot)

# parameters
p = []

# dynamics
f_expl = vertcat(dtheta1, dtheta2, (l1 ** 2 * l2 * m2 * dtheta1 ** 2 * sin(-2 * theta2 + 2 * theta1) + 2 * C2 * cos(-theta2 + theta1) * l1 + 2 * (g * sin(-2 * theta2 + theta1) * l1 * m2 / 2 + sin(-theta2 + theta1) * dtheta2 ** 2 * l1 * l2 * m2 + g * l1 * (m1 + m2 / 2) * sin(theta1) - C1) * l2) / l1 ** 2 / l2 / (m2 * cos(-2 * theta2 + 2 * theta1) - 2 * m1 - m2), (-g *
                 l1 * l2 * m2 * (m1 + m2) * sin(-theta2 + 2 * theta1) - l1 * l2 ** 2 * m2 ** 2 * dtheta2 ** 2 * sin(-2 * theta2 + 2 * theta1) - 2 * dtheta1 ** 2 * l1 ** 2 * l2 * m2 * (m1 + m2) * sin(-theta2 + theta1) + 2 * C1 * cos(-theta2 + theta1) * l2 * m2 + l1 * (m1 + m2) * (sin(theta2) * g * l2 * m2 - 2 * C2)) / l2 ** 2 / l1 / m2 / (m2 * cos(-2 * theta2 + 2 * theta1) - 2 * m1 - m2))
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
Q = 2*np.diag([0.0, 0.0, 1e-2, 1e-2])
R = 2*np.diag([0.0, 0.0])
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
Cmax = 10
thetamax = np.pi/2 + np.pi
thetamin = np.pi
dthetamax = 2.

ocp.constraints.lbu = np.array([-Cmax, -Cmax])
ocp.constraints.ubu = np.array([Cmax, Cmax])
ocp.constraints.idxbu = np.array([0, 1])
ocp.constraints.lbx = np.array([thetamin, thetamin, -dthetamax, -dthetamax])
ocp.constraints.ubx = np.array([thetamax, thetamax, dthetamax, dthetamax])
ocp.constraints.idxbx = np.array([0, 1, 2, 3])
ocp.constraints.lbx_e = np.array([thetamin, thetamin, -dthetamax, -dthetamax])
ocp.constraints.ubx_e = np.array([thetamax, thetamax, dthetamax, dthetamax])
ocp.constraints.idxbx_e = np.array([0, 1, 2, 3])

ocp.solver_options.nlp_solver_type = 'SQP'

# Initial cnditions
ocp.constraints.x0 = np.array([4.426990816987241, 4.026990816987241, 0., 1.1])

# Solver
ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
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

t = np.linspace(0, Tf, N+1)

plt.figure()
plt.subplot(2, 1, 1)
line, = plt.step(t, np.append([simU[0, 0]], simU[:, 0]))
plt.ylabel('$C1$')
plt.xlabel('$t$')
plt.hlines(Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
plt.hlines(-Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
plt.ylim([-1.2*Cmax, 1.2*Cmax])
plt.title('Controls')
plt.grid()
plt.subplot(2, 1, 2)
line, = plt.step(t, np.append([simU[0, 1]], simU[:, 1]))
plt.ylabel('$C2$')
plt.xlabel('$t$')
plt.hlines(Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
plt.hlines(-Cmax, t[0], t[-1], linestyles='dashed', alpha=0.7)
plt.ylim([-1.2*Cmax, 1.2*Cmax])
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