import time
from utils import plot_pendulum
import scipy.linalg as lin
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template import AcadosModel
from casadi import SX, vertcat, sin
import matplotlib.pyplot as plt

model_name = "pendulum_ode"

# constants
m = 0.5  # mass of the ball [kg]
g = 9.81  # gravity constant [m/s^2]
d = 0.3  # length of the rod [m]
#b = 0.1  # damping

# states
theta = SX.sym("theta")
dtheta = SX.sym("dtheta")
x = vertcat(theta, dtheta)

# controls
F = SX.sym("F")
u = vertcat(F)

# xdot
theta_dot = SX.sym("theta_dot")
dtheta_dot = SX.sym("dtheta_dot")
xdot = vertcat(theta_dot, dtheta_dot)

# parameters
p = []

# dynamics
f_expl = vertcat(dtheta, (m * g * d * sin(theta) + F ) / (d * d * m)) #- b * dtheta
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

Tf = 1.
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
ny_e = nx
N = int(100 * Tf)

# set dimensions
ocp.dims.N = N

# set prediction horizon
ocp.solver_options.tf = Tf

# set cost
ocp.cost.W_0 = 2 * np.diag([0., 1., 0.])
ocp.cost.W = 2 * np.diag([1., 0., 0.])
ocp.cost.W_e = 2 * np.diag([0., 0.])

ocp.cost.cost_type = "LINEAR_LS"
ocp.cost.cost_type_e = "LINEAR_LS"

ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx, :nx] = np.eye(nx)
ocp.cost.Vu = np.zeros((ny, nu))
ocp.cost.Vu[2, 0] = 1.0
ocp.cost.Vx_e = np.eye(nx)

# set constraints
Fmax = 3
thetamax = np.pi / 2
thetamin = 0.0
dthetamax = 10.0

ocp.cost.yref_0 = np.array([thetamin, dthetamax, 0.])
ocp.cost.yref = np.array([thetamax, 0., 0.])
ocp.cost.yref_e = np.array([0., 0.])

# Calculate max torque to sustain the gravity force:
print("Max torque to sustain the gravity force:", m * sin(thetamax) * 9.81 * d)

ocp.constraints.lbu = np.array([-Fmax])
ocp.constraints.ubu = np.array([+Fmax])
ocp.constraints.idxbu = np.array([0])
ocp.constraints.lbx = np.array([thetamin, -dthetamax])
ocp.constraints.ubx = np.array([thetamax, dthetamax])
ocp.constraints.idxbx = np.array([0, 1])
ocp.constraints.lbx_e = np.array([thetamax, 0.0])
ocp.constraints.ubx_e = np.array([thetamax, 0.0])
ocp.constraints.idxbx_e = np.array([0, 1])
ocp.constraints.lbx_0 = np.array([thetamin, -dthetamax])
ocp.constraints.ubx_0 = np.array([thetamin, dthetamax])
ocp.constraints.idxbx_0 = np.array([0, 1])

ocp.solver_options.nlp_solver_type = "SQP"

# Solver
ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

# x_guess = np.array([thetamin, dthetamax])

ls = np.linspace(thetamin, thetamax, N, endpoint=False)
vel = np.full(N, dthetamax)
x_guess = np.append([ls], [vel], axis=0).T

for i in range(N):
    ocp_solver.set(i, "x", x_guess[i])
ocp_solver.set(N, "x", np.array([thetamax, 0.0]))
    
start_time = time.time()

status = ocp_solver.solve()

print("Execution time: %s seconds" % (time.time() - start_time))

if status == 0:
    ocp_solver.print_statistics()

    # get solution
    simX = np.ndarray((N + 1, nx))
    simU = np.ndarray((N, nu))

    for i in range(N):
        simX[i, :] = ocp_solver.get(i, "x")
        simU[i, :] = ocp_solver.get(i, "u")
    simX[N, :] = ocp_solver.get(N, "x")

    plot_pendulum(np.linspace(0, Tf, N + 1), Fmax, simU, simX, latexify=False)

# def model_pid(x, t, Fmax, thetamax, dthetamax):
#     y = x[0]
#     dydt = x[1]
#     u = -10 * dydt

#     if u >= Fmax:
#         u = Fmax
#     elif u <= -Fmax:
#         u = -Fmax

#     dy2dt2 = (m * g * d * sin(y) + u - b * dydt) / (d * d * m)
#     return [dydt, dy2dt2]


# times = np.linspace(0, Tf, int(100 * Tf))
# solution = odeint(model_pid, [q0, v0], times, args=(Fmax, thetamax, dthetamax))

# solution[:, 0] = [
#     np.sign(solution[i, 0]) * thetamax
#     if abs(solution[i, 0]) > thetamax
#     else solution[i, 0]
#     for i in range(int(100 * Tf))
# ]
# solution[:, 1] = [
#     np.sign(solution[i, 1]) * dthetamax
#     if abs(solution[i, 1]) > dthetamax
#     else solution[i, 1]
#     for i in range(int(100 * Tf))
# ]

# plt.figure()
# plt.plot(times, solution[:, 0], "r-", linewidth=1)
# plt.figure()
# plt.plot(times, solution[:, 1], "r-", linewidth=1)
# plt.show()


