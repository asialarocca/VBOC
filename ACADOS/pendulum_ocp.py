import sys
sys.path.insert(0, '../common')
from acados_template import AcadosOcp, AcadosOcpSolver
from pendulum_model import export_pendulum_ode_model
import numpy as np
import scipy.linalg as lin
from utils import plot_pendulum

# create ocp object to formulate the OCP
ocp = AcadosOcp()

# set model
model = export_pendulum_ode_model()
ocp.model = model

Tf = 1.0
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
ny_e = nx
N = 20

# set dimensions
ocp.dims.N = N

# set cost
Q = 2*np.diag([0.0, 1e-2])
R = 2*np.diag([0.0])
ocp.cost.W_e = Q
ocp.cost.W = lin.block_diag(Q, R)

ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.cost_type_e = 'LINEAR_LS'

ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx,:nx] = np.eye(nx)
Vu = np.zeros((ny, nu))
Vu[2,0] = 1.0
ocp.cost.Vu = Vu
ocp.cost.Vx_e = np.eye(nx)

ocp.cost.yref  = np.zeros((ny, ))
ocp.cost.yref_e = np.zeros((ny_e, ))

# set constraints
Fmax = 10
ocp.constraints.lbu = np.array([-Fmax])
ocp.constraints.ubu = np.array([+Fmax])
ocp.constraints.idxbu = np.array([0])
thetamax = np.pi/2
thetamin = 0.0
dthetamax = 10.
ocp.constraints.lbx = np.array([thetamin,-dthetamax])
ocp.constraints.ubx = np.array([thetamax,dthetamax])
ocp.constraints.idxbx = np.array([0,1])
ocp.constraints.lbx_e = np.array([thetamin,-dthetamax])
ocp.constraints.ubx_e = np.array([thetamax,dthetamax])
ocp.constraints.idxbx_e = np.array([0,1])

# set options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'

# set prediction horizon
ocp.solver_options.tf = Tf

# Initial cnditions
ocp.constraints.x0 = np.array([0., 4.])

simX = np.ndarray((N+1, nx))
simU = np.ndarray((N, nu))

# Solver
ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

status = ocp_solver.solve()

if status != 0:
	ocp_solver.print_statistics()
	raise Exception(f'acados returned status {status}.')

# get solution
for i in range(N):
	simX[i,:] = ocp_solver.get(i, "x")
	simU[i,:] = ocp_solver.get(i, "u")
simX[N,:] = ocp_solver.get(N, "x")

ocp_solver.print_statistics()

plot_pendulum(np.linspace(0, Tf, N+1), Fmax, simU, simX, latexify=False)

