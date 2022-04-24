from acados_template import AcadosModel
from casadi import SX, vertcat, sin, Function, exp, norm_2


def export_pendulum_ode_model():

    model_name = 'pendulum_ode'

    # constants
    m = 0.4  # mass of the ball [kg]
    g = 9.81  # gravity constant [m/s^2]
    l = 0.8  # length of the rod [m]
    b = 0.  # damping

    # set up states & controls
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
    f_expl = vertcat(dtheta, (g*sin(theta)+F/m-b*dtheta/m)/(l*l))

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name

    return model
