import numpy as np
from numpy import nan
import time
from example_robot_data.robots_loader import loadDoublePendulum
from arc.utils.robot_wrapper import RobotWrapper
from arc.utils.robot_simulator import RobotSimulator
import pinocchio as pin
from numpy.linalg import norm as norm


class PIDdoublependulum:

    def __init__(self, conf):
        r = loadDoublePendulum()
        self.conf = conf
        self.robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
        self.N = int(conf.T_SIMULATION/conf.dt)
        self.tau_max = conf.tau_max
        self.tau_min = conf.tau_min
        self.q_max = conf.q_max
        self.q_min = conf.q_min
        self.v_max = conf.v_max
        self.v_min = conf.v_min
        self.dt = conf.dt
        self.ndt = conf.ndt
        self.PRINT_T = conf.PRINT_T
        self.kd = conf.kd
        self.simulate_real_time = conf.simulate_real_time
        self.q = np.empty((2, self.N+1))*nan
        self.v = np.empty((2, self.N+1))*nan
        self.niter = 0

    def compute_problem(self, q0, v0):
        t = 0.0

        tau = np.empty((2, self.N+1))*nan    # joint torques

        self.q.fill(nan)
        self.v.fill(nan)

        PRINT_N = int(self.PRINT_T/self.dt)

        simu = RobotSimulator(self.conf, q0, v0, self.robot)
        self.niter = 0

        self.q[:, 0] = simu.q
        self.v[:, 0] = simu.v

        if (self.v[:, 0] > self.v_max).all() or (self.v[:, 0] < self.v_min).all() or (self.q[:, 0] > self.q_max).all() or (self.q[:, 0] < self.q_min).all():
            # print('Unfeasible initial conditions')
            return 0

        for i in range(1, self.N):
            time_start = time.time()

            sg = self.robot.gravity(self.q[:, i-1])
            tau[:, i] = - self.kd*self.v[:, i-1] + sg

            if (tau[:, i] < self.tau_min).all():
                tau[:, i] = self.tau_min
            elif (tau[:, i] > self.tau_max).all():
                tau[:, i] = self.tau_max

            # send joint torques to simulator
            simu.simulate(tau[:, i], self.dt, self.ndt)

            # read current state from simulator
            self.q[:, i] = simu.q
            self.v[:, i] = simu.v

            if (self.v[:, i] > self.v_max).all() or (self.v[:, i] < self.v_min).all() or (self.q[:, i] > self.q_max).all() or (self.q[:, i] < self.q_min).all():
                self.niter = i
                # print('constraints violated')
                return 0

            if norm(self.v[:, i]) < 0.01:
                self.niter = i
                # print('zero velocity reached')
                return 1

            t += self.dt

            time_spent = time.time() - time_start
            if(self.simulate_real_time and time_spent < self.dt):
                time.sleep(self.dt-time_spent)

        self.niter = self.N
        # print('It neither violated the constraints or stopped')
        return 0
