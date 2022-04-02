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
        self.simu = RobotSimulator(self.conf, np.array(
            [0., 0.]), np.array([0., 0.]), self.robot)
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
        self.q = np.empty((2, self.N*10+1))*nan
        self.v = np.empty((2, self.N*10+1))*nan
        self.tau = np.empty((2, self.N*10+1))*nan
        self.niter = 0

    def compute_problem(self, q0, v0):
        if (self.v[:, 0] > self.v_max).all() or (self.v[:, 0] < self.v_min).all() or (self.q[:, 0] > self.q_max).all() or (self.q[:, 0] < self.q_min).all():
            # print('Unfeasible initial conditions')
            return 0

        # Simulate the controller with an increasing number of iterations:
        for k in range(1, 10):
            t = 0.0
            self.niter = 0

            self.q.fill(nan)
            self.v.fill(nan)
            self.tau.fill(nan)

            self.simu.init(q0, v0, True)

            self.q[:, 0] = self.simu.q
            self.v[:, 0] = self.simu.v

            for i in range(1, self.N*k):
                time_start = time.time()

                sg = self.robot.gravity(self.q[:, i-1])
                self.tau[:, i] = - self.kd*self.v[:, i-1] + sg

                if (self.tau[:, i] < self.tau_min).all():
                    self.tau[:, i] = self.tau_min
                elif (self.tau[:, i] > self.tau_max).all():
                    self.tau[:, i] = self.tau_max

                # send joint torques to simulator
                self.simu.simulate(self.tau[:, i], self.dt, self.ndt)

                # read current state from simulator
                self.q[:, i] = self.simu.q
                self.v[:, i] = self.simu.v

                if (self.v[:, i] > self.v_max).all() or (self.v[:, i] < self.v_min).all() or (self.q[:, i] > self.q_max).all() or (self.q[:, i] < self.q_min).all():
                    self.niter = i
                    # print('constraints violated')
                    return 0

                if norm(self.v[:, i]) < 0.01:
                    self.niter = i
                    # print('zero velocity reached')
                    return 1

                t += self.dt
                self.niter += 1

                time_spent = time.time() - time_start
                if(self.simulate_real_time and time_spent < self.dt):
                    time.sleep(self.dt-time_spent)

            if k > 1:
                print('try with', round(self.conf.T_SIMULATION*k, 2), 'seconds')

        # print('It neither violated the constraints or stopped')
        return 0
