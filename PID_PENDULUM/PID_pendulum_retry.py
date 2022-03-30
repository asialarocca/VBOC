import numpy as np
from numpy import nan
import time
import sys
from arc.utils.robot_loaders import loadPendulum
from arc.utils.robot_wrapper import RobotWrapper
from arc.utils.robot_simulator import RobotSimulator
import pinocchio as pin
from numpy.linalg import norm as norm


class PIDpendulum:

	def __init__(self, conf):
		r = loadPendulum()
		self.conf = conf
		self.robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
		self.simu = RobotSimulator(self.conf, np.array([0.]), np.array([0.]), self.robot)
		self.N = int(conf.T_SIMULATION/conf.dt)   #number of iterations of the first simulation
		self.tau_max = conf.tau_max
		self.tau_min = conf.tau_min
		self.q_max = conf.q_max
		self.q_min = conf.q_min
		self.v_max = conf.v_max
		self.v_min = conf.v_min
		self.q = np.empty(self.N*sum(range(1, 10)))*nan
		self.v = np.empty(self.N*sum(range(1, 10)))*nan
		self.tau = np.empty(self.N*sum(range(1, 10)))*nan   
		self.niter = 0    #number of simulated iterations in the current simulation

	def compute_problem(self, q0, v0):
		# Check initial conditions feasibility: 
		if v0 > self.v_max or v0 < self.v_min or q0 > self.q_max or q0 < self.q_min:
			print('Unfeasible initial conditions')
			return 0

		# Simulate the controller with an increasing number of iterations:
		for k in range(1, 10):
			# Initialize the simulation:
			t = 0.0
			self.niter = 0 
			
			self.q.fill(nan)
			self.v.fill(nan)
			self.tau.fill(nan)
			
			self.simu.init(q0, v0, True)
			
			self.q[0] = self.simu.q
			self.v[0] = self.simu.v
			
			# Simulate the controller:
			for i in range(1, self.N*k):
				time_start = time.time()
				
				# Control law:
				self.tau[i] = - self.conf.kd*self.v[i-1]
				
				#M = self.robot.mass(np.array([self.q[i]]))
				
				# Control limits:
				if self.tau[i] < self.tau_min:
					self.tau[i] = self.tau_min
				elif self.tau[i] > self.tau_max:
					self.tau[i] = self.tau_max
				
				# Send joint torques to the simulator:
				self.simu.simulate(self.tau[i], self.conf.dt, self.conf.ndt)
				
				# Read current state from the simulator:
				self.q[i] = self.simu.q
				self.v[i] = self.simu.v
				
				# Check position and velocity contraints:
				if self.v[i] > self.v_max or self.v[i] < self.v_min or self.q[i] > self.q_max or self.q[i] < self.q_min:
					# print('constraints violated')
					return 0
				
				# Check if the robot stopped:
				if norm(self.v[i]) < 0.01:
					# print('zero velocity reached')
					return 1

				t += self.conf.dt
				self.niter += 1
					
				time_spent = time.time() - time_start
				if(self.conf.simulate_real_time and time_spent < self.conf.dt): 
					time.sleep(self.conf.dt-time_spent)
			
			if k > 1:
				print('try with', round(self.conf.T_SIMULATION*k,2), 'seconds')
		
		print('it did not stop anyway')
		return 0

