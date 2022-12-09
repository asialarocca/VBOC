import pinocchio as se3
import tsid
import numpy as np
from numpy import nan
import os
import time
import subprocess
from utils_tsid.robot_loaders import loadUR_urdf
from numpy.linalg import norm as norm


class TsidManipulator:
    ''' Standard TSID formulation for a robot manipulator
        - end-effector task
        - Postural task
        - torque limits
        - pos/vel limits
    '''
    
    def __init__(self, conf, q0, v0):
        self.conf = conf
        self.q0 = q0
        self.v0 = v0
        conf.urdf, conf.path = loadUR_urdf()
        self.robot = tsid.RobotWrapper(conf.urdf, [conf.path], False)
        robot = self.robot
        self.model = model = robot.model()
        q = self.q0
        v = self.v0
        
        formulation = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
        formulation.computeProblemData(0.0, q, v)
                
        postureTask = tsid.TaskJointPosture("task-posture", robot)
        postureTask.setKp(conf.kp_posture * np.ones(robot.nv).T)
        postureTask.setKd(2.0 * np.sqrt(conf.kp_posture) * np.ones(robot.nv).T)
        formulation.addMotionTask(postureTask, conf.w_posture, 1, 0.0)
        
        self.tau_max = conf.tau_max_scaling*model.effortLimit
        self.tau_min = -self.tau_max
        actuationBoundsTask = tsid.TaskActuationBounds("task-actuation-bounds", robot)
        actuationBoundsTask.setBounds(self.tau_min, self.tau_max)
        if(conf.w_torque_bounds>0.0):
            formulation.addActuationTask(actuationBoundsTask, conf.w_torque_bounds, 0, 0.0)
            
        jointBoundsTask = tsid.TaskJointBounds("task-joint-bounds", robot, conf.dt)
        self.v_max = conf.v_max_scaling * model.velocityLimit
        self.v_min = -self.v_max
        self.q_max = conf.q_max_scaling * model.upperPositionLimit
        self.q_min = conf.q_max_scaling * model.lowerPositionLimit
        #jointBoundsTask.setPositionBounds(self.q_min, self.q_max) #it doesn't exist -> to be implemented
        jointBoundsTask.setVelocityBounds(self.v_min, self.v_max)
        if(conf.w_joint_bounds>0.0):
            formulation.addMotionTask(jointBoundsTask, conf.w_joint_bounds, 0, 0.0)
        
        trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q)
        postureTask.setReference(trajPosture.computeNext())
        
        solver = tsid.SolverHQuadProgFast("qp solver")
        solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)
        
        self.trajPosture = trajPosture
        self.postureTask  = postureTask
        self.actuationBoundsTask = actuationBoundsTask
        self.jointBoundsTask = jointBoundsTask
        self.formulation = formulation
        self.solver = solver
        self.q = q
        self.v = v
        self.q_res  = np.empty((self.robot.nq, self.conf.N_SIMULATION+1))*nan
        self.v_res  = np.empty((self.robot.nv, self.conf.N_SIMULATION+1))*nan
        
    def integrate_dv(self, q, v, dv, dt):
        v_mean = v + 0.5*dt*dv
        v += dt*dv
        q = se3.integrate(self.model, q, dt*v_mean)
        return q,v
    
    def compute_problem(self, q0, v0):
        N = self.conf.N_SIMULATION
        tau    = np.empty((self.robot.na, N))*nan
        samplePosture = self.trajPosture.computeNext()

        # desired configuration
        pXX =  np.array([-0.3223527113543909, -0.7805794638446351, -2.5675506591796875, -1.6347843609251917, -1.5715253988849085, -1.0017417112933558])
        vXX = np.zeros(6)
        aXX = np.zeros(6)

        t = 0.0
        self.q_res[:,0], self.v_res[:,0] = q0, v0
        
        for i in range(0, N):
            time_start = time.time()

            samplePosture.value(pXX)
            samplePosture.derivative(vXX)
            samplePosture.second_derivative(aXX)
            self.postureTask.setReference(samplePosture)

            HQPData = self.formulation.computeProblemData(t, self.q_res[:,i], self.v_res[:,i])
            # if i == 0: HQPData.print_all()

            sol = self.solver.solve(HQPData)
            if(sol.status!=0):
                print(("Time %.3f QP problem could not be solved! Error code:"%t, sol.status))
                break
            
            tau[:,i] = self.formulation.getActuatorForces(sol)
            dv = self.formulation.getAccelerations(sol)

            self.q_res[:,i+1], self.v_res[:,i+1] = self.integrate_dv(self.q_res[:,i], self.v_res[:,i], dv, self.conf.dt)
            t += self.conf.dt
            
            pXX = self.q_res[:,i+1]
            
            time_spent = time.time() - time_start
            
            if i%self.conf.DISPLAY_N == 0 and self.viewer: 
                self.robot_display.display(self.q_res[:,i])
            
            if(time_spent < self.conf.dt) and self.viewer: 
                time.sleep(self.conf.dt-time_spent)
            
        return tau[:,i]
        
        def __del__ (self):
            del self.conf
            del self.q0
            del self.v0 
            del self.robot
            del self.model
            del self.eeTask
            del self.EE 
            del self.trajEE
            del self.tau_max 
            del self.tau_min 
            del self.v_max 
            del self.v_min
            del self.trajPosture
            del self.postureTask
            del self.actuationBoundsTask
            del self.jointBoundsTask
            del self.formulation
            del self.solver
            del self.q
            del self.v
            if(viewer):
                del self.robot_display
                del self.gui 
            del self.viewer 

