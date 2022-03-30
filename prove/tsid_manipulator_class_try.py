import pinocchio as se3
import tsid
import numpy as np
from numpy import nan
import os
import gepetto.corbaserver
import time
import subprocess
from arc.utils.robot_loaders import loadUR_urdf
from numpy.linalg import norm as norm


class TsidManipulator:
    ''' Standard TSID formulation for a robot manipulator
        - end-effector task
        - Postural task
        - torque limits
        - pos/vel limits
    '''
    
    def __init__(self, conf, q0, v0, viewer=False):
        self.conf = conf
        self.q0 = q0
        self.v0 = v0
        conf.urdf, conf.path = loadUR_urdf()
        self.robot = tsid.RobotWrapper(conf.urdf, [conf.path], False)
        robot = self.robot
        self.model = model = robot.model()
        self.viewer = viewer
        try:
            se3.loadReferenceConfigurations(model, conf.srdf, False)
            q = model.referenceConfigurations['default']
        except:
            q = self.q0
        v = self.v0
        
        assert model.existFrame(conf.ee_frame_name)
        
        formulation = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
        formulation.computeProblemData(0.0, q, v)
                
        postureTask = tsid.TaskJointPosture("task-posture", robot)
        postureTask.setKp(conf.kp_posture * np.ones(robot.nv).T)
        postureTask.setKd(2.0 * np.sqrt(conf.kp_posture) * np.ones(robot.nv).T)
        formulation.addMotionTask(postureTask, conf.w_posture, 1, 0.0)
        
        self.eeTask = tsid.TaskSE3Equality("task-ee", self.robot, self.conf.ee_frame_name)
        self.eeTask.setKp(self.conf.kp_ee * np.ones(6))
        self.eeTask.setKd(2.0 * np.sqrt(self.conf.kp_ee) * np.ones(6))
        self.eeTask.setMask(conf.ee_task_mask)
        self.eeTask.useLocalFrame(False)
        self.EE = model.getFrameId(conf.ee_frame_name)
        H_ee_ref = self.robot.framePosition(formulation.data(), self.EE)
        self.trajEE = tsid.TrajectorySE3Constant("traj-ee", H_ee_ref)
        formulation.addMotionTask(self.eeTask, conf.w_ee, 1, 0.0)
        
        self.tau_max = conf.tau_max_scaling*model.effortLimit
        self.tau_min = -self.tau_max
        actuationBoundsTask = tsid.TaskActuationBounds("task-actuation-bounds", robot)
        actuationBoundsTask.setBounds(self.tau_min, self.tau_max)
        if(conf.w_torque_bounds>0.0):
            formulation.addActuationTask(actuationBoundsTask, conf.w_torque_bounds, 0, 0.0)
            
        jointBoundsTask = tsid.TaskJointBounds("task-joint-bounds", robot, conf.dt)
        self.v_max = conf.v_max_scaling * model.velocityLimit
        self.v_min = -self.v_max
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
                
        # for gepetto viewer
        if(viewer):
            self.robot_display = se3.RobotWrapper.BuildFromURDF(conf.urdf, [conf.path, ])
            l = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
            if int(l[1]) == 0:
                os.system('gepetto-gui &')
            time.sleep(1)
            gepetto.corbaserver.Client()
            self.robot_display.initViewer(loadModel=True)
            self.robot_display.displayCollisions(False)
            self.robot_display.displayVisuals(True)
            self.robot_display.display(q)
            self.gui = self.robot_display.viewer.gui
            self.gui.setCameraTransform('python-pinocchio', conf.CAMERA_TRANSFORM)
        
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
        pXX = q0
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
            
        if norm(self.v_res[:,N]) < 0.001:
            return 1
        else:
            return 0
        

