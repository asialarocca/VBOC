# -*- coding: utf-8 -*-
"""
Created on 3 May  2022

@author: mfocchi
"""

from __future__ import print_function
import tf
import numpy as np
from double_pendulum_ocp_class_mpc_reduced import OCPdoublependulumNN, OCPdoublependulumINIT
from my_nn import NeuralNet
import torch
import torch.nn as nn
from base_controllers.utils.kin_dyn_utils import (
    fifthOrderPolynomialTrajectory as coeffTraj,
)
from admittance_controller import AdmittanceControl
from base_controllers.base_controller_fixed import BaseControllerFixed
from obstacle_avoidance.obstacle_avoidance import ObstacleAvoidance
import actionlib
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from std_msgs.msg import Float64MultiArray
from controller_manager_msgs.srv import LoadControllerRequest, LoadController
from controller_manager_msgs.srv import SwitchControllerRequest, SwitchController
import L8_conf as lab_conf
import params as conf
from utils.common_functions import plotJoint, plotAdmittanceTracking, plotEndeff
from plotjoints import plotPos
import matplotlib.pyplot as plt
from termcolor import colored
from six.moves import (
    input,
)  # solves compatibility issue bw pyuthon 2.x and 3 for raw input that does exists in python 3

import os
import rospy as ros
import sys

# messages for topic subscribers
from docutils.nodes import label
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Wrench, Point
from std_srvs.srv import Trigger

# ros utils
import roslaunch
import rosnode
import rosgraph
import rospkg
from rospy import Time
import time

# other utils
from utils.math_tools import *
from numpy import nan
import pinocchio as pin

np.set_printoptions(threshold=np.inf, precision=5, linewidth=1000, suppress=True)


robotName = "ur5"

# controller manager management


class LabAdmittanceController(BaseControllerFixed):
    def __init__(self, robot_name="ur5"):
        super().__init__(robot_name=robot_name)
        self.real_robot = conf.robot_params[self.robot_name]["real_robot"]
        # self.homing_flag = self.real_robot
        self.homing_flag = True
        if conf.robot_params[self.robot_name]["control_type"] == "torque":
            self.use_torque_control = 1
        else:
            self.use_torque_control = 0

        if lab_conf.obstacle_avoidance:
            self.world_name = "tavolo_obstacles.world"
            if not self.use_torque_control:
                print(
                    colored(
                        "ERRORS: you can use obstacle avoidance only on torque control mode",
                        "red",
                    )
                )
                sys.exit()
        else:
            self.world_name = None

        if lab_conf.admittance_control and (
            (not self.real_robot) and (not self.use_torque_control)
        ):
            print(
                colored(
                    "ERRORS: you can use admittance control only on torque control mode or in real robot (need contact force estimation or measurement)",
                    "red",
                )
            )
            sys.exit()

        if self.use_torque_control and self.real_robot:
            print(
                colored(
                    "ERRORS: unfortunately...you cannot use ur5 in torque control mode, talk with your course coordinator to buy a better robot...:))",
                    "red",
                )
            )
            sys.exit()

        print(
            "Initialized L8 admittance  controller---------------------------------------------------------------"
        )

    def startRealRobot(self):
        os.system("killall rviz gzserver gzclient")
        print(
            colored(
                "------------------------------------------------ROBOT IS REAL!", "blue"
            )
        )
        if (not rosgraph.is_master_online()) or (
            "/" + self.robot_name + "/ur_hardware_interface"
            not in rosnode.get_node_names()
        ):
            print(colored("Error: you need to launch the ur driver!", "red"))
            sys.exit()
        else:
            package = "rviz"
            executable = "rviz"
            args = (
                "-d "
                + rospkg.RosPack().get_path("ros_impedance_controller")
                + "/config/operator.rviz"
            )
            node = roslaunch.core.Node(package, executable, args=args)
            launch = roslaunch.scriptapi.ROSLaunch()
            launch.start()
            process = launch.launch(node)

    def loadModelAndPublishers(self, xacro_path):
        print("loadModelAndPublishers inizio")
        super().loadModelAndPublishers(xacro_path)
        print("loadModelAndPublishers 2")
        self.sub_ftsensor = ros.Subscriber(
            "/" + self.robot_name + "/wrench",
            WrenchStamped,
            callback=self._receive_ftsensor,
            queue_size=1,
            tcp_nodelay=True,
        )
        print("loadModelAndPublishers 3")
        self.switch_controller_srv = ros.ServiceProxy(
            "/" + self.robot_name + "/controller_manager/switch_controller",
            SwitchController,
        )
        print("loadModelAndPublishers 4")
        self.load_controller_srv = ros.ServiceProxy(
            "/" + self.robot_name + "/controller_manager/load_controller",
            LoadController,
        )
        print("loadModelAndPublishers 5")
        # specific publisher for joint_group_pos_controller that publishes only position
        self.pub_reduced_des_jstate = ros.Publisher(
            "/" + self.robot_name + "/joint_group_pos_controller/command",
            Float64MultiArray,
            queue_size=10,
        )
        print("loadModelAndPublishers 6")

        self.zero_sensor = ros.ServiceProxy(
            "/" + self.robot_name + "/ur_hardware_interface/zero_ftsensor", Trigger
        )
        print("loadModelAndPublishers 7")

        #  different controllers are available from the real robot and in simulation
        if self.real_robot:
            self.available_controllers = [
                "joint_group_pos_controller",
                "scaled_pos_joint_traj_controller",
            ]
        else:
            self.available_controllers = [
                "joint_group_pos_controller",
                "pos_joint_traj_controller",
            ]
        print("loadModelAndPublishers 8")
        self.active_controller = self.available_controllers[0]

        self.broadcaster = tf.TransformBroadcaster()

    def applyForce(self):
        wrench = Wrench()
        wrench.force.x = 0
        wrench.force.y = 0
        wrench.force.z = 30
        wrench.torque.x = 0
        wrench.torque.y = 0
        wrench.torque.z = 0
        # you can apply forces only in this frame because this service is buggy, it will ignore any other frame
        reference_frame = "world"
        reference_point = Point(x=0, y=0, z=0)
        try:
            self.apply_body_wrench(
                body_name="ur5::wrist_3_link",
                reference_frame=reference_frame,
                reference_point=reference_point,
                wrench=wrench,
                duration=ros.Duration(10),
            )
        except:
            pass

    def _receive_ftsensor(self, msg):
        contactForceTool0 = np.zeros(3)
        contactMomentTool0 = np.zeros(3)
        contactForceTool0[0] = msg.wrench.force.x
        contactForceTool0[1] = msg.wrench.force.y
        contactForceTool0[2] = msg.wrench.force.z
        contactMomentTool0[0] = msg.wrench.torque.x
        contactMomentTool0[1] = msg.wrench.torque.y
        contactMomentTool0[2] = msg.wrench.torque.z
        self.contactForceW = self.w_R_tool0.dot(contactForceTool0)
        self.contactMomentW = self.w_R_tool0.dot(contactMomentTool0)

    def deregister_node(self):
        print("deregistering nodes")
        self.ros_pub.deregister_node()
        if not self.real_robot:
            os.system(" rosnode kill /" + self.robot_name + "/ros_impedance_controller")
            os.system(" rosnode kill /gzserver /gzclient")

    def updateKinematicsDynamics(self):
        # q is continuously updated
        # to compute in the base frame  you should put neutral base
        self.robot.computeAllTerms(self.q, self.qd)
        # joint space inertia matrix
        self.M = self.robot.mass(self.q)
        # bias terms
        self.h = self.robot.nle(self.q, self.qd)
        # gravity terms
        self.g = self.robot.gravity(self.q)
        # compute ee position  in the world frame
        frame_name = conf.robot_params[self.robot_name]["ee_frame"]
        # this is expressed in a workdframe with the origin attached to the base frame origin
        self.x_ee = self.robot.framePlacement(
            self.q, self.robot.model.getFrameId(frame_name)
        ).translation
        self.w_R_tool0 = self.robot.framePlacement(
            self.q, self.robot.model.getFrameId(frame_name)
        ).rotation
        # compute jacobian of the end effector in the world frame
        self.J6 = self.robot.frameJacobian(
            self.q,
            self.robot.model.getFrameId(frame_name),
            False,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        # take first 3 rows of J6 cause we have a point contact
        self.J = self.J6[:3, :]
        # compute contact forces
        self.estimateContactForces()
        # broadcast base world TF
        self.broadcaster.sendTransform(
            self.base_offset, (0.0, 0.0, 0.0, 1.0), Time.now(), "/base_link", "/world"
        )

    def estimateContactForces(self):
        # estimate ground reaction forces from torques tau
        if self.use_torque_control:
            self.contactForceW = np.linalg.inv(self.J6.T).dot(self.h - self.tau)[:3]

    def startupProcedure(self):
        if self.use_torque_control:
            # set joint pdi gains
            self.pid.setPDjoints(
                conf.robot_params[self.robot_name]["kp"],
                conf.robot_params[self.robot_name]["kd"],
                np.zeros(self.robot.na),
            )
            # only torque loop
            # self.pid.setPDs(0.0, 0.0, 0.0)
        if self.real_robot:
            self.zero_sensor()
        print(colored("finished startup -- starting controller", "red"))

    def initVars(self):
        super().initVars()

        # log variables relative to admittance controller
        self.q_des_adm_log = (
            np.empty((self.robot.na, conf.robot_params[self.robot_name]["buffer_size"]))
            * nan
        )
        self.x_ee_des_adm_log = (
            np.empty((3, conf.robot_params[self.robot_name]["buffer_size"])) * nan
        )
        self.EXTERNAL_FORCE = False
        self.payload_weight_avg = 0.0
        self.polynomial_flag = False
        self.obs_avoidance = ObstacleAvoidance()
        # position of the center of the objects is in WF
        self.obs_avoidance.setCubeParameters(0.25, np.array([0.125, 0.75, 0.975]))
        self.obs_avoidance.setCylinderParameters(0.125, 0.3, np.array([0.6, 0.25, 1.0]))
        self.admit = AdmittanceControl(
            self.ikin, lab_conf.Kx, lab_conf.Dx, conf.robot_params[self.robot_name]
        )

        if lab_conf.USER_TRAJECTORY:
            data = np.load('ur5_q_ref' + '.npz')
            self.q_ref = data['q']
            self.traj_duration = self.q_ref.shape[0]

    def logData(self):
        if conf.robot_params[self.robot_name]["control_type"] == "admittance":
            self.q_des_adm_log[:, self.log_counter] = self.q_des_adm
            self.x_ee_des_adm_log[:, self.log_counter] = self.x_ee_des_adm
        # I neeed to do after because it updates log counter
        super().logData()

    def switch_controller(self, target_controller):
        """Activates the desired controller and stops all others from the predefined list above"""
        print("Available controllers: ", self.available_controllers)
        print("Controller manager: loading ", target_controller)

        other_controllers = self.available_controllers
        other_controllers.remove(target_controller)
        print("Controller manager:Switching off  :  ", other_controllers)

        srv = LoadControllerRequest()
        srv.name = target_controller

        self.load_controller_srv(srv)

        srv = SwitchControllerRequest()
        srv.stop_controllers = other_controllers
        srv.start_controllers = [target_controller]
        srv.strictness = SwitchControllerRequest.BEST_EFFORT
        self.switch_controller_srv(srv)
        self.active_controller = target_controller

    def send_reduced_des_jstate(self, q_des):
        msg = Float64MultiArray()
        msg.data = q_des
        self.pub_reduced_des_jstate.publish(msg)

    def send_joint_trajectory(self):

        # Creates a trajectory and sends it using the selected action server
        trajectory_client = actionlib.SimpleActionClient(
            "{}/follow_joint_trajectory".format(
                "/" + self.robot_name + "/" + self.active_controller
            ),
            FollowJointTrajectoryAction,
        )
        # Create and fill trajectory goal
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names

        # The following list are arbitrary positions
        # Change to your own needs if desired q0 [ 0.5, -0.7, 1.0, -1.57, -1.57, 0.5]), #limits([0,pi],   [0, -pi], [-pi/2,pi/2],)
        print(colored("JOINTS ARE: ", "blue"), self.q.transpose())
        # position_list = [[0.5, -0.7, 1.0, -1.57, -1.57, 0.5]]  # limits([0,-pi], [-pi/2,pi/2],  [0, -pi])
        # position_list.append([0.5, -0.7 - 0.2, 1.0 - 0.1, -1.57, -1.57, 0.5])
        # position_list.append([0.5 + 0.5, -0.7 - 0.3, 1.0 - 0.1, -1.57, -1.57, 0.5])
        # position_list.append([0.5 + 0.5, -0.7 - 0.3, 1.0 , -1., -1.57, 0.5])

        self.q0 = conf.robot_params[p.robot_name]["q_0"]
        dq1 = np.array([0.2, 0, 0, 0, 0, 0])
        dq2 = np.array([0.2, -0.2, 0, 0, 0, 0])
        dq3 = np.array([0.2, -0.2, 0.4, 0, 0, 0])
        position_list = [self.q0]  # limits([0,-pi], [-pi/2,pi/2],  [0, -pi])
        position_list.append(self.q0 + dq1)
        position_list.append(self.q0 + dq2)
        position_list.append(self.q0 + dq3)
        print(colored("List of targets for joints: ", "blue"))
        print(position_list[0])
        print(position_list[1])
        print(position_list[2])
        print(position_list[3])

        duration_list = [5.0, 10.0, 20.0, 30.0]
        for i, position in enumerate(position_list):
            point = JointTrajectoryPoint()
            point.positions = position
            point.time_from_start = ros.Duration(duration_list[i])
            goal.trajectory.points.append(point)

        self.ask_confirmation(position_list)
        print("Executing trajectory using the {}".format("pos_joint_traj_controller"))
        trajectory_client.send_goal(goal)
        trajectory_client.wait_for_result()

        result = trajectory_client.get_result()
        print("Trajectory execution finished in state {}".format(result.error_code))

    def ask_confirmation(self, waypoint_list):
        """Ask the user for confirmation. This function is obviously not necessary, but makes sense
        in a testing script when you know nothing about the user's setup."""
        ros.logwarn(
            "The robot will move to the following waypoints: \n{}".format(waypoint_list)
        )
        confirmed = False
        valid = False
        while not valid:
            input_str = input(
                "Please confirm that the robot path is clear of obstacles.\n"
                "Keep the EM-Stop available at all times. You are executing\n"
                "the motion at your own risk. Please type 'y' to proceed or 'n' to abort: "
            )
            valid = input_str in ["y", "n"]
            if not valid:
                ros.loginfo("Please confirm by entering 'y' or abort by entering 'n'")
            else:
                if input_str == "y":
                    confirmed = True
        if not confirmed:
            ros.loginfo("Exiting as requested by user.")
            sys.exit(0)

    def deregister_node(self):
        super().deregister_node()
        if not self.real_robot:
            os.system(" rosnode kill /" + self.robot_name + "/ros_impedance_controller")
            os.system(" rosnode kill /gzserver /gzclient")

    def plotStuff(self):
        if not (conf.robot_params[p.robot_name]["control_mode"] == "trajectory"):
            if lab_conf.admittance_control:
                plotJoint(
                    "position",
                    0,
                    self.time_log,
                    self.q_log,
                    self.q_des_log,
                    self.qd_log,
                    self.qd_des_log,
                    None,
                    None,
                    self.tau_log,
                    self.tau_ffwd_log,
                    self.joint_names,
                    self.q_des_adm_log,
                )
                plotAdmittanceTracking(
                    3,
                    self.time_log,
                    self.x_ee_log,
                    self.x_ee_des_log,
                    self.x_ee_des_adm_log,
                    self.contactForceW_log,
                )
            else:
                plotJoint('position', 0, self.time_log, self.q_log, self.q_des_log, self.qd_log, self.qd_des_log, None, None, self.tau_log,
                          None, self.joint_names, None)
            plotJoint('torque', 1, self.time_log, self.q_log, self.q_des_log, self.qd_log, self.qd_des_log, None, None, self.tau_log,
                      None, self.joint_names, None)
            #plotEndeff('force', 1, p.time_log, p.contactForceW_log)
            plt.show(block=True)


def talker(p):
    # Ocp initialization:
    model = torch.load('model_save_3')
    with open("mean_3.txt", "r") as f:
        mean = float(f.readlines()[0])
    with open("std_3.txt", "r") as f:
        std = float(f.readlines()[0])
    mpc = OCPdoublependulumNN(model, mean, std)  # double pendulum MPC
    #mpc = OCPdoublependulumINIT()

    p.start()
    if p.real_robot:
        p.startRealRobot()
    else:
        p.startSimulator(p.world_name, p.use_torque_control)

    # specify xacro location
    xacro_path = (
        rospkg.RosPack().get_path("ur_description") + "/urdf/" + p.robot_name + ".xacro"
    )
    p.loadModelAndPublishers(xacro_path)
    p.initVars()
    p.startupProcedure()

    p.q_des_q0 = conf.robot_params[p.robot_name]['q_0']
    p.q_des = np.copy(p.q_des_q0)
    p.qd_des = np.zeros((6,))
    p.admit.setPosturalTask(np.copy(p.q_des_q0))

    # loop frequency
    rate = ros.Rate(1/conf.robot_params[p.robot_name]['dt'])

    if conf.robot_params[p.robot_name]["control_mode"] == "trajectory":
        # to test the trajectory
        if p.real_robot:
            p.switch_controller("scaled_pos_joint_traj_controller")
        else:
            p.switch_controller("pos_joint_traj_controller")
        p.send_joint_trajectory()
    else:

        if not p.use_torque_control:
            p.switch_controller("joint_group_pos_controller")
        # reset to actual
        p.updateKinematicsDynamics()
        p.time_poly = None

        ext_traj_counter = 0

        # control loop
        iteration = 0
        it = 0

        while True:
            # homing procedure
            if p.homing_flag:
                print(colored("STARTING HOMING PROCEDURE", "red"))
                while True:
                    joint_error = np.linalg.norm(
                        p.q - conf.robot_params[p.robot_name]["q_0"]
                    )
                    p.q_des = conf.robot_params[p.robot_name]["q_0"]
                    p.qd_des = np.zeros((6,))
                    p.tau_ffwd = conf.robot_params[p.robot_name]["kp"] * (np.subtract(
                        p.q_des, p.q)) + conf.robot_params[p.robot_name]["kd"] * (np.subtract(p.qd_des, p.qd))
                    p.send_des_jstate(p.q_des, p.qd_des, p.tau_ffwd)
                    rate.sleep()

                    if p.use_torque_control:
                        err_min = 0.1
                    else:
                        err_min = 0.001

                    if joint_error <= err_min:
                        p.homing_flag = False
                        print(colored("HOMING PROCEDURE ACCOMPLISHED", "red"))
                        homing_done = p.time
                        current_pos = p.q
                        break

                yref = mpc.ocp.cost.yref[:mpc.nx]
                uref = np.array([10.0688, -12.1775])

                x_guess = np.array([[-0.78058, -2.56755,  0.,       0., ],
                                    [-0.78058, -2.56469, -0.00001,  0.57132],
                                    [-0.78057, -2.55612, -0.00001,  1.14287],
                                    [-0.78056, -2.54184, -0.,  1.71483],
                                    [-0.78055, -2.52189, -0.00001,  2.27516],
                                    [-0.78055, -2.49906, -0.00001,  2.29023],
                                    [-0.78055, -2.47651, -0.00001,  2.22104],
                                    [-0.78055, -2.45464, -0.00001,  2.15232],
                                    [-0.78055, -2.43346, -0.00001,  2.08406],
                                    [-0.78055, -2.41296, -0.00001,  2.01617],
                                    [-0.78056, -2.39313, -0.,  1.94856]])
                u_guess = np.array([[14.01156,  19.99269],
                                    [14.35993,  19.98784],
                                    [15.0738,  19.9776],
                                    [16.10986,  19.20529],
                                    [13.51133, -14.69311],
                                    [12.86046, -19.97341],
                                    [12.68813, -19.98708],
                                    [12.52072, -19.99199],
                                    [12.35759, -19.99434],
                                    [12.19901, -19.99565]])

                # Plots:
                h = 0.01
                xx2, yy2 = np.meshgrid(np.arange(mpc.xmin[1], mpc.xmax[1], h),
                                       np.arange(mpc.xmin[3], mpc.xmax[3], h))
                xrav2 = xx2.ravel()
                yrav2 = yy2.ravel()

            # update the kinematics
            p.updateKinematicsDynamics()

            if lab_conf.USER_TRAJECTORY and (p.time > 6.0) and (ext_traj_counter < p.traj_duration):
                p.q_des = p.q_ref[ext_traj_counter, :]
                ext_traj_counter += 1

            # Double pendulum MPC:
            if iteration % 10 == 0:

                #print('p_current', p.q[1:3], p.qd[1:3])

                current_pos = p.q
                current_vel = p.qd

                #initial_time = time.time()

                status = mpc.compute_problem(current_pos[1:3], current_vel[1:3], x_guess, u_guess)

                #print(time.time() - initial_time)
                # mpc.ocp_solver.print_statistics()

                # if times >= 0.01:
                #    mpc.ocp_solver.print_statistics()

                if status == 1:
                    for i in range(1, mpc.N+1):
                        x_guess[i-1, :] = mpc.ocp_solver.get(i, "x")
                    x_guess[mpc.N, :] = yref

                    print('The controller could find a feasible solution')

                    current_val = mpc.ocp_solver.get(1, "x")

                    p.q_des[1:3] = current_val[:2]
                    p.qd_des[1:3] = current_val[2:]

                    for i in range(1, mpc.N):
                        u_guess[i-1, :] = mpc.ocp_solver.get(i, "u")

                    p.tau_ffwd = conf.robot_params[p.robot_name]["kp"] * (np.subtract(
                        p.q_des, p.q)) + conf.robot_params[p.robot_name]["kd"] * (np.subtract(p.qd_des, p.qd))

                    p.tau_ffwd[1:3] = mpc.ocp_solver.get(1, "u")

                    it = 0

                else:
                    #current_val = x_guess[1, :]
                    print('The controller could not find a feasible solution')
                    # put here the mpc without terminal constraint and above the one with terminal constraint

                    it = it + 1

                    if it == 10:
                        p.time = p.time + conf.robot_params[p.robot_name]["dt"]
                        # stops the while loop if  you prematurely hit CTRL+C
                        plotPos(mpc, 'position', 0, p.time_log, p.q_log, p.q_des_log, p.qd_log, p.qd_des_log, None, None, p.tau_log,
                                None, p.joint_names, None)

                        with torch.no_grad():
                            # Plot the results:
                            plt.figure()
                            inp = torch.from_numpy(
                                np.float32(
                                    np.c_[
                                        -0.7805794638446351 * np.ones(xrav2.shape[0]),
                                        xrav2,
                                        np.zeros(yrav2.shape[0]),
                                        yrav2,
                                    ]
                                )
                            )
                            inp = (inp - mean) / std
                            out = model(inp)
                            y_pred = np.argmax(out.numpy(), axis=1)
                            Z = y_pred.reshape(xx2.shape)
                            plt.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.8)
                            plt.plot(
                                p.q_des_log[2],
                                p.qd_des_log[2],
                                '-o'
                            )
                            plt.xlim([mpc.xmin[1], mpc.xmax[1]])
                            plt.ylim([mpc.xmin[3], mpc.xmax[3]])
                            plt.grid()
                            plt.title("Elbow joint")

                            plt.show(block=True)
                        break

                    current_val = x_guess[it, :]

                    p.q_des[1:3] = current_val[:2]
                    p.qd_des[1:3] = current_val[2:]

                    p.tau_ffwd[1:3] = u_guess[it, :]

            iteration = iteration + 1

            # # controller with gravity coriolis comp
            # p.tau_ffwd = p.h + np.zeros(p.robot.na)

            # only torque loop (not used)
            # p.tau_ffwd = conf.robot_params[p.robot_name]['kp']*(np.subtract(p.q_des,   p.q))  - conf.robot_params[p.robot_name]['kd']*p.qd

            # send commands to gazebo
            if (p.use_torque_control):
                if (lab_conf.obstacle_avoidance):
                    p.tau_ffwd = p.obs_avoidance.computeTorques(p,  lab_conf.des_ee_goal)
                p.send_des_jstate(p.q_des, p.qd_des, p.tau_ffwd)
            else:
                p.send_reduced_des_jstate(p.q_des)

            # log variables
            if p.time > homing_done:
                p.logData()

            # plot end-effector
            p.ros_pub.add_marker(p.x_ee + p.base_offset)
            p.ros_pub.publishVisual()

            # wait for synconization of the control loop
            rate.sleep()

            p.time = p.time + conf.robot_params[p.robot_name]["dt"]
            # stops the while loop if  you prematurely hit CTRL+C
            if ros.is_shutdown() or p.log_counter >= conf.robot_params[p.robot_name]['buffer_size']:
                plotPos(mpc, 'position', 0, p.time_log, p.q_log, p.q_des_log, p.qd_log, p.qd_des_log, None, None, p.tau_log,
                        None, p.joint_names, None)

                with torch.no_grad():
                    # Plot the results:
                    plt.figure()
                    inp = torch.from_numpy(
                        np.float32(
                            np.c_[
                                -0.7805794638446351 * np.ones(xrav2.shape[0]),
                                xrav2,
                                np.zeros(yrav2.shape[0]),
                                yrav2,
                            ]
                        )
                    )
                    inp = (inp - mean) / std
                    out = model(inp)
                    y_pred = np.argmax(out.numpy(), axis=1)
                    Z = y_pred.reshape(xx2.shape)
                    plt.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.8)
                    plt.plot(
                        p.q_des_log[2],
                        p.qd_des_log[2],
                        '-o'
                    )
                    plt.xlim([mpc.xmin[1], mpc.xmax[1]])
                    plt.ylim([mpc.xmin[3], mpc.xmax[3]])
                    plt.grid()
                    plt.title("Elbow joint")

                    plt.show(block=True)

                break

    print("Shutting Down")
    ros.signal_shutdown("killed")
    p.deregister_node()


if __name__ == "__main__":

    p = LabAdmittanceController(robotName)

    try:
        talker(p)
    except ros.ROSInterruptException:
        # these plots are for simulated robot
        # p.plotStuff()
        pass