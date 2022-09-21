# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:07:44 2020

@author: mfocchi
"""
import os
import psutil
#from pinocchio.visualize import GepettoVisualizer
from base_controllers.utils.custom_robot_wrapper import RobotWrapper
import numpy as np
import matplotlib.pyplot as plt
import sys
from termcolor import colored
import rospkg
import rospy as ros
import rosnode
import roslaunch
import rosgraph
from roslaunch.parent import ROSLaunchParent
import copy

#from urdf_parser_py.urdf import URDF
#make plot interactive
plt.ion()
plt.close()                 

def plotPos(mpc,name, figure_id, time_log, q_log=None, q_des_log=None, qd_log=None, qd_des_log=None, qdd_log=None, qdd_des_log=None, tau_log=None, tau_ffwd_log = None, joint_names = None, q_adm = None):
    plot_var_des_log = None
    if name == 'position':
        plot_var_log = q_log
        if   (q_des_log is not None):
            plot_var_des_log = q_des_log
        else:
            plot_var_des_log = None
    elif name == 'velocity':
        plot_var_log = qd_log
        if   (qd_des_log is not None):
            plot_var_des_log  = qd_des_log
        else:
            plot_var_des_log = None
    elif name == 'acceleration':
        plot_var_log = qdd_log
        if   (qdd_des_log is not None):
            plot_var_des_log  = qdd_des_log
        else:
            plot_var_des_log = None
    elif name == 'torque':
        plot_var_log = tau_log
        if   (tau_ffwd_log is not None):                                    
            plot_var_des_log  = tau_ffwd_log 
        else:
          plot_var_des_log = None                                                
    else:
       print(colored("plotJopnt error: wrong input string", "red") )
       return                                   

    njoints = min(plot_var_log.shape)                                                                

    #neet to transpose the matrix other wise it cannot be plot with numpy array    
    fig = plt.figure(figure_id)                
    fig.suptitle(name, fontsize=20)

    labels = ["1 - Shoulder Pan", "2 - Shoulder Lift", "3 - Elbow", "4 - Wrist 1", "5 - Wrist 2", "6 - Wrist 3"]
    
    xmax = np.array([6.14, mpc.xmax[0], mpc.xmax[1], 6.28, 6.28, 6.28, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14])
    xmin = np.array([-6.14, mpc.xmin[0], mpc.xmin[1], -6.28, -6.28, -6.28, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14])
    
    lw_des=7
    lw_act=4   
    marker_size= 0   


    for jidx in range(njoints):
        plt.subplot(njoints / 3, 3, jidx + 1)
        plt.ylabel(labels[jidx])
        
        plt.hlines(xmax[jidx], time_log[0], time_log[-1], linestyles='dashed', alpha=0.7, color = 'black')
        plt.hlines(xmin[jidx], time_log[0], time_log[-1], linestyles='dashed', alpha=0.7, color = 'black')
        
        #if jidx == 1 or jidx == 2:
            #plt.ylim([1.2*xmin[jidx], xmax[jidx]/1.2])
        #else:
            #plt.ylim([1.2*xmin[jidx], 1.2*xmax[jidx]])
        
        plt.plot(time_log, plot_var_des_log[jidx,:], linestyle='-', marker="o",markersize=marker_size, lw=lw_des,color = 'red')
        plt.plot(time_log, plot_var_log[jidx,:],linestyle='-',marker="o",markersize=marker_size, lw=lw_act,color = 'blue')
        plt.grid()
        
    xref = mpc.ocp.cost.yref
        
    fig = plt.figure(figure_id+1)    
    fig.suptitle('Focus on Shoulder Lift joint', fontsize=20)

    plt.ylabel(labels[1])
    
    plt.hlines(xref[0], time_log[0], time_log[-1], linestyles='dashed', alpha=0.7, color = 'green')
        
    plt.hlines(xmax[1], time_log[0], time_log[-1], linestyles='dashed', alpha=0.7)
    plt.hlines(xmin[1], time_log[0], time_log[-1], linestyles='dashed', alpha=0.7, color = 'black')
        
    #plt.ylim([1.2*xmin[1], xmax[1]/1.2])
        
    plt.plot(time_log, plot_var_des_log[1,:], linestyle='-', marker="o",markersize=marker_size, lw=lw_des,color = 'red')
    plt.plot(time_log, plot_var_log[1,:],linestyle='-',marker="o",markersize=marker_size, lw=lw_act,color = 'blue')
    plt.grid()
    
    fig = plt.figure(figure_id+2)    
    fig.suptitle('Focus on Elbow joint', fontsize=20)

    plt.ylabel(labels[2])
    
    plt.hlines(xref[1], time_log[0], time_log[-1], linestyles='dashed', alpha=0.7, color = 'green')
        
    plt.hlines(xmax[2], time_log[0], time_log[-1], linestyles='dashed', alpha=0.7, color = 'black')
    #plt.hlines(xmin[2], time_log[0], time_log[-1], linestyles='dashed', alpha=0.7)
        
    #plt.ylim([1.2*xmin[2], xmax[2]/1.2])
        
    plt.plot(time_log, plot_var_des_log[2,:], linestyle='-', marker="o",markersize=marker_size, lw=lw_des,color = 'red')
    plt.plot(time_log, plot_var_log[2,:],linestyle='-',marker="o",markersize=marker_size, lw=lw_act,color = 'blue')
    plt.grid()

        
