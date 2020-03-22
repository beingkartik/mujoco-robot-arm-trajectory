#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:04:04 2020

@author: kartik
"""

# Get the windowing packages
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGroupBox, QSlider, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPainter, QBrush, QPen, QFont, QColor
from random import random
import numpy as np
from numpy import sin, cos, pi

import sys
sys.path.append('/home/kartik/Desktop/Courses/Intro-Kinematics/hw')
import robot_arm_2D_sent_by_cindy as robotArm
import icp_matching_2d 

from scipy import integrate

import matplotlib.pyplot as plt
from time import sleep

if  '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path : sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2

class RobotArmGUItraj(robotArm.RobotArmGUI):
    def __init__(self,app):
        super(RobotArmGUItraj,self).__init__(app)
        self.setWindowTitle('ROB 514 2D robot arm')

        # Control buttons for the interface
        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)

        # Different do reach commands
        reach_gradient_button = QPushButton('Reach gradient')
        reach_gradient_button.clicked.connect(self.reach_gradient)

        reach_jacobian_button = QPushButton('Reach Jacobian')
        reach_jacobian_button.clicked.connect(self.reach_jacobian)

        go_home_button = QPushButton('Go Home')
        go_home_button.clicked.connect(self.go_home)

        trajectory_jacobian_button = QPushButton('Follow Trajectory(jacobian)')
        trajectory_jacobian_button.clicked.connect(self.trajectory_jacobian)

        trajectory_ode_button = QPushButton('Follow Trajectory(ODE)')
        trajectory_ode_button.clicked.connect(self.estimate_linear_error)
        
        trajectory_bvp_button = QPushButton('Follow Trajectory(BVP)')
        trajectory_bvp_button.clicked.connect(self.trajectory_bvp)
        reaches = QGroupBox('Reaches')
        reaches_layout = QVBoxLayout()
        reaches_layout.addWidget( reach_gradient_button )
        reaches_layout.addWidget( reach_jacobian_button )
        reaches_layout.addWidget( go_home_button)
        reaches_layout.addWidget( trajectory_jacobian_button)
        reaches_layout.addWidget( trajectory_ode_button)
        reaches_layout.addWidget( trajectory_bvp_button)
        reaches.setLayout( reaches_layout)

        # The parameters of the robot arm we're simulating
        parameters = QGroupBox('Arm parameters')
        parameter_layout = QVBoxLayout()
        self.theta_base = robotArm.SliderDisplay('Angle base', -np.pi/2, np.pi/2, 0)
        self.theta_elbow = robotArm.SliderDisplay('Angle elbow', -np.pi/2, np.pi/2, 0)
        self.theta_wrist = robotArm.SliderDisplay('Angle wrist', -np.pi/2, np.pi/2, 0)
        self.theta_fingers = robotArm.SliderDisplay('Angle fingers', -np.pi/4, 0, -np.pi/8)
        self.length_upper_arm = robotArm.SliderDisplay('Length upper arm', 0.2, 0.4, 0.3)
        self.length_lower_arm = robotArm.SliderDisplay('Length lower arm', 0.1, 0.2, 0.15)
        self.length_fingers = robotArm.SliderDisplay('Length fingers', 0.05, 0.1, 0.075)
        self.theta_slds = []
        self.theta_slds.append( self.theta_base )
        self.theta_slds.append( self.theta_elbow )
        self.theta_slds.append( self.theta_wrist )

        parameter_layout.addWidget(self.theta_base)
        parameter_layout.addWidget(self.theta_elbow)
        parameter_layout.addWidget(self.theta_wrist)
        parameter_layout.addWidget(self.theta_fingers)
        parameter_layout.addWidget(self.length_upper_arm)
        parameter_layout.addWidget(self.length_lower_arm)
        parameter_layout.addWidget(self.length_fingers)

        parameters.setLayout(parameter_layout)

        # The point to reach to
        reach_point = QGroupBox('Reach point')
        reach_point_layout = QVBoxLayout()
        self.reach_x = robotArm.SliderDisplay('x', 0, 1, 0.5)
        self.reach_y = robotArm.SliderDisplay('y', 0, 1, 0.5)
        random_button = QPushButton('Random')
        random_button.clicked.connect(self.random_reach)
        reach_point_layout.addWidget(self.reach_x)
        reach_point_layout.addWidget(self.reach_y)
        reach_point_layout.addWidget(random_button)
        reach_point.setLayout(reach_point_layout)

        # The display for the graph
        self.robot_arm = robotArm.DrawRobot(self)

        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)

        top_level_layout = QHBoxLayout()
        widget.setLayout(top_level_layout)
        left_side_layout = QVBoxLayout()
        right_side_layout = QVBoxLayout()

        left_side_layout.addWidget(reaches)
        left_side_layout.addWidget(reach_point)
        left_side_layout.addStretch()
        left_side_layout.addWidget(parameters)

        right_side_layout.addWidget( self.robot_arm )
        right_side_layout.addWidget(quit_button)

        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(right_side_layout)

        robotArm.SliderDisplay.gui = self
        self.trajectory_point = 0
        self.trajectory_length = 100

        self.calibration_matrices(est_rot=0,est_dx=0.05,est_dy = 0.05)        
        
        #local noises added to the joint angles
        self.mean_local_noise,self.std_local_noise = [0.01,0.005]
#        self.local_noises = np.random.normal(mean_local_noise,std_local_noise,len(self.theta_slds)) #mean,std,num_of_joints

        
    #modelling choice 
    #for robot, its base is the origin with end effector position defined w.r.t the base --robot frame
    #for target, it is defined w.r.t the object frame
    #for object frame, it is aligned with the robot frame in ideal conditions (or as defined in self.true_calibration)
    #global error is simulated as an error in the calibration matrix that the robot uses
    #NOTE : calibration_matrices here convert a point from a object frame to the robot frame
    def calibration_matrices(self,est_rot=None,est_dx=None, est_dy=None):
        
        true_rotation_matrix = self.robot_arm.rotation_matrix(0)
        true_translation_matrix = self.robot_arm.translation_matrix(0,0)
        self.true_calibration_matrix = true_rotation_matrix @ true_translation_matrix
        
        if est_dx is None and est_dy is None and est_rot is None:
            self.est_calibration_matrix = self.true_calibration_matrix
        else:
            est_rotation_matrix = self.robot_arm.rotation_matrix(est_rot)
            est_translation_matrix = self.robot_arm.translation_matrix(est_dx,est_dy)
            self.est_calibration_matrix = est_rotation_matrix@ est_translation_matrix

     
    #if frame == 'robot' then trajectory is given in robot coordinate frame  (as per calibration matrices)
    #if frame == 'object' then trajectory is given in objects local fram of reference
    
    def desired_traj(self,frame = 'robot',cal_matrix=None):
        x = 0.3
        y = np.linspace(0.5,0.3,self.trajectory_length)
        traj = []
        
        if frame == 'robot':
            for y_i in y:
                transformed_coordinate = cal_matrix@[x,y_i,1]
                traj.append(transformed_coordinate[:-1])
        
        elif frame == 'object':
            for y_i in y:
                traj.append([x,y_i])
        
        return traj

    
    #given an array of solution of ode, executes the joint angle changes 
    def execute_jointangle_trajectory(self,sol,ef_trajectory=None):     
        
        for item in sol:
            for i in range(len(item)-2):
                self.theta_slds[i].set_value( item[i+2])
            if ef_trajectory is not None:
                ef_trajectory.append(self.robot_arm.arm_end_pt())
            


#%% solving using jacobian method    
    def recursive_reach_jacobian(self,target_pt,dist_thres=0.02):
        pt_ef_position = self.robot_arm.arm_end_pt()
        self.reach_x.set_value(target_pt[0])
        self.reach_y.set_value(target_pt[1])
        self.robot_arm.repaint()
        distance_to_target = pow(pow( pt_ef_position[0] - self.reach_x.value(), 2 ) + pow( pt_ef_position[1] - self.reach_y.value(), 2 ),0.5)
        
        while distance_to_target>=dist_thres:
            self.reach_jacobian()
            pt_ef_position = self.robot_arm.arm_end_pt()
            distance_to_target = pow(pow( pt_ef_position[0] - self.reach_x.value(), 2 ) + pow( pt_ef_position[1] - self.reach_y.value(), 2 ),0.5)
#            print(distance_to_target)
        return distance_to_target
    
    def go_home(self):
        home_position = self.desired_traj(frame='robot',cal_matrix=self.est_calibration_matrix)[0]
        self.trajectory_point = 0
        self.recursive_reach_jacobian(home_position)
        
    def trajectory_jacobian(self):
        trajectory = self.desired_traj(frame='robot',cal_matrix=self.est_calibration_matrix)
        for target in trajectory:
            print("Atempted to reach: ",str(target),"Reached: ",str(self.robot_arm.arm_end_pt()))
            
        
#%% various kinds of model equation function needed for solving the ODE    
    def model_eqn(self,y,t,total_time,dx,dy):
        omega_hat = [0,0,1]
        mats = self.robot_arm.get_matrics()
        jacob = np.zeros([2,3])
        matrix_order = ['wrist', 'forearm', 'upperarm']
        mat_accum = np.identity(3)
        for i,c in enumerate(matrix_order):
            mat_accum = mats[c + '_R'] @ mats[c + '_T'] @ mat_accum
            r = [ mat_accum[0,2], mat_accum[1,2], 0 ]
            omega_cross_r = np.cross( omega_hat, r )
            jacob[0:2,2-i] = np.transpose( omega_cross_r[0:2] )


        # Desired change in x,y
        dx_dy = np.zeros([2,1])
        dx_dy[0,0] = dx
        dx_dy[1,0] = dy
        
        # Solve
        d_ang = np.linalg.lstsq( jacob, dx_dy, rcond = None )[0]
        gradient = [d for d in dx_dy] + [ang for ang in d_ang]
        print(gradient)
        return np.array(gradient).flatten()
    
    ##same as model eqn but calculates the exact jacobian (by updating the matrices with new joints) 
    ##exact dx_xy(by gettting current ef)
    def model_eqn_precise_target(self,y,t,total_time,target):
        d_ang_save = [ self.theta_slds[i].value() for i in range(0,3) ]
        #update latest angles
        for i,ang in enumerate(y[2:]):
            self.theta_slds[i].set_value(ang)
        #get updated ef position
        pt_ef_position = self.robot_arm.arm_end_pt()
        
        #calculate accruate jacobian
        omega_hat = [0,0,1]
        mats = self.robot_arm.get_matrics()
        jacob = np.zeros([2,3])
        matrix_order = ['wrist', 'forearm', 'upperarm']
        mat_accum = np.identity(3)
        for i,c in enumerate(matrix_order):
            mat_accum = mats[c + '_R'] @ mats[c + '_T'] @ mat_accum
            r = [ mat_accum[0,2], mat_accum[1,2], 0 ]
            omega_cross_r = np.cross( omega_hat, r )
            jacob[0:2,2-i] = np.transpose( omega_cross_r[0:2] )
        # Desired change in x,y
        dx_dy = np.zeros([2,1])
        [dx,dy] = (target - pt_ef_position)/(total_time)
        dx_dy[0,0] = dx
        dx_dy[1,0] = dy
        
        # Solve
        d_ang = np.linalg.lstsq( jacob, dx_dy, rcond = None )[0]
        gradient = [d for d in dx_dy] + [ang for ang in d_ang]

        #restore whatever the robot position was
        for i,ang_save in enumerate(d_ang_save):
            self.theta_slds[i].set_value(ang_save)

#        print(gradient)
        return np.array(gradient).flatten()

    ##same as model eqn but calculates the exact jacobian (by updating the matrices with new joints) 
    ##dx dy calculated at starting point
    def model_eqn_precise_dxdy(self,y,t,total_time,dx,dy):
        d_ang_save = [ theta.value() for theta in self.theta_slds]
        #update latest angles
        for i,ang in enumerate(y[2:]):
            self.theta_slds[i].set_value(ang)
        
        #calculate accruate jacobian
        omega_hat = [0,0,1]
        mats = self.robot_arm.get_matrics()
        jacob = np.zeros([2,3])
        matrix_order = ['wrist', 'forearm', 'upperarm']
        mat_accum = np.identity(3)
        for i,c in enumerate(matrix_order):
            mat_accum = mats[c + '_R'] @ mats[c + '_T'] @ mat_accum
            r = [ mat_accum[0,2], mat_accum[1,2], 0 ]
            omega_cross_r = np.cross( omega_hat, r )
            jacob[0:2,2-i] = np.transpose( omega_cross_r[0:2] )
        # Desired change in x,y
        dx_dy = np.zeros([2,1])
        dx_dy[0,0] = dx
        dx_dy[1,0] = dy
        
        # Solve
        d_ang = np.linalg.lstsq( jacob, dx_dy, rcond = None )[0]
        gradient = [d for d in dx_dy] + [ang for ang in d_ang]

        #restore whatever the robot position was
        for i,ang_save in enumerate(d_ang_save):
            self.theta_slds[i].set_value(ang_save)

        return np.array(gradient).flatten()

#%% solving ODE with initial value problem        
    def trajectory_ode(self,is_helper_for_bc=False):
        #initialise robots estimate of the object frame calibration
        self.calibration_matrices(0,0.0,0.0)


        print('is_helper: ',is_helper_for_bc)
        target_list = self.desired_traj(frame = 'robot', cal_matrix = self.est_calibration_matrix)[self.trajectory_point+1:]
        for i,target in enumerate(target_list):
            pt_ef_position = self.robot_arm.arm_end_pt()
            
            self.reach_x.set_value(target[0])
            self.reach_y.set_value(target[1])
            self.robot_arm.repaint()
            d_ang_save = [ theta.value() for theta in self.theta_slds ]
            y0 = [pt_ef_position[0],pt_ef_position[1]] + [ theta.value() for theta in self.theta_slds ]
            print("y0:",str(y0))
            total_time = 10
            [dx,dy] = (target - pt_ef_position)/total_time
            t = np.linspace(0, total_time, 101)
            
            
            #solving ODE with constant jacobian and constant (dx,dy). Constant means calculated at initial point
#            sol = integrate.odeint(self.model_eqn,y0,t,args=(total_time,dx,dy))
           
            #solving ODE with updated jacobian(as function of theta)  and constant (dx,dy). Constant means calculated at initial point
            sol = integrate.odeint(self.model_eqn_precise_dxdy,y0,t,args=(total_time,dx,dy))
            
            #solving ODE with updated jacobian(as function of theta)  and updated (dx,dy) (as function of end-effector position). Constant means calculated at initial point
#            sol = integrate.odeint(self.model_eqn_precise_target,y0,t,args=(total_time,target))

            if is_helper_for_bc:
                print("Completed initial ODE. Now solving bvp")
                #setting boundary conditions as final position and initial joint config
                self.bc = [target[0],target[1]] + d_ang_save
                #solve bvp with constant (dx,dy) at each time step
#                sol = integrate.solve_bvp(fun= lambda t,y:self.model_eqn_precise_dxdy_bvp(y,t,total_time,dx,dy),bc=self.bc_model,x=t,y=sol.transpose())
                
                #solve bvp with updated (dx,dy) at each time step
                sol = integrate.solve_bvp(fun= lambda t,y:self.model_eqn_precise_target_bvp(y,t,total_time,target),bc=self.bc_model,x=t,y=sol.transpose())
                sol = sol.y.transpose()
            
            
            #add local error
            local_error = np.random.normal(self.mean_local_noise,self.std_local_noise, sol.shape)
            sol = sol+local_error
            
            self.execute_jointangle_trajectory(sol)
            print("Atempted to reach: ",str(target),"Actual Target: ",str(self.desired_traj(frame = 'robot', cal_matrix = self.true_calibration_matrix)[self.trajectory_point+1+i]),
            "Reached: ",str(self.robot_arm.arm_end_pt()))
#            print(overall_solution)
        
#%% solving ODE with boundary value problem    
    def model_eqn_precise_dxdy_bvp(self,y,t,total_time,dx,dy):
        gradient_all_time = np.zeros((len(t),5))
        y = y.copy().transpose()
        for i,y0 in enumerate(y):
            gradient_timestep=self.model_eqn_precise_dxdy(y0,t,total_time,dx,dy)
            gradient_all_time[i] = gradient_timestep
        return gradient_all_time.transpose()
    
    def model_eqn_precise_target_bvp(self,y,t,total_time,target):
        gradient_all_time = np.zeros((len(t),5))
        y = y.copy().transpose()
        for i,y0 in enumerate(y):
            gradient_timestep=self.model_eqn_precise_target(y0,t,total_time,target)
            gradient_all_time[i] = gradient_timestep
        return gradient_all_time.transpose()
                
    ##boundary value conditions - one boundary condition needs to be specified for each variable ie dimension = dimension of y
    ## returns the residual values
    def bc_model(self,ya,yb):         
        return np.array([yb[0]-self.bc[0], yb[1]-self.bc[1],ya[2]-self.bc[2],ya[3]-self.bc[3],ya[4]-self.bc[4]])

    ##solve piecewise bvp -- a bvp is solved between each two waypoints        
    def trajectory_bvp(self):
        _ = self.trajectory_ode(is_helper_for_bc=True)

    ##solve one common bvp for the entire trajectory - from start to end of trajectory
    def trajectory_bvp_together(self):
        ##attempt to solve the complete ODE as one problem
        print('Complete single BVP being solved')
        target_list = self.desired_traj(frame = 'robot', cal_matrix = self.est_calibration_matrix)[self.trajectory_point+1:]
        total_time = 10 #for each waypoint

        t_steps = 101
        t_total_list = np.linspace(0,total_time*len(target_list),t_steps*len(target_list) - (len(target_list)-1))
        
        initial_values_bvp = np.zeros((t_steps*len(target_list) - (len(target_list)-1),2+len(self.theta_slds)))
        d_ang_save = [ theta.value() for theta in self.theta_slds ]

        for i,target in enumerate(target_list):
            pt_ef_position = self.robot_arm.arm_end_pt()
            
            self.reach_x.set_value(target[0])
            self.reach_y.set_value(target[1])
            self.robot_arm.repaint()
            y0 = [pt_ef_position[0],pt_ef_position[1]] + [ theta.value() for theta in self.theta_slds ]
            
            [dx,dy] = (target - pt_ef_position)/total_time
            t = np.linspace(0, total_time, t_steps)
            
            #solving ODE with constant jacobian and constant (dx,dy). Constant means calculated at initial point
#            sol = integrate.odeint(self.model_eqn,y0,t,args=(total_time,dx,dy))
           
            #solving ODE with updated jacobian(as function of theta)  and constant (dx,dy). Constant means calculated at initial point
            sol = integrate.odeint(self.model_eqn_precise_dxdy,y0,t,args=(total_time,dx,dy))
            for item in sol:
                for i in range(len(item)-2):
                    self.theta_slds[i].set_value( item[i+2])
            print("Atempted to reach: ",str(target),"Reached: ",str(self.robot_arm.arm_end_pt()))
            initial_values_bvp[i*t_steps-1:(i+1)*t_steps-1] = sol
            print(i*t_steps-1,(i+1)*t_steps-1)
        
        self.bc = [target_list[-1][0],target_list[-1][1]] + d_ang_save
        
        print("Resetting arm")
        for i,ang_save in enumerate(d_ang_save):
            self.theta_slds[i].set_value(ang_save)
        sol = integrate.solve_bvp(fun= lambda t,y:self.model_eqn_precise_target_bvp(y,t,total_time,target),bc=self.bc_model,x=t_total_list,y=initial_values_bvp.transpose())
        sol = sol.y.transpose()
        
        self.mode = 'test'
        self.execute_jointangle_trajectory(sol)
        print("Atempted to reach: ",str(target),"Actual Target: ",str(self.desired_traj(frame='robot',cal_matrix=self.true_calibration_matrix)[self.trajectory_point+1+i]),
                                        "Reached: ",str(self.robot_arm.arm_end_pt()))
            
#%% estimating linear error (ie global pose error + robot linear error) -- 
#   pass through trajectory multiple times, make estimate of linear error each time through ICP
#   take the average linear error

    #num_runs: number of times robot makes a pass through the trajectory (with the same solution of ode)        
    def estimate_linear_error(self,is_helper_for_bc= False,num_runs=10):
#        self.calibration_matrices(est_rot=0,est_dx=1.0,est_dy = 1.0)
        print(num_runs)
        print('is_helper: ',is_helper_for_bc)
        target_list = self.desired_traj(frame='robot',cal_matrix = self.est_calibration_matrix)[self.trajectory_point+1:]
        overall_solution = []

        for i,target in enumerate(target_list):
            pt_ef_position = self.robot_arm.arm_end_pt()
            
            self.reach_x.set_value(target[0])
            self.reach_y.set_value(target[1])
            self.robot_arm.repaint()
            d_ang_save = [ theta.value() for theta in self.theta_slds ]
            y0 = [pt_ef_position[0],pt_ef_position[1]] + [ theta.value() for theta in self.theta_slds ]
            print("y0:",str(y0))
            total_time = 10
            total_timesteps = 101
            [dx,dy] = (target - pt_ef_position)/total_time
            t = np.linspace(0, total_time, total_timesteps)
            
            self.mode ='train'
            #solving ODE with constant jacobian and constant (dx,dy). Constant means calculated at initial point
#            sol = integrate.odeint(self.model_eqn,y0,t,args=(total_time,dx,dy))
           
            #solving ODE with updated jacobian(as function of theta)  and constant (dx,dy). Constant means calculated at initial point
            sol = integrate.odeint(self.model_eqn_precise_dxdy,y0,t,args=(total_time,dx,dy))
            
            #solving ODE with updated jacobian(as function of theta)  and updated (dx,dy) (as function of end-effector position). Constant means calculated at initial point
#            sol = integrate.odeint(self.model_eqn_precise_target,y0,t,args=(total_time,target))

            if is_helper_for_bc:
                print("Completed initial ODE. Now solving bvp")
                #setting boundary conditions as final position and initial joint config
                self.bc = [target[0],target[1]] + d_ang_save
                #solve bvp with constant (dx,dy) at each time step
#                sol = integrate.solve_bvp(fun= lambda t,y:self.model_eqn_precise_dxdy_bvp(y,t,total_time,dx,dy),bc=self.bc_model,x=t,y=sol.transpose())
                
                #solve bvp with updated (dx,dy) at each time step
                sol = integrate.solve_bvp(fun= lambda t,y:self.model_eqn_precise_target_bvp(y,t,total_time,target),bc=self.bc_model,x=t,y=sol.transpose())
                sol = sol.y.transpose()
            
            self.execute_jointangle_trajectory(sol)
            overall_solution=overall_solution + sol.tolist()
            print("Atempted to reach: ",str(target),"Actual Target: ",str(self.desired_traj(frame='robot',cal_matrix=self.true_calibration_matrix)[self.trajectory_point+1+i]),
            "Reached: ",str(self.robot_arm.arm_end_pt()))         
        
        #solution has been found. Now it needs to run num_runs number of times
        #each run will have new local error and have icp performed
        #ICP is performed in the object coordinate frame (ie local - in which trajectory is defined and EF is tracked)
        
        ##to get the target trajectory points corresponding to the ODE solution points
        target_fine_list = []
        target_list_local = self.desired_traj(frame='object')[self.trajectory_point:]
         
        for i in range(len(target_list_local)-1):
            target_trajectory = np.linspace(target_list_local[i],target_list_local[i+1],total_timesteps)
            target_fine_list = target_fine_list + target_trajectory.tolist()
        target_fine_list = np.asarray(target_fine_list)        
        
        #run robot num_runs times and perform ICP each time    
        for run in range(num_runs):
            print(run)
            for i,angle in enumerate(overall_solution[0][2:]):
                self.theta_slds[i].set_value(angle)

            ef_trajectory = []
            sol = np.asarray(overall_solution)
            
            #add local error
            local_error = np.random.normal(self.mean_local_noise,self.std_local_noise, (sol.shape[0],sol.shape[1]-2))
            sol[:,2:] = sol[:,2:]+local_error
            self.execute_jointangle_trajectory(sol,ef_trajectory)
            ef_trajectory = np.asarray(ef_trajectory)
            
            #convert ef_trajectory from robot frame to local frame 
            #(as observed using sensor in local frame -- hence here actual transformation is used)
            #(it will not match with target points in their local frame if the est_calibration_matrix (used for ODE) was wrong)
            transform_robot_to_local_frame = np.linalg.inv(self.true_calibration_matrix)
            ef_trajectory_homogenous = np.ones((ef_trajectory.shape[0],ef_trajectory.shape[1]+1))
            ef_trajectory_homogenous[:,:-1] = ef_trajectory
            ef_trajectory_object_frame = np.transpose(transform_robot_to_local_frame @np.transpose(ef_trajectory_homogenous))[:,:-1]
            
            plt.scatter(ef_trajectory_object_frame[:,0],ef_trajectory_object_frame[:,1])
            plt.scatter(target_fine_list[:,0],target_fine_list[:,1])
            plt.show()
            plt.pause(1)
            plt.close()

            icp = icp_matching_2d.icp_matching_2d()
            T,error = icp.icp(ef_trajectory_object_frame,target_fine_list,self.est_calibration_matrix,max_time=10)
#%% plotting icp data and results            
            dx = T[0,2]
            dy = T[1,2]
            rotation = np.arcsin(T[0,1]) * 360 / 2 / np.pi
        
            print("T",T)
            print("error",error)
            print("rotation°",rotation)
            print("dx",dx)
            print("dy",dy)
        
            result = cv2.transform(np.array([ef_trajectory_object_frame], copy=True).astype(np.float32), T).T
        
            plt.plot(target_fine_list.T[0], target_fine_list.T[1], label="target_fine_list.T")
            plt.plot(ef_trajectory_object_frame.T[0], ef_trajectory_object_frame.T[1], label="ef_trajectory_object_frame.T")
            plt.plot(result[0], result[1], label="result: "+str(rotation)+"° - "+str([dx,dy]))
            plt.legend(loc="upper left")
            plt.axis('square')
            plt.show()
            plt.pause(4)
            plt.close()

            
                
                

if __name__=='__main__':
    app = QApplication([])
    gui = RobotArmGUItraj(app)
    gui.show()
    app.exec_()

