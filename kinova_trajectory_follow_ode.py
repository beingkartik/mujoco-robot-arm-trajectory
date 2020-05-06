#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 09:09:02 2020

@author: kartik
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:04:04 2020

@author: kartik
"""

# Get the windowing packages
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGroupBox, QSlider, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
#from PyQt5.QtCore import Qt, QSize
#from PyQt5.QtGui import QPainter, QBrush, QPen, QFont, QColor
#from random import random
import numpy as np
from numpy import sin, cos, pi
from scipy import integrate
import pyquaternion as pyq
import matplotlib.pyplot as plt
import math
#from time import sleep
import sys

if  '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path : sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2
sys.path.append('/home/kartik/Desktop/Courses/Intro-Kinematics/hw')
import robot_arm_2D_sent_by_cindy as robotArm
sys.path.append('/home/kartik/arm_tracking/trajectory_planning/icp')
import basicICP_class
 
sys.path.append('/home/kartik/arm_tracking/mujoco_kinova')
import mj_kinova_jac
import pid

class KinovaArmGUItraj(QMainWindow):
    def __init__(self,app):
        QMainWindow.__init__(self)

        self.setWindowTitle('Kinova robot arm')
        
        # Control buttons for the interface
        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)

        reach_jacobian_button = QPushButton('Reach Jacobian')
        reach_jacobian_button.clicked.connect(self.reach_jacobian)

        go_home_button = QPushButton('Go Home')
        go_home_button.clicked.connect(self.go_home)

        go_target_point = QPushButton('Go to target point')
        go_target_point.clicked.connect(self.go_target_point)
        
        trajectory_jacobian_button = QPushButton('Follow Trajectory(jacobian)')
        trajectory_jacobian_button.clicked.connect(self.trajectory_jacobian)

        trajectory_ode_button = QPushButton('Follow Trajectory(ODE)')
        trajectory_ode_button.clicked.connect(self.trajectory_ode)
        
        trajectory_ode_with_linear_error = QPushButton('Trajectory ODE (Linear Error)')
        trajectory_ode_with_linear_error.clicked.connect(self.estimate_linear_error)

        trajectory_bvp_button = QPushButton('Follow Trajectory(BVP)')
        trajectory_bvp_button.clicked.connect(self.trajectory_bvp)
                
        trajectory_pid_button = QPushButton('Follow Trajectory(PID)')
        trajectory_pid_button.clicked.connect(self.trajectory_pid)
        
        reaches = QGroupBox('Reaches')
        reaches_layout = QVBoxLayout()
        reaches_layout.addWidget( reach_jacobian_button )
        reaches_layout.addWidget( go_home_button)
        reaches_layout.addWidget( go_target_point)
        reaches_layout.addWidget( trajectory_jacobian_button)
        reaches_layout.addWidget( trajectory_ode_button)
        reaches_layout.addWidget( trajectory_ode_with_linear_error)
        reaches_layout.addWidget( trajectory_bvp_button)
        reaches_layout.addWidget( trajectory_pid_button)
        reaches.setLayout( reaches_layout)

        # The parameters of the robot arm we're simulating
        parameters = QGroupBox('Arm parameters')
        parameter_layout = QVBoxLayout()
        self.theta_base = robotArm.SliderDisplay('Angle base', -np.pi/2, np.pi/2, 0)
#        self.theta_elbow = robotArm.SliderDisplay('Angle elbow', -np.pi/2, np.pi/2, 0)
#        self.theta_wrist = robotArm.SliderDisplay('Angle wrist', -np.pi/2, np.pi/2, 0)
#        self.theta_fingers = robotArm.SliderDisplay('Angle fingers', -np.pi/4, 0, -np.pi/8)
#        self.length_upper_arm = robotArm.SliderDisplay('Length upper arm', 0.2, 0.4, 0.3)
#        self.length_lower_arm = robotArm.SliderDisplay('Length lower arm', 0.1, 0.2, 0.15)
#        self.length_fingers = robotArm.SliderDisplay('Length fingers', 0.05, 0.1, 0.075)
#        self.theta_slds = []
#        self.theta_slds.append( self.theta_base )
#        self.theta_slds.append( self.theta_elbow )
#        self.theta_slds.append( self.theta_wrist )

        parameter_layout.addWidget(self.theta_base)
#        parameter_layout.addWidget(self.theta_elbow)
#        parameter_layout.addWidget(self.theta_wrist)
#        parameter_layout.addWidget(self.theta_fingers)
#        parameter_layout.addWidget(self.length_upper_arm)
#        parameter_layout.addWidget(self.length_lower_arm)
#        parameter_layout.addWidget(self.length_fingers)

        parameters.setLayout(parameter_layout)

        # The point to reach to
        reach_point = QGroupBox('Reach point')
        reach_point_layout = QVBoxLayout()
        self.reach_x = robotArm.SliderDisplay('x', -2.50, 2.50, 0.5)
        self.reach_y = robotArm.SliderDisplay('y', -2.50, 2.50, 0.5)
        self.reach_z = robotArm.SliderDisplay('z', -2.50, 2.50, 0.5)
        self.jac_dist_thres = robotArm.SliderDisplay('jac_dist_thres',0,0.05,0.01)
        
        reach_point_layout.addWidget(self.reach_x)
        reach_point_layout.addWidget(self.reach_y)
        reach_point_layout.addWidget(self.reach_z)
        reach_point_layout.addWidget(self.jac_dist_thres)
        reach_point.setLayout(reach_point_layout)
        
        pid_control= QGroupBox('PID Controls')
        pid_control_layout = QVBoxLayout()
        self.kp = robotArm.SliderDisplay('kp', 0, 10, 1)
        self.kd= robotArm.SliderDisplay('kd', 0, 10, 1)
        self.ki= robotArm.SliderDisplay('ki', 0, 10, 0.5)
        self.error_thres = robotArm.SliderDisplay('error_thres',0,0.1,0.01)
        self.jac_step = robotArm.SliderDisplay('jac step size',0,1,0.2)
        self.max_iter = robotArm.SliderDisplay('max num of pid iterations',0,1000,100)
        go_target_point_pid_button = QPushButton('Go to target point(PID)')
        go_target_point_pid_button.clicked.connect(self.reach_pid)
        
        pid_control_layout.addWidget(self.kp)
        pid_control_layout.addWidget(self.kd)
        pid_control_layout.addWidget(self.ki)
        pid_control_layout.addWidget(self.error_thres)
        pid_control_layout.addWidget(self.jac_step)
        pid_control_layout.addWidget(self.max_iter)
        pid_control_layout.addWidget(go_target_point_pid_button)
        pid_control.setLayout(pid_control_layout)
       
        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)

        top_level_layout = QHBoxLayout()
        widget.setLayout(top_level_layout)
        left_side_layout = QVBoxLayout()
        right_side_layout = QVBoxLayout()

        left_side_layout.addWidget(reaches)
        left_side_layout.addWidget(reach_point)
        left_side_layout.addWidget(pid_control)
        left_side_layout.addStretch()
        left_side_layout.addWidget(parameters)

        right_side_layout.addWidget(quit_button)

        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(right_side_layout)

        robotArm.SliderDisplay.gui = self
        self.trajectory_point = 0
        self.trajectory_length = 10
        
        self.kinova = mj_kinova_jac.Kinova_MJ()
        est_quat_degree = 0
        est_quat_axis = (0,0,1)
        est_t = (0.2,0.2,0) #dx,dy,dz
        self.calibration_matrices((est_quat_degree,est_quat_axis),est_t)        
#        self.calibration_matrices()
        
        #local noises added to the joint angles
        self.mean_local_noise,self.std_local_noise = [0.01,0.005]
        self.reach_x.set_value(0.142)
        self.reach_y.set_value(0.028)
        self.reach_z.set_value(0.416)
    #modelling choice 
    #for robot, its base is the origin with end effector position defined w.r.t the base --robot frame
    #for target, it is defined w.r.t the object frame
    #for object frame, it is aligned with the robot frame in ideal conditions (or as defined in self.true_calibration)
    #global error is simulated as an error in the calibration matrix that the robot uses
    #NOTE : calibration_matrices here convert a point from a object frame to the robot frame
    def calibration_matrices(self,est_rot=None,est_t=None):
        
        true_rotation_matrix = pyq.Quaternion(degree=0,axis = (0,0,1)).transformation_matrix
        true_translation_matrix = np.identity(4)
        true_translation_matrix[:3,3] = (0,0,0) #dx,dy,dz
        
        self.true_calibration_matrix = true_rotation_matrix @ true_translation_matrix
        
        if est_rot is None and est_t is None:
            self.est_calibration_matrix = self.true_calibration_matrix
        else:
            est_rotation_matrix = pyq.Quaternion(degree=est_rot[0],axis = est_rot[1]).transformation_matrix
            est_translation_matrix = np.identity(4)
            est_translation_matrix[:3,3] = est_t #dx,dy,dz
            self.est_calibration_matrix = est_rotation_matrix@ est_translation_matrix
    
    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self,R) :

        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        
        singular = sy < 1e-6
    
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
    
        return np.array([x, y, z])
    
    # Calculates Rotation Matrix given euler angles.
    def eulerAnglesToRotationMatrix(self,theta) :
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                        ])
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                        ])
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    #if frame == 'robot' then trajectory is given in robot coordinate frame  (as per calibration matrices)
    #if frame == 'object' then trajectory is given in objects local fram of reference
    def desired_traj(self,frame = 'robot',cal_matrix=None):
        x = 0.142
        y =np.linspace(0.028,-0.5,self.trajectory_length)
#        y =np.linspace(0.028,0,self.trajectory_length)
        z = 0.416
        traj = []
        
        if frame == 'robot':
            for y_i in y:
                transformed_coordinate = cal_matrix@[x,y_i,z,1]
                traj.append(transformed_coordinate[:-1])
        
        elif frame == 'object':
            for y_i in y:
                traj.append([x,y_i,z])
        
        return traj

    
    #given an array of solution of ode, executes the joint angle changes 
    def execute_jointangle_trajectory(self,sol,ef_trajectory=None):     
        
        for item in sol:
            self.kinova.set_arm_thetas(item[-10:].reshape(-1))
            if ef_trajectory is not None:
                ef_trajectory.append(self.kinova.get_arm_end_pt())


#%% solving using jacobian method   
                
    def reach_jacobian(self,filler = None, d_step=0.002,dxdy=None):
        """ Use the Jacobian to calculate the desired angle change"""

        jacob = self.kinova.get_arm_jac()
        d_ang_save = self.kinova.get_arm_thetas()
        # Desired change in x,y,z
        pt_reach = self.kinova.get_arm_end_pt()
        dx_dy = np.zeros([3,1])
        if dxdy is None : 
            dx_dy[0,0] = self.reach_x.value() - pt_reach[0]
            dx_dy[1,0] = self.reach_y.value() - pt_reach[1]
            dx_dy[2,0] = self.reach_z.value() - pt_reach[2]

        else:
            dx_dy[0,0] = dxdy[0]
            dx_dy[1,0] = dxdy[1]
            dx_dy[2,0] = dxdy[2]
            
        # Use pseudo inverse to solve
        d_ang = np.linalg.lstsq( jacob, dx_dy, rcond = None )[0]
        res = jacob @ d_ang
#        print(jacob)
        d_try = d_step
        self.kinova.set_arm_thetas((d_ang_save + d_try*d_ang.T).reshape(-1))

        pt_reach_res = self.kinova.get_arm_end_pt()
        desired_text = "Desired dx dy {0:0.4f},{1:0.4f},{2:0.4f}".format(dx_dy[0,0], dx_dy[1,0],dx_dy[2,0])
        got_text = " got {0:0.4f},{1:0.4f},{1:0.4f}".format(res[0,0], res[1,0],res[2,0])
        actual_text = ", actual {0:0.4f},{1:0.4f},{1:0.4f}".format(pt_reach_res[0], pt_reach_res[1],pt_reach_res[2])
        
        self.kinova.text = desired_text + got_text + actual_text
#        print("Reach Jacobian:",self.kinova.text)1
#        print("Target pt:",(self.reach_x.value(),self.reach_y.value(),self.reach_z.value()),"Curr pos:",self.kinova.get_arm_end_pt())
        # to set text
        # self.robot_arm.text = text
                
    def recursive_reach_jacobian(self,target_pt):
        pt_ef_position = self.kinova.get_arm_end_pt()
        self.reach_x.set_value(target_pt[0])
        self.reach_y.set_value(target_pt[1])
        self.reach_z.set_value(target_pt[2])
        distance_to_target = pow( pow( pt_ef_position[0] - self.reach_x.value(), 2 ) 
                                + pow( pt_ef_position[1] - self.reach_y.value(), 2 ) 
                                + pow( pt_ef_position[2] - self.reach_z.value(), 2 ),0.5)
        
        counter = 0
        while distance_to_target>=self.jac_dist_thres.value():
            self.reach_jacobian()
            pt_ef_position = self.kinova.get_arm_end_pt()
            distance_to_target = distance_to_target = pow(pow( pt_ef_position[0] - self.reach_x.value(), 2 ) 
                                                        + pow( pt_ef_position[1] - self.reach_y.value(), 2 ) 
                                                        + pow( pt_ef_position[2] - self.reach_z.value(), 2 ),0.5)
            
            if counter%50 == 0 :                                           
                img = self.kinova.get_camera_image()
                plt.imshow(img)
                plt.draw()
                plt.pause(0.1)
                plt.clf()
            counter +=1 
            
#        print("Final distance: ",distance_to_target)
        return distance_to_target
    
    def go_home(self):
        home_position = self.desired_traj(frame='robot',cal_matrix=self.est_calibration_matrix)[0]
        self.trajectory_point = 0
        dist = self.recursive_reach_jacobian(home_position)
        print("REACHED HOME! Atempted to reach: ",str(home_position),"Reached: ",str(self.kinova.get_arm_end_pt()))

    def trajectory_jacobian(self):
        trajectory = self.desired_traj(frame='robot',cal_matrix=self.est_calibration_matrix)
        for target in trajectory:
            dist = self.recursive_reach_jacobian(target)
            print("Atempted to reach: ",str(target),"Reached: ",str(self.kinova.get_arm_end_pt()),"Dist: ",str(dist))
            
       
    def go_target_point(self):
        target_pt = (self.reach_x.value(),self.reach_y.value(),self.reach_z.value())
        self.recursive_reach_jacobian(target_pt)
        
#%% various kinds of model equation function needed for solving the ODE    

    ##same as model eqn but calculates the exact jacobian (by updating the matrices with new joints) 
    ##exact dx_xy(by gettting current ef)
    def model_eqn_precise_target(self,y,t,total_time,target):
        d_ang_save = self.kinova.get_arm_thetas()
        #update latest angles
        self.kinova.set_arm_thetas(y[-10:])

        #get updated ef position
        pt_ef_position = self.kinova.get_arm_end_pt()
        
        #calculate accruate jacobian
        jacob = self.kinova.get_arm_jac()

        # Desired change in x,y
        dx_dy = np.zeros([3,1])
        d_t = (target - pt_ef_position)/(total_time)
        dx_dy[0,0] = d_t[0]
        dx_dy[1,0] = d_t[1]
        dx_dy[2,0] = d_t[2]
        
        # Solve
        d_ang = np.linalg.lstsq( jacob, dx_dy, rcond = None )[0]
        gradient = [d for d in dx_dy] + [ang for ang in d_ang]

        #restore whatever the robot position was
        self.kinova.set_arm_thetas(d_ang_save)

#        print(gradient)
        return np.array(gradient).flatten()
    
    ##same as model eqn but calculates the exact jacobian (by updating the matrices with new joints) 
    ##d_t = (dx dy dz) :  calculated at starting point
    def model_eqn_precise_dxdy(self,y,t,total_time,d_t):
        d_ang_save = self.kinova.get_arm_thetas()
        #update latest angles
        self.kinova.set_arm_thetas(y[-10:])
        
        #calculate accruate jacobian
        jacob = self.kinova.get_arm_jac()
        
        dx_dy = np.zeros([3,1])
        dx_dy[0,0] = d_t[0]
        dx_dy[1,0] = d_t[1]
        dx_dy[2,0] = d_t[2]
        
        # Solve
        d_ang = np.linalg.lstsq( jacob, dx_dy, rcond = None )[0]
        gradient = [d for d in dx_dy] + [ang for ang in d_ang]

        #restore whatever the robot position was
        self.kinova.set_arm_thetas(d_ang_save)

        return np.array(gradient).flatten()

#%% solving ODE with initial value problem        
    def trajectory_ode(self,is_helper_for_bvp=False,add_local_error=False):

        print('is_helper: {} add_local_error: {}'.format(is_helper_for_bvp,add_local_error))
        target_list = self.desired_traj(frame = 'robot', cal_matrix = self.est_calibration_matrix)[self.trajectory_point+1:]
        overall_solution = []
        
        for i,target in enumerate(target_list):
            pt_ef_position = self.kinova.get_arm_end_pt()
            
            self.reach_x.set_value(target[0])
            self.reach_y.set_value(target[1])
            self.reach_z.set_value(target[2])
            
            d_ang_save = self.kinova.get_arm_thetas()
            y0 = [pt_ef_position[0],pt_ef_position[1],pt_ef_position[2]] + self.kinova.get_arm_thetas().tolist()
            total_time = 10
            self.total_timesteps = 101
            d_t = (target - pt_ef_position)/total_time
            t = np.linspace(0, total_time, self.total_timesteps)

            #solving ODE with updated jacobian(as function of theta)  and constant (dx,dy). Constant means calculated at initial point
#            sol = integrate.odeint(self.model_eqn_precise_dxdy,y0,t,args=(total_time,d_t))
            
            #solving ODE with updated jacobian(as function of theta)  and updated (dx,dy) (as function of end-effector position). Constant means calculated at initial point
            sol = integrate.odeint(self.model_eqn_precise_target,y0,t,args=(total_time,target))

            if is_helper_for_bvp:
                print("Completed initial ODE. Now solving bvp")
                #setting boundary conditions as final position and initial joint config
                self.bc = [target[0],target[1]] + d_ang_save
                #solve bvp with constant (dx,dy) at each time step
#                sol = integrate.solve_bvp(fun= lambda t,y:self.model_eqn_precise_dxdy_bvp(y,t,total_time,dx,dy),bc=self.bc_model,x=t,y=sol.transpose())
                
                #solve bvp with updated (dx,dy) at each time step
                sol = integrate.solve_bvp(fun= lambda t,y:self.model_eqn_precise_target_bvp(y,t,total_time,target),bc=self.bc_model,x=t,y=sol.transpose())
                sol = sol.y.transpose()
            
            
            #add local error
            if add_local_error:
                local_error = np.random.normal(self.mean_local_noise,self.std_local_noise, sol.shape)
                sol = sol+local_error
            
            self.execute_jointangle_trajectory(sol)
            overall_solution=overall_solution + sol.tolist()
            
            print("Atempted to reach: [{:.3}, {:.3}, {:.3}] Reached: [{:.3}, {:.3}, {:.3}]".
                                       format(target[0],target[1],target[2],self.kinova.get_arm_end_pt()[0],self.kinova.get_arm_end_pt()[1],self.kinova.get_arm_end_pt()[2]))
#            print("Atempted to reach: ",str(target),"Actual Target: ",str(self.desired_traj(frame = 'robot', cal_matrix = self.true_calibration_matrix)[self.trajectory_point+1+i]),
#            "Reached: ",str(self.kinova.get_arm_end_pt()))
            img = self.kinova.get_camera_image()
            plt.imshow(img)
            plt.draw()
            plt.pause(0.1)
            plt.clf()

        return overall_solution
        
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

    def trajectory_bvp(self):
        pass
    
    def estimate_linear_error(self,is_helper_for_bvp= False,num_runs=10):
        
        #get solution with the assumed calibration (and no robot local error)
        overall_solution = self.trajectory_ode(add_local_error=False)
        #solution has been found. Now it needs to run num_runs number of times
        #each run will have new local error and have icp performed
        #ICP is performed in the object coordinate frame (ie local - in which trajectory is defined and EF is tracked)
        
        ##to get the target trajectory points corresponding to the ODE solution points
        target_fine_list = []
        target_list_local = self.desired_traj(frame='object')[self.trajectory_point:]
         
        for i in range(len(target_list_local)-1):
            target_trajectory = np.linspace(target_list_local[i],target_list_local[i+1],self.total_timesteps)
            target_fine_list = target_fine_list + target_trajectory.tolist()
        target_fine_list = np.asarray(target_fine_list)        
        
        #run robot num_runs times and perform ICP each time    
        for run in range(num_runs):
            print(run)
            self.kinova.set_arm_thetas(overall_solution[0][-10:])

            ef_trajectory = []
            sol = np.asarray(overall_solution)
            
            #add local error
            local_error = np.random.normal(self.mean_local_noise,self.std_local_noise, (sol.shape[0],sol.shape[1]-2))*0
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
            
            plt.clf()
#            plt.scatter(ef_trajectory_object_frame[:,0],ef_trajectory_object_frame[:,1],)
#            plt.scatter(target_fine_list[:,0],target_fine_list[:,1])
#
#            plt.draw()
#            plt.pause(2)
            icp = basicICP_class.basicICP3d(max_loop=500)
            euler_angles = self.rotationMatrixToEulerAngles(self.est_calibration_matrix[:3,:3])
            initial_estimate = np.zeros((6,1))  #alpha,beta,gamma,tx,ty,tz
            initial_estimate[:3,0] = euler_angles 
            initial_estimate[3:6,0] = self.est_calibration_matrix[:3,3] 
            T= icp.icp_point_to_point_lm(ef_trajectory_object_frame.copy(),target_fine_list.copy(),initial_estimate,loop=10)
#%% plotting icp data and results            
#            dx = T[0,2]
#            dy = T[1,2]
#            rotation = np.arcsin(T[0,1]) * 360 / 2 / np.pi
#        
            print("Result: {}".format(T))
#            print("error",error)
#            print("rotation°",rotation)
#            print("dx",dx)
#            print("dy",dy)
#        
            result_transformation = np.identity(4)
            result_transformation[:3,:3] = self.eulerAnglesToRotationMatrix(T[:3])
            result_transformation[:3,3]  = T[3:6][:,0]
            result = (np.transpose(result_transformation @np.transpose(ef_trajectory_homogenous))[:,:-1]).T
            
            
            initial_transformation = np.identity(4)
            result_transformation[:3,:3] = self.eulerAnglesToRotationMatrix(T[:3])
            result_transformation[:3,3]  = T[3:6][:,0]
            initial_result = (np.transpose(initial_transformation @np.transpose(ef_trajectory_homogenous))[:,:-1]).T
            
            plt.plot(target_fine_list.T[0], target_fine_list.T[1], label="target_fine_list.T")
            plt.plot(ef_trajectory_object_frame.T[0], ef_trajectory_object_frame.T[1], label="ef_trajectory_object_frame.T")
            plt.legend(loc="upper left")
            plt.axis('square')
            plt.draw()
            plt.pause(5)            

            plt.plot(target_fine_list.T[0], target_fine_list.T[1], label="target_fine_list.T")
            plt.plot(ef_trajectory_object_frame.T[0], ef_trajectory_object_frame.T[1], label="ef_trajectory_object_frame.T")           
            plt.plot(result[0], result[1], label="result")
            plt.plot(initial_result[0], initial_result[1], label="initial result")
            plt.legend(loc="upper left")
            plt.axis('square')
            plt.draw()
            plt.pause(4)
               
#%% PID Control
            
    def reach_pid(self,filler=None,jointAngles=None, target=None):
        pid_control =  pid.PID(Kp=self.kp.value(),Kd=self.kd.value(),Ki=self.ki.value(),origin_time=0)
        if jointAngles is not None:
            self.kinova.set_arm_thetas(jointAngles)

        if target is None:
            target = np.zeros(3)
            target[0] = self.reach_x.value()
            target[1] = self.reach_y.value()
            target[2] = self.reach_y.value()
            
        transform_robot_to_local_frame = np.linalg.inv(self.true_calibration_matrix)
#        robot_end_point_homo = np.ones(4,1)
#        robot_end_point_homo[:-1] = self.kinova.get_arm_end_pt()
#        robot_end_point_object_frame = np.transpose(transform_robot_to_local_frame @robot_end_point_homo)[:,:-1]
#            
        t = 0
#        print("next point:",str(jointAngles))
        while(True):
            robot_end_point_homo = np.ones((4))
            robot_end_point_homo[:-1] = self.kinova.get_arm_end_pt()
            robot_end_point_object_frame = np.transpose(transform_robot_to_local_frame @robot_end_point_homo)[:-1]
            
            error = target - robot_end_point_object_frame
            dx = pid_control.Update(error,t)
#            print("Error {}, dx {}, end pt {}".format(error.round(3),dx, self.robot_arm.arm_end_pt()))
            t = t+1
            if t==0:
                continue
            if max(abs(error)) < self.error_thres.value():
                print("Point reached #Iter {}".format(t))
#                print("Error {}, Iter {} , Thres {:.3}, EndPt {}".format(error,t,self.error_thres.value(),self.robot_arm.arm_end_pt()))
                return
            if t >self.max_iter.value():
                print("Max iter Reached #Iter {}".format(t))
#                print("Error {}, Iter {} , Thres {:.3}, EndPt {}".format(error,t,self.error_thres.value(),self.robot_arm.arm_end_pt()))
                return 
            
            self.reach_jacobian(dxdy = dx, d_step = self.jac_step.value())  
            
    
    #given an array of solution of ode, executes the joint angle changes 
    def execute_jointangle_trajectory_pid(self,sol,target_traj,ef_trajectory=None):     
        for num,item in enumerate(sol):
            self.reach_pid(jointAngles=item[-10:],target=target_traj[num])
            if ef_trajectory is not None:
                ef_trajectory.append(self.robot_arm.arm_end_pt())
    
    def trajectory_pid(self):
        #get solution with the assumed calibration (and no robot local error)
        overall_solution = self.trajectory_ode(add_local_error=False)
        print("found ode solution")
        ##to get the target trajectory points corresponding to the ODE solution points
        target_fine_list = []
        target_list_local = self.desired_traj(frame='object')[self.trajectory_point:]
         
        for i in range(len(target_list_local)-1):
            target_trajectory = np.linspace(target_list_local[i],target_list_local[i+1],self.total_timesteps)
            target_fine_list = target_fine_list + target_trajectory.tolist()
        target_fine_list = np.asarray(target_fine_list)        
        
        for i,angle in enumerate(overall_solution[0][2:]):
            self.theta_slds[i].set_value(angle)

        ef_trajectory = []
        sol = np.asarray(overall_solution)
        self.execute_jointangle_trajectory_pid(sol,target_fine_list)   


if __name__=='__main__':
    app = QApplication([])
    gui = KinovaArmGUItraj(app)
    gui.show()
    app.exec_()

