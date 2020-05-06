#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 19:25:26 2020

@Modifier: kartik
Modified into a class for ease of use within a project

Original Code :
Created on Mon Jan 23 18:08:58 2017

@author: agnivsen

A basic Iterative Closes Point matching betwwen two set of point cloud, 
                        or one point cloud and one point + normal pair.
                        
Basic point to plane matching has been done using a Least squares approach and a Gauss-Newton approach
Point to point matching has been done using Gauss-Newton only
"""

import numpy as np
import re
import sys
sys.path.append('/home/kartik/arm_tracking/trajectory_planning/icp')

import transformations as transform


class basicICP3d():
    def __init__(self,max_loop = 50):
        self.max_loop = max_loop
        
    def icp_point_to_point_lm(self,source_points, dest_points,initial,loop):
        """
        Point to point matching using Gauss-Newton
        
        source_points:  nx3 matrix of n 3D points
        dest_points: nx3 matrix of n 3D points, which have been obtained by some rigid deformation of 'source_points'
        initial: 1x6 matrix, denoting alpha, beta, gamma (the Euler angles for rotation and tx, ty, tz (the translation along three axis). 
                    this is the initial estimate of the transformation between 'source_points' and 'dest_points'
        loop: start with zero, to keep track of the number of times it loops, just a very crude way to control the recursion            
                    
        """
        
        J = []
        e = []
        
        for i in range (0,dest_points.shape[0]-1):
            
            #print dest_points[i][3],dest_points[i][4],dest_points[i][5]
            dx = dest_points[i][0]
            dy = dest_points[i][1]
            dz = dest_points[i][2]
            
            sx = source_points[i][0]
            sy = source_points[i][1]
            sz = source_points[i][2]
            
            alpha = initial[0][0]
            beta = initial[1][0]
            gamma = initial[2][0]
            tx = initial[3][0]        
            ty = initial[4][0]
            tz = initial[5][0]
            #print alpha
            
            a1 = (-2*beta*sx*sy) - (2*gamma*sx*sz) + (2*alpha*((sy*sy) + (sz*sz))) + (2*((sz*dy) - (sy*dz))) + 2*((sy*tz) - (sz*ty))
            a2 = (-2*alpha*sx*sy) - (2*gamma*sy*sz) + (2*beta*((sx*sx) + (sz*sz))) + (2*((sx*dz) - (sz*dx))) + 2*((sz*tx) - (sx*tz))
            a3 = (-2*alpha*sx*sz) - (2*beta*sy*sz) + (2*gamma*((sx*sx) + (sy*sy))) + (2*((sy*dx) - (sx*dy))) + 2*((sx*ty) - (sy*tx))
            a4 = 2*(sx - (gamma*sy) + (beta*sz) +tx -dx)
            a5 = 2*(sy - (alpha*sz) + (gamma*sx) +ty -dy)
            a6 = 2*(sz - (beta*sx) + (alpha*sy) +tz -dz)
            
            _residual = (a4*a4/4)+(a5*a5/4)+(a6*a6/4)
            
            _J = np.array([a1, a2, a3, a4, a5, a6])
            _e = np.array([_residual])
            
            J.append(_J)
            e.append(_e)
            
        jacobian = np.array(J)
        residual = np.array(e)
        
        update = -np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(jacobian),jacobian)),np.transpose(jacobian)),residual)
        
        #print update, initial
        
        initial = initial + update
        
#        print( np.transpose(initial))
        
        loop = loop + 1
        
        if(loop < self.max_loop):  # here lies the control variable, control the number of iteration from here
        
            return self.icp_point_to_point_lm(source_points,dest_points,initial, loop)
        else:
            return initial
            
if __name__ == '__main__' :

    fileOriginal = '/home/kartik/arm_tracking/trajectory_planning/icp/data/original.xyz'
    deformed = '/home/kartik/arm_tracking/trajectory_planning/icp/data/deformed.xyz'
    
    source_points = read_file_original(fileOriginal)
    dest_points_et_normal = read_file_deformed(deformed)
    
    initial = np.array([[0.01], [0.05], [0.01], [0.001], [0.001], [0.001]])    
    icp_point_to_point_lm(source_points,dest_points_et_normal,initial,0)
