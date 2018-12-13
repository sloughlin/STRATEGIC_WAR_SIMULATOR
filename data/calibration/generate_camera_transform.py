#!/usr/bin/env python
import rospy
import numpy as np
import json
import sys


with open(sys.argv[1]) as f:
    calibration_data = json.load(f)


constraints_A = np.array([0,0,0,0,0,0,0,0,0,0,0,0]) 
rhs_b = np.array([])

for data in calibration_data['4']:
    constraints_A = np.vstack([constraints_A, 
		      np.array([data['kinect_frame']['x'], 
                                data['kinect_frame']['y'], 
                                data['kinect_frame']['z'], 1, 0,0,0,0,0,0,0,0])])
    constraints_A = np.vstack([constraints_A, 
		      np.array([0,0,0,0, 
                                data['kinect_frame']['x'], 
                                data['kinect_frame']['y'], 
                                data['kinect_frame']['z'], 1, 0,0,0,0])])
    constraints_A = np.vstack([constraints_A, 
		      np.array([0,0,0,0,0,0,0,0,data['kinect_frame']['x'], 
                                data['kinect_frame']['y'], 
                                data['kinect_frame']['z'], 1])])

    rhs_b = np.append(rhs_b, data['robot_frame']['x'])
    rhs_b = np.append(rhs_b, data['robot_frame']['y'])
    rhs_b = np.append(rhs_b, data['robot_frame']['z'])


for data in calibration_data['3']:
    constraints_A = np.vstack([constraints_A, 
		      np.array([data['kinect_frame']['x'], 
                                data['kinect_frame']['y'], 
                                data['kinect_frame']['z'], 1, 0,0,0,0,0,0,0,0])])
    constraints_A = np.vstack([constraints_A, 
		      np.array([0,0,0,0, 
                                data['kinect_frame']['x'], 
                                data['kinect_frame']['y'], 
                                data['kinect_frame']['z'], 1, 0,0,0,0])])
    constraints_A = np.vstack([constraints_A, 
		      np.array([0,0,0,0,0,0,0,0,data['kinect_frame']['x'], 
                                data['kinect_frame']['y'], 
                                data['kinect_frame']['z'], 1])])

    rhs_b = np.append(rhs_b, data['robot_frame']['x'])
    rhs_b = np.append(rhs_b, data['robot_frame']['y'])
    rhs_b = np.append(rhs_b, data['robot_frame']['z'])



constraints_A = np.delete(constraints_A, (0), axis=0)

(solution_vector, _, _, _) = np.linalg.lstsq(constraints_A, rhs_b)

solution_matrix = np.vstack([np.reshape(solution_vector, (3,4)),[0,0,0,1]])


np.save('calibration_matrix.npy',solution_matrix)
