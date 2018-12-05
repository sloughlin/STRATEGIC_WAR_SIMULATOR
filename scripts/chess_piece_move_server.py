#!/usr/bin/env python
from chess_bot.srv import *
from apriltags2_ros.msg import *

import rospy
import os
import numpy as np
import tf
import threading
import Queue

configuration_matrix_filename = os.path.join(os.path.dirname(__file__), '..', 'data', 'calibration_matrix.npy')


camera_to_robot_tf = np.load(configuration_matrix_filename)

h1_offset_x = -0.133
h1_offset_y = 0.155
board_square_size = 0.056

home_cords = np.array([0.4, -0.4, 0.2])

normal1_orientation_quaternion = np.array([0,1,0,0])
normal2_orientation_quaternion = np.array([0.707,0,0.707,0])

quaternion_toggle = False


tag_detection_lock = threading.Lock()
queue = Queue.Queue(maxsize=1)
latest_chess_board_pose = None

def aprilTagsCallback(data):
    board_found = False
    temp_pose = None
    for detection in data.detections:
        if detection.id[0] == 2:
            board_found = True 
            temp_pose = detection.pose.pose.pose
    tag_detection_lock.acquire()
    try: 
        if queue.full():
            queue.get()
        queue.put(temp_pose)
    finally:
        tag_detection_lock.release()

def handle_chess_piece_move(req): 
    start_position = [req.start_x, req.start_y]
    end_position = [req.end_x, req.end_y]
    
    temp_chess_board_pose = []
    tag_detection_lock.acquire()
    try: 
        temp_chess_board_pose = queue.get()
        print temp_chess_board_pose
    finally:
        tag_detection_lock.release()
    quaternion_pose = (temp_chess_board_pose.orientation.x,
                       temp_chess_board_pose.orientation.y,
                       temp_chess_board_pose.orientation.z,
                       temp_chess_board_pose.orientation.w)
    euler_pose = np.array(tf.transformations.euler_from_quaternion(quaternion_pose))
    z_rotation = euler_pose[2]


    print "camera_to_robot_tf"
    print camera_to_robot_tf

    chess_board_pose_robot_cords = np.matmul(camera_to_robot_tf, [temp_chess_board_pose.position.x, temp_chess_board_pose.position.y, temp_chess_board_pose.position.z, 1])
    print 'chess_board_pose_robot_cords'
    print chess_board_pose_robot_cords

    start_target = np.array([chess_board_pose_robot_cords[0] + (h1_offset_x + req.start_x * board_square_size) * np.cos(z_rotation) 
                              + (h1_offset_y + req.start_y * board_square_size) * np.sin(z_rotation), 
                              chess_board_pose_robot_cords[1] + (h1_offset_y + req.start_y * board_square_size) * np.cos(z_rotation) 
                              + (h1_offset_x + req.start_x * board_square_size) * np.sin(z_rotation)])
    end_target = np.array([chess_board_pose_robot_cords[0] + (h1_offset_x + req.end_x * board_square_size) * np.cos(z_rotation) 
                              + (h1_offset_y + req.end_y * board_square_size) * np.sin(z_rotation), 
                              chess_board_pose_robot_cords[1] + (h1_offset_y + req.end_y * board_square_size) * np.cos(z_rotation) 
                              + (h1_offset_x + req.end_x * board_square_size) * np.sin(z_rotation)])

    print "start_target"
    print start_target
    print "end_target"
    print end_target
    
    rospy.wait_for_service('cartesian_robot_move')
    rospy.wait_for_service('actuate_robot_gripper')
    try: 
        cartesian_robot_move = rospy.ServiceProxy('cartesian_robot_move', CartesianRobotMove)
        actuate_robot_gripper = rospy.ServiceProxy('actuate_robot_gripper', ActuateRobotGripper)
        if(req.get_extra_queen):
            raise NotImplementedError()
        elif(req.capture_piece):
            raise NotImplementedError()
        elif(req.castle):
            raise NotImplementedError()
        else:
            temp_quat = normal1_orientation_quaternion
            # if quaternion_toggle == True: 
            #     quaternion_toggle = False
            #     temp_quat = normal1_orientation_quaternion
            # else:
            #     quaternion_toggle = False
            #     temp_quat = normal2_orientation_quaternion

            cartesian_robot_move(start_target[0], start_target[1], 0.2, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(start_target[0], start_target[1], 0, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(0.0)
            cartesian_robot_move(start_target[0], start_target[1], 0.2, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(end_target[0], end_target[1], 0.2, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(end_target[0], end_target[1], 0, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(100.0)
            cartesian_robot_move(end_target[0], end_target[1], 0.2, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(home_cords[0], home_cords[1], home_cords[2], temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
    except Exception as inst: 
        print(type(inst))    # the exception instance
        print(inst.args)     # arguments stored in .args
        print(inst)          # __str__ allows args to be printed directly,
    return 0
         

def chess_piece_move_server():
    rospy.init_node('chess_piece_move_server')

    rospy.Subscriber('tag_detections', AprilTagDetectionArray, aprilTagsCallback)
    
    s = rospy.Service('chess_piece_move', ChessPieceMove, handle_chess_piece_move)
    print "Ready for moves"
    rospy.spin()


if __name__ == "__main__":
    chess_piece_move_server()
