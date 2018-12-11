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

h1_offset_x = -0.121
h1_offset_y = 0.144
board_square_size = 0.057

grab_height = 0.0
translate_height = 0.2

home_cords = np.array([0.4, -0.4, 0.2])

normal1_orientation_quaternion = np.array([0,1,0,0])
normal2_orientation_quaternion = np.array([0.707,0,0.707,0])

quaternion_toggle = False

magic_number = -0.07

capture_index = 0
capture_rows = 5

queen_position = np.array([7, -2])


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

def calculate_target(chessboard_pose, z_rotation, board_x, board_y):
    return np.array([ chessboard_pose[0] + (h1_offset_x + board_x * board_square_size) * np.cos(z_rotation) + (h1_offset_y + board_y * board_square_size) * np.sin(z_rotation), 
                      chessboard_pose[1] + (h1_offset_y + board_y * board_square_size) * np.cos(z_rotation) - (h1_offset_x + board_x * board_square_size) * np.sin(z_rotation)])



def handle_chess_piece_move(req): 
    global capture_index
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
    z_rotation = euler_pose[2] + magic_number


    print "camera_to_robot_tf"
    print camera_to_robot_tf

    chess_board_pose_robot_cords = np.matmul(camera_to_robot_tf, [temp_chess_board_pose.position.x, temp_chess_board_pose.position.y, temp_chess_board_pose.position.z, 1])
    print 'chess_board_pose_robot_cords'
    print chess_board_pose_robot_cords


    rospy.wait_for_service('cartesian_robot_move')
    rospy.wait_for_service('actuate_robot_gripper')
    try: 
        cartesian_robot_move = rospy.ServiceProxy('cartesian_robot_move', CartesianRobotMove)
        actuate_robot_gripper = rospy.ServiceProxy('actuate_robot_gripper', ActuateRobotGripper)

        temp_quat = normal1_orientation_quaternion
        # if quaternion_toggle == True: 
        #     quaternion_toggle = False
        #     temp_quat = normal1_orientation_quaternion
        # else:
        #     quaternion_toggle = False
        #     temp_quat = normal2_orientation_quaternion

        if(req.get_extra_queen):
            # req.start is current piece to promote
            # req.end is target for queen
            capture_start = calculate_target(chess_board_pose_robot_cords, z_rotation, req.start_x, req.start_y)
            capture_target = calculate_target(chess_board_pose_robot_cords, z_rotation, 3 + capture_index % capture_rows, 9 + np.floor(capture_index /capture_rows))
            capture_index = capture_index + 1

            start_target = calculate_target(chess_board_pose_robot_cords, z_rotation, queen_position[0], queen_position[1])
            end_target = capture_start(chess_board_pose_robot_cords, z_rotation, req.end_x, req.end_y)


            cartesian_robot_move(capture_start[0], capture_start[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(capture_start[0], capture_start[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(0.0)
            cartesian_robot_move(capture_start[0], capture_start[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(capture_target[0], capture_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(capture_target[0], capture_target[1], grab_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(100.0)
            cartesian_robot_move(capture_target[0], capture_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])


            cartesian_robot_move(start_target[0], start_target[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(start_target[0], start_target[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(0.0)
            cartesian_robot_move(start_target[0], start_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(end_target[0], end_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(end_target[0], end_target[1], grab_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(100.0)
            cartesian_robot_move(end_target[0], end_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(home_cords[0], home_cords[1], home_cords[2], temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

        elif(req.capture_piece):
            # req.start is current piece doing the taking
            # req.end is target for capture
            capture_start = calculate_target(chess_board_pose_robot_cords, z_rotation, req.end_x, req.end_y)
            capture_target = calculate_target(chess_board_pose_robot_cords, z_rotation, 3 + capture_index % capture_rows, 9 + np.floor(capture_index /capture_rows))

            start_target = calculate_target(chess_board_pose_robot_cords, z_rotation, req.start_x, req.start_y)
            end_target = capture_start

            capture_index = capture_index + 1


            cartesian_robot_move(capture_start[0], capture_start[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(capture_start[0], capture_start[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(0.0)
            cartesian_robot_move(capture_start[0], capture_start[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(capture_target[0], capture_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(capture_target[0], capture_target[1], grab_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(100.0)
            cartesian_robot_move(capture_target[0], capture_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])


            cartesian_robot_move(start_target[0], start_target[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(start_target[0], start_target[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(0.0)
            cartesian_robot_move(start_target[0], start_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(end_target[0], end_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(end_target[0], end_target[1], grab_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(100.0)
            cartesian_robot_move(end_target[0], end_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(home_cords[0], home_cords[1], home_cords[2], temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

        elif(req.castle_right):
            # req.start does nothing
            # req.end does nothing
            rook_start = calculate_target(chess_board_pose_robot_cords, z_rotation, 7, 7)
            rook_target = calculate_target(chess_board_pose_robot_cords, z_rotation, 7, 4)
            king_start = calculate_target(chess_board_pose_robot_cords, z_rotation, 7, 3)
            king_target = calculate_target(chess_board_pose_robot_cords, z_rotation, 7, 5)


            cartesian_robot_move(king_start[0], king_start[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(king_start[0], king_start[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(0.0)
            cartesian_robot_move(king_start[0], king_start[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(king_target[0], king_target[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(king_target[0], king_target[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(100.0)
            cartesian_robot_move(king_target[0], king_target[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(rook_start[0], rook_start[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(rook_start[0], rook_start[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(0.0)
            cartesian_robot_move(rook_start[0], rook_start[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(rook_target[0], rook_target[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(rook_target[0], rook_target[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(100.0)
            cartesian_robot_move(rook_target[0], rook_target[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(home_cords[0], home_cords[1], home_cords[2], temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
           
        elif(req.castle_left):
            # req.start does nothing
            # req.end does nothing
            rook_start = calculate_target(chess_board_pose_robot_cords, z_rotation, 7, 0)
            rook_target = calculate_target(chess_board_pose_robot_cords, z_rotation, 7, 2)
            king_start = calculate_target(chess_board_pose_robot_cords, z_rotation, 7, 3)
            king_target = calculate_target(chess_board_pose_robot_cords, z_rotation, 7, 1)


            cartesian_robot_move(king_start[0], king_start[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(king_start[0], king_start[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(0.0)
            cartesian_robot_move(king_start[0], king_start[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(king_target[0], king_target[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(king_target[0], king_target[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(100.0)
            cartesian_robot_move(king_target[0], king_target[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(rook_start[0], rook_start[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(rook_start[0], rook_start[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(0.0)
            cartesian_robot_move(rook_start[0], rook_start[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(rook_target[0], rook_target[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(rook_target[0], rook_target[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(100.0)
            cartesian_robot_move(rook_target[0], rook_target[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(home_cords[0], home_cords[1], home_cords[2], temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
           
        elif(req.enpassent):
            # req.start is the piece taking
            # req.end is the piece to take
            capture_start = calculate_target(chess_board_pose_robot_cords, z_rotation, req.end_x, req.end_y)
            capture_target = calculate_target(chess_board_pose_robot_cords, z_rotation, 3 + capture_index % capture_rows, 9 + np.floor(capture_index /capture_rows))
            capture_index = capture_index + 1

            start_target = calculate_target(chess_board_pose_robot_cords, z_rotation, req.start_x, req.start_y)
            end_target = calculate_target(chess_board_pose_robot_cords, z_rotation, req.end_x - 1, req.end_y)

            cartesian_robot_move(capture_start[0], capture_start[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(capture_start[0], capture_start[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(0.0)
            cartesian_robot_move(capture_start[0], capture_start[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(capture_target[0], capture_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(capture_target[0], capture_target[1], grab_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(100.0)
            cartesian_robot_move(capture_target[0], capture_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])


            cartesian_robot_move(start_target[0], start_target[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(start_target[0], start_target[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(0.0)
            cartesian_robot_move(start_target[0], start_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(end_target[0], end_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(end_target[0], end_target[1], grab_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(100.0)
            cartesian_robot_move(end_target[0], end_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(home_cords[0], home_cords[1], home_cords[2], temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            
        else:
            # req.start is the start position of piece
            # req.end is the end position of piece

            start_target = calculate_target(chess_board_pose_robot_cords, z_rotation, req.start_x, req.start_y)
            end_target = calculate_target(chess_board_pose_robot_cords, z_rotation, req.end_x, req.end_y)

            cartesian_robot_move(start_target[0], start_target[1], translate_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(start_target[0], start_target[1], grab_height, temp_quat[0], temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(0.0)
            cartesian_robot_move(start_target[0], start_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

            cartesian_robot_move(end_target[0], end_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            cartesian_robot_move(end_target[0], end_target[1], grab_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])
            actuate_robot_gripper(100.0)
            cartesian_robot_move(end_target[0], end_target[1], translate_height, temp_quat[0],temp_quat[1], temp_quat[2], temp_quat[3])

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
