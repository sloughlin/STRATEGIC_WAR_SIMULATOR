#!/usr/bin/env python
from chess_bot.srv import *
import rospy
import intera_interface

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header
from sensor_msgs.msg import JointState

from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)
from intera_motion_msgs.msg import TrajectoryOptions
from geometry_msgs.msg import PoseStamped
import PyKDL
from tf_conversions import posemath


rospy.init_node('cartesian_robot_move_server')

limb = intera_interface.Limb('right')

gripper = intera_interface.Gripper('right_gripper')

def handle_actuate_robot_grasper(req):
    if(req.gripper > 100 or req.gripper < 0):
        rospy.logerr('Error: Gripper request ' + req.gripper + ' out of range [0,100]')
        return 1
    gripper.set_position(req.gripper)
    return 0


def handle_cartesian_robot_move(req):
    linear_speed = 0.2
    linear_accel = 0.2
    rotational_speed = 1.57
    rotational_accel = 1.57
    
    position = [req.pose_x, req.pose_y, req.pose_z]
    orientation = [req.orientation_x,
                   req.orientation_y,
                   req.orientation_z,
                   req.orientation_w]
    
    joint_angles = [];
    
    tip_name = 'right_hand'
    
    relative_pose = None 
    
    traj_options = TrajectoryOptions()
    traj_options.interpolation_type = TrajectoryOptions.CARTESIAN
    traj = MotionTrajectory(trajectory_options = traj_options, limb = limb)
    
    wpt_opts = MotionWaypointOptions(max_linear_speed=linear_speed,
                                     max_linear_accel=linear_accel,
                                     max_rotational_speed=rotational_speed,
                                     max_rotational_accel=rotational_accel,
                                     max_joint_speed_ratio=1.0)
    waypoint = MotionWaypoint(options = wpt_opts.to_msg(), limb = limb)
    
    joint_names = limb.joint_names()
    
    if joint_angles and len(joint_angles) != len(joint_names):
        rospy.logerr('len(joint_angles) does not match len(joint_names!)')
        print "error 1"
        return 1
    
    if (position is None and orientation is None
        and relative_pose is None):
        if joint_angles:
            # does Forward Kinematics
            waypoint.set_joint_angles(joint_angles, tip_name, joint_names)
        else:
            rospy.loginfo("No Cartesian pose or joint angles given. Using default")
            waypoint.set_joint_angles(joint_angles=None, active_endpoint=tip_name)
    else:
        endpoint_state = limb.tip_state(tip_name)
        if endpoint_state is None:
            rospy.logerr('Endpoint state not found with tip name %s', tip_name)
            print 'error 2'
            return 2
        pose = endpoint_state.pose
    
        if relative_pose is not None:
            if len(relative_pose) != 6:
                rospy.logerr('Relative pose needs to have 6 elements (x,y,z,roll,pitch,yaw)')
                print "error 3"
                return 3
            # create kdl frame from relative pose
            rot = PyKDL.Rotation.RPY(relative_pose[3],
                                     relative_pose[4],
                                     relative_pose[5])
            trans = PyKDL.Vector(relative_pose[0],
                                 relative_pose[1],
                                 relative_pose[2])
            f2 = PyKDL.Frame(rot, trans)
            # and convert the result back to a pose message
            if in_tip_frame:
              # end effector frame
              pose = posemath.toMsg(posemath.fromMsg(pose) * f2)
            else:
              # base frame
              pose = posemath.toMsg(f2 * posemath.fromMsg(pose))
        else:
            if position is not None and len(position) == 3:
                pose.position.x = position[0]
                pose.position.y = position[1]
                pose.position.z = position[2]
            if orientation is not None and len(orientation) == 4:
                pose.orientation.x = orientation[0]
                pose.orientation.y = orientation[1]
                pose.orientation.z = orientation[2]
                pose.orientation.w = orientation[3]
        poseStamped = PoseStamped()
        poseStamped.pose = pose
    
        if not joint_angles:
            # using current joint angles for nullspace bais if not provided
            joint_angles = limb.joint_ordered_angles()
            waypoint.set_cartesian_pose(poseStamped, tip_name, joint_angles)
        else:
            waypoint.set_cartesian_pose(poseStamped, tip_name, joint_angles)



    rospy.loginfo('Sending waypoint: \n%s', waypoint.to_string())

    traj.append_waypoint(waypoint.to_msg())

    result = traj.send_trajectory(timeout=None)
    if result is None:
        rospy.logerr('Trajectory FAILED to send')
    

    if result.result:
        rospy.loginfo('Motion controller successfully finished the trajectory!')
        return 0
    else:
        rospy.logerr('Motion controller failed to complete the trajectory with error %s',
                     result.errorId)
        return 4






def cartesian_robot_move_server():
    robot_state = intera_interface.RobotEnable()
    robot_state.enable()
    limb.move_to_neutral()
    
    s = rospy.Service('cartesian_robot_move', CartesianRobotMove, handle_cartesian_robot_move)
    g = rospy.Service('actuate_robot_gripper', ActuateRobotGripper, handle_actuate_robot_grasper)
    print "Ready for movement requests"
    rospy.spin()


if __name__ == "__main__":
    cartesian_robot_move_server()
