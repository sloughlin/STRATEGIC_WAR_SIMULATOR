#!/usr/bin/env python
import rospy
import roslib
import cv2
import piece_recognition
import Queue
import threading
import numpy as np
from sensor_msgs.msg import Image
from chess_bot.srv import *
from cv_bridge import CvBridge, CvBridgeError


ir_image_lock = threading.Lock()
queue = Queue.Queue(maxsize=1)

def callback(ros_data):
        image_np = None
	bridge = CvBridge()
        #print(ros_data)
	try:
		image_np = bridge.imgmsg_to_cv2(ros_data)
		print("image received")
	except CvBridgeError as e:
		print(e)
	# np_arr = np.fromstring(ros_data.data, np.uint8)
 #        print(ros_data.data) 
	# image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR) #May need different mode?
	#image_np = image_np.convertTo(CV_32F)
        image_np = np.array(image_np)
        image_np[image_np > 10000] = 10000
        image_np = image_np.astype('float32')/10000
        #print(image_np)
        #print(np.max(image_np))
	#cv2.imshow("garbage?", image_np)
	#cv2.waitKey(0)
	#cv2.normalize(image_np, image_np, 0, 1, cv2.NORM_MINMAX)
	image_np = image_np * 255

        image_np = image_np.astype('uint8')
        # image_np is normalized as a float between 0 and 1
	ir_image_lock.acquire()
	try:
		if queue.full():
			queue.get()
		queue.put(image_np)
	finally:
		ir_image_lock.release()


def handle_detect_pieces(req):
        print('hello')
	board_state = req.data

	#get image
	ir_image_lock.acquire()
	try:
		temp_ir_image = queue.get()
		print("Acquired image")
	finally:
		ir_image_lock.release()

	
	#run piece recognition
	images_2d = piece_recognition.detect_pieces(temp_ir_image)

	#board_state is y-major 1D array
	images_2d = np.array(images_2d)
	images = []
	for i in range(images_2d.shape[0]):
		images.append(images[i])
	#Write image name, label to CSV:
	labels_csv = open(os.path.join(os.path.dirname(__file__), '..', '..', 'data','image_labels.csv') , "ra+")
	#Generate random image name, check if exists in csv
	#if exists, generate new name until not exists
	#write name, label to csv
	#write file to train/

	#random numbers for names (make ints so names look better):
        print('searching for name')
	names = int(np.random.choice(10000, size=10000, replace=False))
	for i in range(len(images)):
		image_name = str(np.random.choice(names))
		while(image_name in labels_csv):
			image_name = str(np.random.choice(names))
		labels_csv.append(image_name, board_state[i])
		cv2.imwrite(os.path.join(os.path.dirname(__file__), '..', '..', 'data','train', '{}.png'.format(image_name)))
	# subscriber = rospy.Subscriber(ir_topic, image_raw, callback, queue_size=1)

	print("Returning Board State: ")
	print(req.data)

	return handle_detect_pieces_response(req.data)

def detect_pieces_server():
	rospy.init_node('handle_detect_pieces_server')

	ir_topic = '/kinect2/sd/image_ir_rect'
	rospy.Subscriber(ir_topic, Image, callback, queue_size=1)

	s = rospy.Service('detect_pieces', BoardState, handle_detect_pieces)
	print("Ready to receive board state")
	rospy.spin()


if __name__ == "__main__":
	detect_pieces_server()
