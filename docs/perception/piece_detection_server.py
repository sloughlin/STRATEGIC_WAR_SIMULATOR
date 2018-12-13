#!/usr/bin/env python
import rospy
import roslib
import cv2
import piece_recognition
import Queue
import threading
import numpy as np


ir_image_lock = threading.Lock()
queue = Queue.Queue(maxsize=1)

def callback(ros_data):
	# if VERBOSE:
	# 	print("received image of type {}".format(ros_data.format))

	np_arr = np.fromstring(ros_data.data, np.uint8)
	image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR) #May need different mode?

	ir_image_lock.acquire()

	try:
		if queue.full():
			queue.get()
		queue.put(image_np)
	finally:
		ir_image_lock.release()


def handle_detect_pieces(req):
	board_state = req.board_state

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
	images_2d = np.array(images_2d).T
	images = []
	for i in range(images_2d.shape[0]):
		images.append(images[i])
	#Write image name, label to CSV:
	labels_csv = open("image_labels.csv", "ra+")
	#Generate random image name, check if exists in csv
	#if exists, generate new name until not exists
	#write name, label to csv
	#write file to train/

	#random numbers for names (make ints so names look better):
	names = int(np.random.choice(10000, size=10000, replace=False))
	for i in range(len(images)):
		image_name = str(np.random.choice(names))
		while(image_name in labels_csv):
			image_name = str(np.random.choice(names))
		labels_csv.append(image_name, board_state[i])
		cv2.imwrite("train/{}.png".format(image_name))
	# subscriber = rospy.Subscriber(ir_topic, image_raw, callback, queue_size=1)

	print("Returning Board State: ")
	print(req.board_state)

	return handle_detect_pieces_response(req.board_state)
def detect_pieces_server():
	rospy.init_node('handle_detect_pieces_server')

	ir_topic = '/kinect2/sd/image_ir_rect'
	rospy.Subscriber(ir_topic, image_raw, callback, queue_size=1)
	if VERBOSE:
		print("subscribed to {}".format(ir_topic))

	s = rospy.Service('detect_pieces', detect_pieces, handle_detect_pieces)
	print("Ready to receive board state")
	rospy.spin()


if __name__ == "__main__":
	detect_pieces_server()