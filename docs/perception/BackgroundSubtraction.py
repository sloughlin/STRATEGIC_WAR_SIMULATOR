import numpy as np
import cv2 as cv2
import argparse
import glob
import sys
import matplotlib.pyplot as plt

# images = glob.glob('data/kinect2/*.png')
# images = glob.glob('data/cropped.png')

def subtract_bg(img):
	# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG2()
	fgbg = cv2.createBackgroundSubtractorMOG2()

	# images = glob.glob('data/kinect1/image_raw_screenshot_30.11.2018.png')
	# images = glob.glob('data/kinect1/enhanced/*')

	for fname in images:

		img = cv2.imread(fname)

		frame = img

		fgmask = fgbg.apply(frame)
		cv2.imshow('frame',fgmask)
		k = cv2.waitKey(500) & 0xff
		if k == 27:
			break
	cv2.destroyAllWindows()