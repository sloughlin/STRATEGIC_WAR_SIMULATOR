#!/usr/bin/env python

import numpy as np
import cv2 as cv2
import argparse
import glob
import sys
import matplotlib.pyplot as plt
# Camera Calibration methods:
# from opencvcalibrationfixes.OpenCVCalibrationFixes import *


PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="path to the input image")
#args = vars(ap.parse_args())
#img = cv2.imread(args["image"])

#img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#cv2.imshow("HSV", img)
#cv2.waitKey(0)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('image_color_rect_screenshot_07.12.2018.png')
# images = glob.glob('data/kinect1/image_raw_screenshot_30.11.2018.png')
# images = glob.glob('data/kinect1/enhanced/*')

for fname in images:

    img = cv2.imread(fname)
    # NEED TO CROP IPHONE IMAGES
    # img = imcrop(img)
    blur = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(gray,100, 255,cv2.THRESH_TOZERO)
    # gray = thresh
    
    #cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
    #imS = cv2.resize(gray, (1500, 2000))   # Resize image
    cv2.imshow("output", gray)                            # Show image
    cv2.waitKey(0)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,7),None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)
    # print(corners)
    # ret = True
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,7), corners,ret)
        # cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
        # imS = cv2.resize(img, (1500, 2000))   # Resize image
        # cv2.imshow('output',imS)
        cv2.imshow("Corners", img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
# np.savez("computervision", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# print(ret, mtx, dist, rvecs, tvecs)

# img = cv2.imread('data/kinect1/image_raw_screenshot_30.11.2018.png')
# img = cv2.imread('data/iphone/IMG_4335.JPG')
# img = cv2.imread('data/kinect2/image_color_screenshot_02.12.2018.png')
img = cv2.imread('data/cropped.png')

h,  w = img.shape[:2]

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# print(roi)
# Kinect 1 camera params:
# fx = 526.37013657
# fy = 526.37013657
# cx = 313.68782938
# cy = 259.01834898
# newcameramtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# mtx = newcameramtx
# # distortion coefficients:
# dist = np.array([ 0.18126525, -0.39866885, 0.00000000, 0.00000000, 0.00000000 ])
# undistort
# mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
# dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

image_corr_, new_camera_matrix_ = undistort_image(newcameramtx, dist, img)
cv2.imshow("Corrected", image_corr_)
cv2.waitKey(0)
dst = image_corr_
# crop the image
# x, y, w, h = roi
# print(roi)
# dst = dst[y:y+h, x:x+w]

cv2.imwrite('calibresult.png', dst)

mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

    print("total error: ", mean_error/len(objpoints))

# New undistort from 3rd party:

# Kinect Images: 

# folder = 'data/kinect1'
# image_dist_ = cv2.imread('data/kinect1/image_raw_screenshot_30.11.2018.png')
# n_cb_points_ = (7, 7)
# image_postfix = '.png'
# folder = 'data/kinect1/enhanced'
# image_dist_ = cv2.imread('data/kinect1/enhanced/enhanced_image1.png')
# n_cb_points_ = (7, 7)
# image_postfix = '.png'

# # Iphone images: 

# # folder = 'data/iphone'
# # image_dist_ = cv2.imread('data/iphone/IMG_4352.JPG')
# # n_cb_points_ = (7, 7)
# # image_postfix = '.JPG'

# # cv2.imshow("Image", image_dist_)
# # cv2.waitKey()

# folder = 'data/kinect2'
# image_dist_ = cv2.imread('data/kinect2/image_color_screenshot_04.12.2018.png')
# n_cb_points_ = (7, 7)
# image_postfix = '.png'


# for with_p, ttl in [(False, 'without partial checkerboards'), (True, 'with partial checkerboards')]:
#     plt.figure()
#     camera_matrix_, dist_coeffs_, _ = find_radial_distortion(folder, 
#         n_cb_points_, 
#         image_postfix, 
#         with_partial=with_p)
#     image_corr_, new_camera_matrix_ = undistort_image(camera_matrix_, 
#         dist_coeffs_, 
#         image_dist_)
#     plt.clf()
#     plt.title(ttl)
#     plt.imshow(image_corr_)

# plt.figure(3)
# plt.subplot(122)
# plt.cla()
# plt.suptitle('distorted image')
# plt.imshow(image_dist_)

# plt.show()


    # mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    # dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
     
    # # crop the image
    # x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]
    # cv2.imwrite('calibresult.png',dst)

