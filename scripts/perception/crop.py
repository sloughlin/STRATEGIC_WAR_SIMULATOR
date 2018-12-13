
import numpy as np
import cv2 as cv2
import argparse
import glob
from matplotlib import pyplot as plt
import sys
# import piece_recognition
# import enhance

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range


def crop(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    a, b = img_crop.shape
    # print(a, b)
    if a > 275:
        start = int((a-275)/2)
        end = int(a-(a-275)/2)
        img_crop = img_crop[start: end, :]
    if b > 275:
        start = int((b-275)/2)
        end = int(b-(b-275)/2)
        img_crop = img_crop[:, start:end]
    if img_crop.shape[0] > 275 or img_crop.shape[1]:
        img_crop = img_crop[:275, :275]
    return img_crop


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
	img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
	                        -min(0, x1), max(x2 - img.shape[1], 0),cv2.BORDER_REPLICATE)
	y2 += -min(0, y1)
	y1 += -min(0, y1)
	x2 += -min(0, x1)
	x1 += -min(0, x1)
	return img, x1, x2, y1, y2

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and (cv2.contourArea(cnt) > 1000 and cv2.contourArea(cnt) < img.shape[0] * img.shape[1]) and cv2.isContourConvex(cnt):
                    #print(cnt)
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

def crop_the_image(ir):
    # filename_ir = 'data/image_ir_rect_screenshot_07.12.2018.png'
    # filename_real = 'data/image_color_rect_screenshot_07.12.2018.png'
    # ir = cv2.imread(filename_ir)
    # img = cv2.imread(filename_real)

    # filename_empty = 'data/ir_empty_board.png'
    # # ir = cv2.imread(filename_empty)
    # ir = cv2.imread('data/kinect_pairs/image_ir_rect_screenshot_02.12.2018.png')
    # img = cv2.imread('data/kinect_pairs/image_color_rect_screenshot_02.12.2018.png')
    # ir = cv2.imread('data/kinect_pairs/image_ir_rect_screenshot_07.12.2018.png')
    # img = cv2.imread('data/kinect_pairs/image_color_rect_screenshot_07.12.2018.png')
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(gray,100, 255,cv2.THRESH_TOZERO)
    # cv2.imshow("ir", ir)
    # cv2.waitKey(0)
    out = ir.copy()

    squares = find_squares(ir)
    # square = max(cv2.contourArea(squares))


    # ratio = ir.shape[1] / ir.shape[1]
    # want h / w = ratio
    # keep h const
    # w = h / ratio
    # h, w, _ = img.shape
    # w = int(h / ratio)
    # # img[y1:y2, x1:x2, :]
    # img = img[:, int(w/4):(w + int(w/2))]
    # out = img.copy()


    # get largest square
    max_area = 0
    square = None
    for cnt in squares:
    	if cv2.contourArea(cnt) > max_area:
    		square = cnt
    		max_area = cv2.contourArea(cnt)

    square = np.array(square)
    # scale the transpose: 
    rect = cv2.minAreaRect(square)
    box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    #cv2.drawContours(ir,[box],0,(0,0,255),2)
    #cv2.imshow("ir",ir)
    #cv2.waitKey(0)

    cropped_ir = crop(ir, rect)
    # print(cropped_ir.shape)

    # cv2.imshow('cropped ir', cropped_ir)
    # cv2.waitKey(0)
    # cv2.imwrite('data/cropped_ir.png', cropped_ir)
    # cv2.imwrite('data/empty.png', cropped_ir)
    return cropped_ir

    # print(square)
    # scaler = np.array([[1.25, 0, 0, 0],[0, 1.25, 0, 0], [0, 0, 1.25, 0], [0, 0, 0, 1.25]])
    # square = square.T @ scaler
    # square = square.T
    # square = square.astype(np.int32)

    # for i in square:
    # 	i -= [175, 125]

    # rect = cv2.minAreaRect(square)
    # box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
    # box = np.int0(box)
    # cv2.drawContours(out,[box],0,(0,0,255),2)

    # print(square)
    # cv2.drawContours( img, [square], -1, (0, 255, 0), 3 )
    # cv2.imshow('squares', out)
    # cv2.waitKey(0)

    # crop image tp bounding box 
    # alpha = 2.0
    # beta = 2.0

    # cropped = crop(img, rect)
    # # w, h = cropped.shape[:2]
    # # center = (w / 2, h / 2)
    # # rotation_matrix = cv2.getRotationMatrix2D(center, 90, 1)
    # # cropped = cv2.warpAffine(cropped, rotation_matrix, (h, w))
    # # cropped = enhance.enhance(cropped, alpha, beta)
    # cv2.imshow('cropped', cropped)
    # cv2.waitKey(0)


    # cv2.imwrite('data/cropped.png', cropped)

    # blur = cv2.GaussianBlur(img,(5,5),0)
    # gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    # # img = piece_recognition.line_detection(cropped, gray)

    # squares = find_squares(img)

    # max_area = 0
    # square = None
    # for cnt in squares:
    # 	if cv2.contourArea(cnt) > max_area:
    # 		square = cnt
    # 		max_area = cv2.contourArea(cnt)

    # square = np.array(square)
    # scaler = np.array([[1.05, 0, 0, 0],[0, 1.05, 0, 0], [0, 0, 1.05, 0], [0, 0, 0, 1.05]])
    # square = square.T @ scaler
    # square = square.T
    # square = square.astype(np.int32)
    # for i in square:
    # 	i -= [175, 125]

    # rect = cv2.minAreaRect(square)
    # box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
    # box = np.int0(box)
    # cv2.drawContours(img,[box],0,(0,255,0),2)
    # cropped_no_ir = crop(img, rect)
    # cv2.imwrite('data/cropped_no_ir.png', cropped_no_ir)


    # # print(square)
    # cv2.drawContours( out, [square], -1, (0, 255, 0), 3 )
    # cv2.imshow("Comparison", out)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()


