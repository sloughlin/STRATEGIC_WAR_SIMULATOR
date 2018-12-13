import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
import operator
#from GaussianMixture import GMM
#from sklearn.mixture import GaussianMixture
#from sklearn.cluster import KMeans
from glob import glob
from crop import crop_the_image

# from enhance import enhance

# PY3 = sys.version_info[0] == 3

# if PY3:
#     xrange = range


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0], line1[1]
    rho2, theta2 = line2[0], line2[1]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]

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

    return img_crop

def draw_lines(img, lines, color=(0, 255, 0)):
	if lines is None or len(lines) == 0:
		return img
	# print(lines.shape)
	color = 100
	a,b = lines.shape
	for i in range(a):
	    rho = lines[i][0]
	    theta = lines[i][1]
	    a = np.cos(theta)
	    b = np.sin(theta)
	    x0, y0 = a*rho, b*rho
	    pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
	    pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )

	    cv2.line(img, pt1, pt2, (0, color, 0), 2, cv2.LINE_AA)
	    color +=25
	return img

def get_lines(img):    
	blur = cv2.GaussianBlur(img,(5,5),0)

	if len(img.shape) > 2:
	
		gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
	else:
		gray = img
	# thresh_test(gray)

	# ret,thresh = cv2.threshold(gray,100, 255,cv2.THRESH_TOZERO)
	# cv2.imshow("thresh", thresh)
	# cv2.waitKey(0)

	# img = line_detection(img, gray)

	# dst = cv2.Canny(gray, 50, 300)
	dst = cv2.Canny(gray, 50, 200)
	# cv2.imshow('dst', dst)
	# cv2.waitKey(0)
	lines= cv2.HoughLines(dst, 1, np.pi/180.0, 100, np.array([]), 0, 0)
	if lines is not None:
		lines = lines.reshape((lines.shape[0], lines.shape[2]))
	return lines
def equalize_hist(img):
	if len(img.shape) > 2:
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray = img
	# create a CLAHE object (Arguments are optional).
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(img.astype('uint8'))
	# cv2.imshow("Contrast", cl1)
	# cv2.waitKey(0)
	# cl1 = cv2.equalizeHist(img)
	return cl1
def run_sobel(img):
	scale = 1
	delta = 0
	ddepth = cv2.CV_16S

	src = cv2.GaussianBlur(img, (3, 3), 0)


	gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


	grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
	# Gradient-Y
	# grad_y = cv2.Scharr(gray,ddepth,0,1)
	grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)


	abs_grad_x = cv2.convertScaleAbs(grad_x)
	abs_grad_y = cv2.convertScaleAbs(grad_y)


	grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


	# cv2.imshow("sobel", grad)
	# cv2.waitKey(0)
	return grad
def resize(im, desired_size=32):
	old_size = im.shape[:2] # old_size is in (height, width) format
	ratio = float(desired_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])
	# new_size should be in (width, height) format
	im = cv2.resize(im, (new_size[1], new_size[0]))
	delta_w = desired_size - new_size[1]
	delta_h = desired_size - new_size[0]
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)
	color = np.mean(im)
	new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
	    value=color)
	return new_im

def fft(img, img_back):


	red_fourier_subtraction = (np.fft.ifft2(np.fft.fft2(img[:,:,0].astype(np.float64)) - np.fft.fft2(img_back[:,:,0].astype(np.float64)))).real
	#green_fourier_subtraction = (np.fft.ifft2(np.fft.fft2(img[:,:,1].astype(np.float64)) - np.fft.fft2(img_back[:,:,1].astype(np.float64)))).real
	#blue_fourier_subtraction = (np.fft.ifft2(np.fft.fft2(img[:,:,2].astype(np.float64)) - np.fft.fft2(img_back[:,:,2].astype(np.float64)))).real

	# print (np.max(red_fourier_subtraction - green_fourier_subtraction))

	red_fourier_subtraction = (((red_fourier_subtraction - np.mean(red_fourier_subtraction)) / np.square(np.std(red_fourier_subtraction))) + 1) /2
	# green_fourier_subtraction = (((green_fourier_subtraction - np.mean(green_fourier_subtraction)) / np.square(np.std(green_fourier_subtraction))) + 1) /2
	# blue_fourier_subtraction = (((blue_fourier_subtraction - np.mean(blue_fourier_subtraction)) / np.square(np.std(blue_fourier_subtraction))) + 1) /2

	red_fourier_subtraction = (red_fourier_subtraction - np.min(red_fourier_subtraction))/(np.max(red_fourier_subtraction) - np.min(red_fourier_subtraction))
	# green_fourier_subtraction = (green_fourier_subtraction - np.min(green_fourier_subtraction))/(np.max(green_fourier_subtraction) - np.min(green_fourier_subtraction))
	# blue_fourier_subtraction = (blue_fourier_subtraction - np.min(blue_fourier_subtraction))/(np.max(blue_fourier_subtraction) - np.min(blue_fourier_subtraction))


	# final_fourier_img = np.zeros([len(red_fourier_subtraction),len(red_fourier_subtraction[1]), 3])
	# final_fourier_img[:,:,0] = red_fourier_subtraction
	# final_fourier_img[:,:,1] = green_fourier_subtraction
	# final_fourier_img[:,:,2] = blue_fourier_subtraction
	return red_fourier_subtraction

def detect_pieces(ir):
	# def detect_pieces(img):
	#GIVEN: xy indexed array of board state, ir_rect image

	# filename = 'data/kinect1/image_raw_screenshot_34.11.2018.png'
	# filename = 'data/kinect2/image_color_screenshot_02.12.2018.png'
	# filename = 'data/cropped.png'
	# filename = 'data/cropped_ir.png'
	# filename = 'data/empty.png'
	# ir = cv2.imread('data/image_ir_rect_screenshot_07.12.2018.png')

	img = crop_the_image(ir)
	file_empty = 'data/empty.png'
	# filename = 'data/cropped_no_ir.png'

	# img = cv2.imread(filename)
	empty_img = cv2.imread(file_empty)

	transformed = fft(img, empty_img)
	# print(transformed.shape)


	# cv2.imshow("New transform", img)
	# cv2.waitKey(0)

	out = img.copy() #for line drawing
	equalized = equalize_hist(out)
	lines = get_lines(equalized)
	# out = draw_lines(img, lines)
	# cv2.imshow("Out", out)
	# cv2.waitKey(0)

	internal_lines = lines.copy()

	#####################Make External Lines####################################

	# np.append(internal_lines, minvert)
	# np.append(internal_lines, minhoriz)
	# np.append(internal_lines, maxvert)
	# np.append(internal_lines, maxhoriz)

	# internal_lines = np.array(internal_lines)
	internal_lines = np.array(sorted(internal_lines, key=operator.itemgetter(0)))
	# out = draw_lines(out, internal_lines)
	# cv2.imshow("lines", out)
	# cv2.waitKey(0)

	####################Remove duplicates:###########################################
	rho_min = 10
	theta_min = 0.5

	a, b = internal_lines.shape
	for i in range(a):
		for j in range(a):
			if i != j: 
				if(abs(abs(internal_lines[i][0])- abs(internal_lines[j][0])) <= rho_min and (abs(internal_lines[i][1]- internal_lines[j][1]) <= theta_min or abs(abs(internal_lines[i][1]- internal_lines[j][1]) - np.pi) <= theta_min)):
				#delete the one with worse error
					i_error = min(internal_lines[i][1] - 0, internal_lines[i][1] - np.pi / 2, internal_lines[i][1] - np.pi)
					j_error = min(internal_lines[j][1] - 0, internal_lines[j][1] - np.pi / 2, internal_lines[i][1] - np.pi)

					if i_error > j_error:
						# np.delete(internal_lines, i)
						internal_lines[i] = [-1, -1]
					else:
						# np.delete(internal_lines, j)
						internal_lines[j] = [-1, -1]

	# print(internal_lines)
	internal_lines = np.array([x for x in internal_lines if x[0] != -1 and x[1] != -1])
	# internal_lines = internal_lines.reshape((internal_lines.shape[0], internal_lines.shape[2]))


	#lines ordered 1st horiz, 1st vert, 2nd horiz, 2nd vert, ...
	####################3#Split into horizontal, vertical lines:##############################
	minvert = (0,0)
	minhoriz = (0, np.pi/2)

	maxvert = ( img.shape[1], 0)
	maxhoriz = (img.shape[1], np.pi/2)
	vertical = []
	horizontal = []
	a, b = internal_lines.shape
	for i in range(a):
		if abs(internal_lines[i][1] - 0) < 0.5 or abs(internal_lines[i][1] - np.pi) < 0.5:
			vertical.append(internal_lines[i])
		elif abs(internal_lines[i][1] -np.pi/2) < 0.5:
			# print(internal_lines[i][1])
			horizontal.append(internal_lines[i])
	# vertical.append(maxvert)
	# horizontal.append(maxhoriz)

	def takeFirst(elem):
	    return abs(elem[0])

	vertical = sorted(vertical, key=takeFirst)
	horizontal = sorted(horizontal, key=takeFirst)

	if abs(vertical[0][0] - 0) > rho_min:
		vertical.insert(0,(1, 0))
	if abs(vertical[-1][0] - img.shape[1]) > rho_min:
		vertical.append((img.shape[1]-1, 0))


	if abs(horizontal[0][0] - 0) > rho_min:
		horizontal.insert(0, (1, np.pi/2))
	if abs(horizontal[-1][0] - img.shape[1]) > rho_min:
		horizontal.append((img.shape[1]-1, np.pi/2))


	out = draw_lines(out, np.array(vertical))
	out = draw_lines(out, np.array(horizontal))
	# out = draw_lines(out, internal_lines)
	# cv2.imshow("out", out)
	# cv2.waitKey(0)



	blur = cv2.GaussianBlur(img,(5,5),0)
	empty_blur = cv2.GaussianBlur(empty_img,(5,5),0)

	# transformed = np.fft.ifft2(np.fft.fft2(blur) - np.fft.fft2(empty_blur)).real
	# transformed = (transformed + 1)/2
	# cv2.imshow("sakdsnf", transformed)
	# cv2.waitKey(0)

	# cv2.imshow('difference', abs(blur - empty_blur))
	# cv2.waitKey(0)


	#Equalize image histogram
	img = equalize_hist(img)

	# GaussianMixture.GaussianMixtureModel(img, plot=True)
	# pca = PCA(n_components=30)
	# scaler = StandardScaler()
	gmm = GaussianMixture(n_components=2)
	img = cv2.GaussianBlur(img,(11,11),0)
	empty_img = equalize_hist(empty_img)
	empty_blur = cv2.GaussianBlur(empty_img, (15, 15), 0)


	# thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	#           cv2.THRESH_BINARY,11,2)
	# cv2.imshow("Blurred", thresh)
	# cv2.waitKey(0)
	#Black piece detector:
	# params_black = cv2.SimpleBlobDetector_Params()
	# # params_black.minThreshold = 10
	# # params_black.maxThreshold = 150

	# ver = (cv2.__version__).split('.')
	# if int(ver[0]) < 3 :
	#     detector_black = cv2.SimpleBlobDetector(params_black)
	# else : 
	#     detector_black = cv2.SimpleBlobDetector_create(params_black)

	#White Piece Detector

	# orb = cv2.ORB_create()
	# kmeans = KMeans(n_clusters=2)
	chessboard_imgs = [[],[],[],[],[],[],[],[]]
	chessboard_vecs = [[],[],[],[],[],[],[],[]]
	chessboard = np.zeros((8, 8))
	chessboard_orig = [[],[],[],[],[],[],[],[]]

	vertical = np.array(vertical)
	horizontal = np.array(horizontal)
	a, _ = horizontal.shape
	b, _ = vertical.shape
	# print(a, b)
	for i in range(a-1):
		for j in range(b-1):
			intersections= []
			#get intersection of h[i], v[j]
			intersections.append(intersection(horizontal[i], vertical[j]))
			# cv2.circle(img, (intersections[0][0], intersections[0][1]), 2, (255, 0, 0), thickness=2)
			#get intersection of h[i], v[j+1]
			intersections.append(intersection(horizontal[i], vertical[j+1]))

			#get intersection of h[i+1], v[j]
			intersections.append(intersection(horizontal[i+1], vertical[j]))
			#get intersection of h[i+1], v[j+1]
			intersections.append(intersection(horizontal[i+1], vertical[j+1]))

			intersections = np.array(intersections)
			square = cv2.minAreaRect(intersections)
			box = cv2.boxPoints(square)
			box = np.array(box)
			# cv2.drawContours(img,[box.astype(int)],0,(0,0,255),2)
			cropped_square = crop(img, square)
			cropped_fft = crop(transformed, square)
			# cropped_fft, _, _ = cv2.split(cropped_fft)
			# cv2.imshow("fft",cropped_fft)
			# cv2.waitKey(0)
			#Check if img already grayscale:
			if len(cropped_fft.shape) > 2:
				gray = cv2.cvtColor(cropped_fft, cv2.COLOR_BGR2GRAY)
			else:
				gray = cropped_fft

			if len(cropped_square.shape) > 2:
				gray_orig = cv2.cvtColor(cropped_square, cv2.COLOR_BGR2GRAY)
			else:
				gray_orig = cropped_square
			gray = resize(gray)
			gray_orig = resize(gray_orig)

			# cv2.imshow("contrast", gray_orig)
			# cv2.waitKey(0)
			# cv2.imshow("?", abs(gray - np.mean(gray)) + )
			# x, y = gray.shape
			# kmeans.fit(gray.flatten().reshape((-1, 1)))
			# cluster_centers = kmeans.cluster_centers_
			# cluster_labels = kmeans.labels_
			# clustered = cluster_labels.reshape((32, 32))
			# print(clustered)
			# cv2.imshow("Clustered image", clustered)
			# cv2.waitKey(0)
			# plt.figure(figsize = (15,8))
			# plt.imshow(cluster_centers[cluster_labels].reshape(x, y))
			# laplacian = cv2.Laplacian(gray,cv2.CV_64F)
			# cv2.imshow("Laplacian", laplacian)
			# cv2.waitKey(0)

			# dx1_detector_kernel = np.ones((4,4))/16
			# dx2_detector_kernel = np.ones((4,4))/16
			# dy1_detector_kernel = np.ones((4,4))/16
			# dy2_detector_kernel = np.ones((4,4))/16
			# dx1_detector_kernel[2:, :] = -1/16
			# dx2_detector_kernel[:-2, :] = -1/16
			# dy1_detector_kernel[:, 2:] = -1/16
			# dy2_detector_kernel[:, :-2] = -1/16

			# post_gradient_detector = cv2.filter2D(gray, -1, dx1_detector_kernel)\
			# +cv2.filter2D(gray, -1, dy1_detector_kernel)

			# # means = np.mean(post_gradient_detector, axis=0)
			# # stddev = np.std(post_gradient_detector, axis=0)
			# # normalized = (post_gradient_detector - means ) / np.square(stddev) 
			# # normalized = (normalized + 1) *1 / 2
			# cv2.imshow("Sobel", post_gradient_detector)
			# cv2.waitKey(0)
			# gray2 = np.uint8(gray)
			# sobel = run_sobel(gray2)
			# cv2.imshow("Sobel", sobel)
			# cv2.waitKey(0)
			# blur = cv2.GaussianBlur(gray,(5,5),0)
			# blur = cropped_square
			# blur = np.uint8(blur)

			# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			# thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	  #           cv2.THRESH_BINARY,11,2)
			# cv2.imshow("thresh", th3)
			# cv2.waitKey(0)
			# cv2.imshow("actual", cropped_square)
			# cv2.waitKey(0)
			# empty = empty_squares[i][j]
			# print(i, j)
			# blur = abs(blur - empty)

			# Initiate STAR detector

			# find the keypoints with ORB
			# kp = orb.detect(blur,None)

			# compute the descriptors with ORB
			# kp, des = orb.compute(blur, kp)

			# draw only keypoints location,not size and orientation
			# img2 = cv2.drawKeypoints(cropped_fft,kp,color=(0,255,0), flags=0)
			# plt.imshow(img2),plt.show()


			# keypoints = detector_black.detect(blur)
			# if len(keypoints) != 0: 
			# 	chessboard[i][j] = 2
			# else:
			# 	inv = abs(255 - blur)
			# 	keypoints = detector_black.detect(inv)
			# 	if len(keypoints) != 0: 
			# 		chessboard[i][j] = 1
			
			# # # # # Draw detected blobs as red circles.


			# # # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
			# im_with_keypoints = cv2.drawKeypoints(cropped_square, kp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			# cv2.imshow("Keypoints", im_with_keypoints)
			# cv2.waitKey(0)

			# sift = cv2.xfeatures2d.SIFT_create()
			# kp = sift.detect(gray,None)
			# img=cv2.drawKeypoints(cropped_square,kp)

			# cv2.imshow("SIFT", cropped_square)
			# cv2.waitKey(0)


			# print(gray.shape)
			# GMM(gray, plot=True)
			chessboard_imgs[i].append(gray)

			chessboard_orig[i].append(gray_orig)
			#Generate set of empty squares:
			# name = "data/empty_board/empty{}{}.png".format(i, j)
			# cv2.imwrite(name, gray)

			# gray = gray.astype('float64')
			# # #Reduce image dimensionality with PCA:
			# gray = scaler.fit_transform(gray)
			# gray = pca.fit_transform(gray)

			gray = gray.flatten()
			# print(gray.shape)
			chessboard_vecs[i].append(gray)


	# chessboard_imgs = np.array(chessboard_imgs)
	# print(gmm.fit_predict(chessboard_imgs))
	# print(labels)
	# print(chessboard_orig[0])

	chessboard = chessboard[::-1]
	chessboard_imgs = chessboard_imgs[::-1]
	chessboard_orig = chessboard_orig[::-1]
	#NOT CORRECT RIGHT NOW
	# chessboard_vecs = np.array(chessboard_vecs[::-1]).reshape(1, len(chessboard_vecs[0])*len(chessboard_vecs[0]))
	chessboard_vecs = chessboard_vecs[::-1]

	return chessboard_imgs
	# print(chessboard_orig[-1])
	# for i in range(len(chessboard_orig)):
	# 	for j in range(len(chessboard_orig[i])):
	# 		cv2.imshow("i",chessboard_orig[i][j])
	# 		cv2.waitKey(0)

	# #IF NOT WORKING, THE SQUARES ARE NOT ALL THE SAME SIZE!!!!
	# chessboard = chessboard.flatten()
	# # print(np.array(chessboard_vecs).shape)
	# chessboard_vecs = np.array(chessboard_vecs)

	# # print(gmm.fit_predict(chessboard_vecs))
	# predictions = gmm.fit_predict(chessboard_vecs)

	# squares0 = []
	# squares1 = []
	# squares2 = []

	# for i in range(len(predictions)):
	# 	if predictions[i] == 0:
	# 		squares0.append(chessboard_imgs[i])
	# 	# elif predictions[i] == 2: 
	# 	# 	squares2.append(chessboard_imgs[i])
	# 	else:
	# 		squares1.append(chessboard_imgs[i])

	# means0 = np.mean(squares0, axis=0)
	# stddev0 = np.std(squares0, axis=0)
	# means1 = np.mean(squares1, axis=0)
	# stddev1 = np.std(squares1, axis=0)
	# normalized0 = (squares0 - means0 ) / np.square(stddev0) 
	# normalized1 = (squares1 - means1 ) / np.square(stddev1)
	# normalized0 = (normalized0 + 1) *1 / 2
	# normalized1 = (normalized1 + 1) * 1 / 2

	# kmeans = KMeans(n_clusters=2)
	# print(len(squares0))
	# print(len(squares1))
	# for i in range(len(squares0)):
	# 	# cv2.imshow("Fuck {}".format(i), normalized0[i])
	# 	cv2.imshow("Label 0", squares0[i])
	# 	cv2.waitKey(0)
	# 	cv2.imshow("Label 1", squares1[i])
	# 	cv2.waitKey(0)
		# cv2.imshow("Label 2", squares2[i])
		# cv2.waitKey(0)
		

		# cv2.imshow("Fuck actual {}".format(i), squares0[i])
		# cv2.imshow("Fuck 1 {}".format(i), normalized1[i])
		# cv2.waitKey(0)
	# train0 = [i.flatten() for i in normalized0]
	# train1 = [i.flatten() for i in normalized1]
	# kmeans.fit_predict(train0)
	# kmeans.fit_predict(train1)

	# cv2.imshow('All lines',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
