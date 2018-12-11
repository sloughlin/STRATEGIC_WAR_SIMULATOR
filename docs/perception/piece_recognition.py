import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
import operator
from GaussianMixture import GMM
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from enhance import enhance

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range


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
	a,b = lines.shape
	for i in range(a):
	    rho = lines[i][0]
	    theta = lines[i][1]
	    a = np.cos(theta)
	    b = np.sin(theta)
	    x0, y0 = a*rho, b*rho
	    pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
	    pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )

	    cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)
	return img

def get_lines(img):    
	blur = cv2.GaussianBlur(img,(5,5),0)

	gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

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
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# hist,bins = np.histogram(img.flatten(),256,[0,256])

	# cdf = hist.cumsum()
	# cdf_normalized = cdf * hist.max()/ cdf.max()

	# # plt.plot(cdf_normalized, color = 'b')
	# # plt.hist(img.flatten(),256,[0,256], color = 'r')
	# # plt.xlim([0,256])
	# # plt.legend(('cdf','histogram'), loc = 'upper left')
	# # plt.show()
	# cdf_m = np.ma.masked_equal(cdf,0)
	# cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	# cdf = np.ma.filled(cdf_m,0).astype('uint8')
	# img2 = cdf[img]

	# hist,bins = np.histogram(img2.flatten(),256,[0,256])

	# cdf = hist.cumsum()
	# cdf_normalized = cdf * hist.max()/ cdf.max()

	# plt.plot(cdf_normalized, color = 'b')
	# plt.hist(img2.flatten(),256,[0,256], color = 'r')
	# plt.xlim([0,256])
	# plt.legend(('cdf','histogram'), loc = 'upper left')
	# plt.show()
	# cv2.imshow("Contrast", img2)
	# cv2.waitKey(0)
	# create a CLAHE object (Arguments are optional).
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(img.astype('uint8'))
	# cv2.imshow("Contrast", cl1)
	# cv2.waitKey(0)
	return cl1
# filename = 'data/kinect1/image_raw_screenshot_34.11.2018.png'
# filename = 'data/kinect2/image_color_screenshot_02.12.2018.png'
# filename = 'data/cropped.png'
filename = 'data/cropped_ir.png'

# filename = 'data/cropped_no_ir.png'


# take in undistorted image
# filename = 'data/kinect1/enhanced/enhanced_image1.png'
# filename = 'data/iphone/IMG_4335.JPG'
img = cv2.imread(filename)

# cv2.imshow("Original", img)
# cv2.waitKey(0)

blur = cv2.GaussianBlur(img,(5,5),0)

# cv2.imshow("Original", blur)
# cv2.waitKey(0)

gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

# thresh_test(gray)

ret,thresh = cv2.threshold(gray,100, 255,cv2.THRESH_TOZERO)
# cv2.imshow("thresh", thresh)
# cv2.waitKey(0)

# img = line_detection(img, gray)

# dst = cv2.Canny(gray, 50, 200)
# dst = cv2.Canny(gray, 50, 300)


# cv2.imshow('dst', dst)
# cv2.waitKey(0)

# lines= cv2.HoughLines(dst, 1, np.pi/180.0, 100, np.array([]), 0, 0)
# lines = lines.reshape((lines.shape[0], lines.shape[2]))
lines = get_lines(img)

# out = draw_lines(img, lines)
# cv2.imshow("lines", out)
# cv2.waitKey(0)

#DRAW ONLY OUTERMOST
#Vertical means theta ~ 0
# vert_lines = np.array([i for i in lines if np.isclose(i[1], 0)])
# print(vert_lines)
# vert_lines = vert_lines.reshape((vert_lines.shape[0], vert_lines.shape[2]))
#Horizontal means theta ~ pi/2
# horiz_lines = np.array([i for i in lines if np.isclose(i[1], np.pi / 2)])
# horiz_lines = horiz_lines.reshape((horiz_lines.shape[0], horiz_lines.shape[2]))

# min_vert = min(vert_lines, key=operator.itemgetter(0))
# max_vert = max(vert_lines, key=operator.itemgetter(0))

# min_horiz = min(horiz_lines, key=operator.itemgetter(0))
# max_horiz = max(horiz_lines, key=operator.itemgetter(0))

# bounding_lines = np.array(sorted([min_horiz, min_vert, max_horiz, max_vert], key=operator.itemgetter(0)))



#Remove bounding lines:
#minimum distances between rho, theta for lines to be distinct
rho_min = 10	
theta_min = 0.5

internal_lines = lines.copy()
# a, b = lines.shape
# for i in range(a):
# 	c, d = bounding_lines.shape
# 	for j in range(c):
# 		if  abs(lines[i][0]- bounding_lines[j][0]) <= rho_min and abs(lines[i][1] - bounding_lines[j][1]) <= theta_min:
# 			# internal_lines.append(lines[i])
# 			# print("spdiuafb")
# 			# internal_lines[i] = [0, 0]
# 			print("")

# internal_lines = np.array([x for x in internal_lines if x[0] != 0 and x[1] != 0])

# internal_lines = np.array(internal_lines)
internal_lines = np.array(sorted(internal_lines, key=operator.itemgetter(0)))
# print(internal_lines[2], internal_lines[3], abs(internal_lines[2][0] - internal_lines[3][0]) <= rho_min, abs(internal_lines[2][1] - internal_lines[3][1]) <= theta_min)

#remove duplicates:
a, b = internal_lines.shape
for i in range(a):
	for j in range(a):
		if i != j and (abs(internal_lines[i][0]- internal_lines[j][0]) <= rho_min and abs(internal_lines[i][1]- internal_lines[j][1]) <= theta_min):
			#delete the one with worse error
			i_error = min(internal_lines[i][1] - 0, internal_lines[i][1] - np.pi / 2)
			j_error = min(internal_lines[j][1] - 0, internal_lines[j][1] - np.pi / 2)

			if i_error > j_error:
				# np.delete(internal_lines, i)
				internal_lines[i] = [-1, -1]
			else:
				# np.delete(internal_lines, j)
				internal_lines[j] = [-1, -1]
		# elif i != j:
		# 	# print(abs(internal_lines[i][0]- internal_lines[j][0]) , abs(internal_lines[i][1]- internal_lines[j][1]))
		# 	print(internal_lines[i][1])

# print(internal_lines)
internal_lines = np.array([x for x in internal_lines if x[0] != -1 and x[1] != -1])
# internal_lines = internal_lines.reshape((internal_lines.shape[0], internal_lines.shape[2]))


out = draw_lines(img, internal_lines)
cv2.imshow("Out", out)
cv2.waitKey(0)

#lines ordered 1st horiz, 1st vert, 2nd horiz, 2nd vert, ...
#Split into horizontal, vertical lines:

# vertical = [i for i in internal_lines if abs(i[1]- 0) < 0.5]
vertical = [i for i in internal_lines if np.isclose(i[1], 0)]

horizontal = [i for i in internal_lines if abs(i[1] - np.pi/2) < 0.5]

#Add min_vert, max_vert, min_horiz, max_horiz (bounding lines):

vertical = np.array(vertical)
horizontal = np.array(horizontal)

chessboard_imgs = []
chessboard = np.zeros((8, 8))


#Equalize image histogram
img = equalize_hist(img)

# GaussianMixture.GaussianMixtureModel(img, plot=True)
pca = PCA()
scaler = StandardScaler()
gmm = GaussianMixture(n_components=3)

#Black piece detector:
params_black = cv2.SimpleBlobDetector_Params()
params_black.minThreshold = 50
params_black.maxThreshold = 70
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector_black = cv2.SimpleBlobDetector(params_black)
else : 
    detector_black = cv2.SimpleBlobDetector_create(params_black)

#White Piece Detector



# ADD FOR EDGES

a, _ = horizontal.shape
b, _ = vertical.shape
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

		#Check if img already grayscale:
		if len(img.shape) > 2:
			gray = cv2.cvtColor(cropped_square, cv2.COLOR_BGR2GRAY)
		else:
			gray = cropped_square
		blur = cv2.GaussianBlur(gray,(5,5),0)
		# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # Detect blobs.
		keypoints = detector_black.detect(gray)
		if len(keypoints) != 0: 
			chessboard[i][j] = 2
		# chessboard_imgs.append(gray)
		# # Draw detected blobs as red circles.
		# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
		# im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		# cv2.imshow("Keypoints", im_with_keypoints)
		# cv2.waitKey(0)

		# print(gray.shape)
		# GMM(gray, plot=True)
		gray = gray.astype('float64')
		# #Reduce image dimensionality with PCA:
		gray = scaler.fit_transform(gray)
		gray = pca.fit_transform(gray)
		gray = gray.flatten()
		# print(gray.shape)
		chessboard_imgs.append(gray)
		# cv2.imshow("blur", blur)
		# cv2.imshow("gray", gray)
		# cv2.waitKey(0)

		# chessboard_imgs.append(gray)
		# print(gray.shape)
		# ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		# th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
  #           cv2.THRESH_BINARY,11,2)
		# cv2.imshow('Square',cropped_square)
		# cv2.waitKey(0)

		#get rid of edge bits:


		# im, bin = GaussianMixture.GaussianBlurianMixtureModel(blur)
		# cv2.imshow("Gauss", im)
		# cv2.waitKey(0)
# chessboard_imgs = np.array(chessboard_imgs)
# print(gmm.fit_predict(chessboard_imgs))
# print(labels)
chessboard = chessboard.flatten()
chessboard_imgs = np.array(chessboard_imgs)
print(chessboard_imgs.shape)
print(gmm.fit(chessboard_imgs))






# cv2.imshow('All lines',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()