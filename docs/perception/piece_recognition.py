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
from sklearn.cluster import KMeans

# from enhance import enhance

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
	cv2.imshow('dst', dst)
	cv2.waitKey(0)
	lines= cv2.HoughLines(dst, 1, np.pi/180.0, 100, np.array([]), 0, 0)
	if lines is not None:
		lines = lines.reshape((lines.shape[0], lines.shape[2]))
	return lines
def equalize_hist(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# create a CLAHE object (Arguments are optional).
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(img.astype('uint8'))
	# cv2.imshow("Contrast", cl1)
	# cv2.waitKey(0)
	# cl1 = cv2.equalizeHist(img)
	return cl1
# filename = 'data/kinect1/image_raw_screenshot_34.11.2018.png'
# filename = 'data/kinect2/image_color_screenshot_02.12.2018.png'
# filename = 'data/cropped.png'
filename = 'data/cropped_ir.png'
# filename = 'data/cropped_no_ir.png'


img = cv2.imread(filename)

# cv2.imshow("Original", img)
# cv2.waitKey(0)

blur = cv2.GaussianBlur(img,(5,5),0)

gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

# thresh_test(gray)

ret,thresh = cv2.threshold(gray,100, 255,cv2.THRESH_TOZERO)
# cv2.imshow("thresh", thresh)
# cv2.waitKey(0)


lines = get_lines(img)
out = draw_lines(img, lines)
cv2.imshow("Out", out)
cv2.waitKey(0)


#Remove bounding lines:
#minimum distances between rho, theta for lines to be distinct
rho_min = 10	
theta_min = 0.5

internal_lines = lines.copy()
minvert = (0,0)
minhoriz = (0, np.pi/2)

maxvert = ( img.shape[1], 0)
maxhoriz = (img.shape[1], np.pi/2)
np.append(internal_lines, minvert)
np.append(internal_lines, minhoriz)
np.append(internal_lines, maxvert)
np.append(internal_lines, maxhoriz)

# internal_lines = np.array(internal_lines)
internal_lines = np.array(sorted(internal_lines, key=operator.itemgetter(0)))

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


#lines ordered 1st horiz, 1st vert, 2nd horiz, 2nd vert, ...
#Split into horizontal, vertical lines:

# vertical = [i for i in internal_lines if abs(i[1]- 0) < 0.5]
vertical = [i for i in internal_lines if np.isclose(i[1], 0)]

horizontal = [i for i in internal_lines if abs(i[1] - np.pi/2) < 0.5]

#Add min_vert, max_vert, min_horiz, max_horiz (bounding lines):


# out = draw_lines(img, internal_lines)
# cv2.imshow("Out", out)
# cv2.waitKey(0)

#Add min_vert, max_vert, min_horiz, 
vertical = np.array(sorted(vertical, key=operator.itemgetter(0)))
horizontal = np.array(sorted(horizontal, key=operator.itemgetter(0)))


chessboard_imgs = []
chessboard_vecs = []
chessboard = np.zeros((8, 8))


#Equalize image histogram
img = equalize_hist(img)

# GaussianMixture.GaussianMixtureModel(img, plot=True)
pca = PCA(n_components=30)
scaler = StandardScaler()
gmm = GaussianMixture(n_components=2)

#Black piece detector:
params_black = cv2.SimpleBlobDetector_Params()
params_black.minThreshold = 10
# params_black.maxThreshold = 150

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
		# keypoints = detector_black.detect(gray)
		# if len(keypoints) != 0: 
		# 	chessboard[i][j] = 2
		# else:
		# 	inv = abs(255 - gray)
		# 	keypoints = detector_black.detect(inv)
		# 	if len(keypoints) != 0: 
		# 		chessboard[i][j] = 1
		
		# # Draw detected blobs as red circles.
		# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
		# im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		# cv2.imshow("Keypoints", im_with_keypoints)
		# cv2.waitKey(0)

		# sift = cv2.xfeatures2d.SIFT_create()
		# kp = sift.detect(gray,None)
		# img=cv2.drawKeypoints(cropped_square,kp)

		# cv2.imshow("SIFT", cropped_square)
		# cv2.waitKey(0)


		# print(gray.shape)
		# GMM(gray, plot=True)
		gray = gray[:34, :32]
		chessboard_imgs.append(gray)


		#Generate set of empty squares:
		name = "data/empty_board/empty{}{}.png".format(i, j)
		cv2.imwrite(name, gray)

		# gray = gray.astype('float64')
		# # #Reduce image dimensionality with PCA:
		# gray = scaler.fit_transform(gray)
		# gray = pca.fit_transform(gray)
		gray = gray.flatten()
		# print(gray.shape)
		chessboard_vecs.append(gray)


# chessboard_imgs = np.array(chessboard_imgs)
# print(gmm.fit_predict(chessboard_imgs))
# print(labels)
chessboard = chessboard.flatten()
chessboard_vecs = np.array(chessboard_vecs)

print(gmm.fit_predict(chessboard_vecs))
predictions = gmm.fit_predict(chessboard_vecs)

squares0 = []
squares1 = []

for i in range(len(predictions)):
	if predictions[i] == 0:
		squares0.append(chessboard_imgs[i])
	else:
		squares1.append(chessboard_imgs[i])


means0 = np.mean(squares0, axis=0)
stddev0 = np.std(squares0, axis=0)
means1 = np.mean(squares1, axis=0)
stddev1 = np.std(squares1, axis=0)
normalized0 = (squares0 - means0 ) / np.square(stddev0) 
normalized1 = (squares1 - means1 ) / np.square(stddev1)
normalized0 = (normalized0 + 1) *1 / 2
normalized1 = (normalized1 + 1) * 1 / 2

kmeans = KMeans(n_clusters=2)

# for i in range(len(normalized0)):
# 	cv2.imshow("Fuck 0 {}".format(i), normalized0[i])

# 	cv2.waitKey(0)
# 	cv2.imshow("Fuck 0 actual", squares0[i])
	# cv2.imshow("Fuck 1 {}".format(i), normalized1[i])
	# cv2.waitKey(0)
train0 = [i.flatten() for i in normalized0]
train1 = [i.flatten() for i in normalized1]
kmeans.fit_predict(train0)
kmeans.fit_predict(train1)

# cv2.imshow('All lines',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
