import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import cv2

def GMM(img, plot=False):
	# img = cv2.imread("data/cropped.png")
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# _, _, img = cv2.split(img)
	orig = img.copy()
	np.random.seed(1)
	n = 10
	l = 256
	im = np.zeros((l, l))
	points = l*np.random.random((2, n**2))
	im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
	im = img
	im = ndimage.gaussian_filter(im, sigma=l/(4.*n))

	mask = (im > im.mean()).astype(np.float)


	img = mask + 0.3*np.random.randn(*mask.shape)

	hist, bin_edges = np.histogram(img, bins=60)
	bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

	classif = GaussianMixture(n_components=2)
	classif.fit(img.reshape((img.size, 1)))

	threshold = np.mean(classif.means_)
	binary_img = img > threshold
	# binary_img = [[[1 if i else 0 for i in row] for row in col] for col in binary_img]

	# binary_img = binary_img.astype(int)
	if plot:
		plt.figure(figsize=(11,4))

		plt.subplot(131)
		plt.imshow(img)
		plt.axis('off')
		plt.subplot(132)
		plt.plot(bin_centers, hist, lw=2)
		plt.axvline(0.5, color='r', ls='--', lw=2)
		plt.text(0.57, 0.8, 'histogram', fontsize=20, transform = plt.gca().transAxes)
		plt.yticks([])
		plt.subplot(133)
		plt.imshow(binary_img, cmap=plt.cm.gray, interpolation='nearest')
		plt.axis('off')

		plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
		plt.show()
	return img, binary_img
# filename = 'data/cropped_ir.png'

# # filename = 'data/cropped_no_ir.png'


# # take in undistorted image
# # filename = 'data/kinect1/enhanced/enhanced_image1.png'
# # filename = 'data/iphone/IMG_4335.JPG'
# img = cv2.imread(filename)
# GMM(img, plot=True)