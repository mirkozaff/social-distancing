# import the necessary packages
import numpy as np
import cv2

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	print(rect)
	pts = pts.astype('float32')

	# return the ordered coordinates
	return pts

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	#rect = order_points(pts)
	rect = np.array(pts, dtype = "float32")
	(tl, tr, br, bl) = rect
	
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)

	'''
	a = np.dot(M, (1700, 212, 1))
	b = np.dot(M, (1637, 212, 1))
	dist = np.sqrt(((a[0]/a[2] - b[0]/b[2]) ** 2) + ((a[1]/a[2] - b[1]/b[2]) ** 2))
	print(dist)

	a = np.dot(M, (1486, 868, 1))
	b = np.dot(M, (1651, 518, 1))
	dist = np.sqrt(((a[0]/a[2] - b[0]/b[2]) ** 2) + ((a[1]/a[2] - b[1]/b[2]) ** 2))
	print(dist)

	a = np.dot(M, (1052, 294, 1))
	b = np.dot(M, (1115, 302, 1))
	dist = np.sqrt(((a[0]/a[2] - b[0]/b[2]) ** 2) + ((a[1]/a[2] - b[1]/b[2]) ** 2))
	print(dist)

	

	a = np.dot(M, (368, 590, 1))
	b = np.dot(M, (546, 465, 1))
	dist = np.sqrt(((a[0]/a[2] - b[0]/b[2]) ** 2) + ((a[1]/a[2] - b[1]/b[2]) ** 2))
	print(dist)

	a = np.dot(M, (1486, 868, 1))
	b = np.dot(M, (1971, 516, 1))
	dist = np.sqrt(((a[0]/a[2] - b[0]/b[2]) ** 2) + ((a[1]/a[2] - b[1]/b[2]) ** 2))
	print(dist)

	point_transformed = cv2.perspectiveTransform(np.array([1491, 581, 1], dtype='float32'), M)

	a = np.dot(M, (1491, 581, 1))

	#c = np.zeros((1, 2), dtype = "float32")
	#a = np.array([368, 590, 1], dtype = "float32")
	#pointsOut = cv2.perspectiveTransform( a,  c, M)
	#print(pointsOut)
	'''
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	#warped = cv2.circle(warped, (int(a[0]/a[2]), int(a[1]/a[2])), radius=20, color=(0, 0, 255), thickness=-1)

	# return the warped image
	return warped, M