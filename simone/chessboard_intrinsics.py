import cv2 as cv2
import numpy as np
import glob

camera_index = 2

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objectPoints = []
imagePoints = []
points_3d = np.zeros((6*9,3), np.float32)
for row in range(6):
    for col in range(9):
        points_3d[9*row+col] = (row,col,0)       # so che è C ma ci mettevo di meno a scriverlo che a leggere i docs di numpy

images = glob.glob('./chessboard/camera' + str(camera_index) + '/*.jpg')

for file in images:

    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    retval, corners = cv2.findChessboardCorners(gray, (9,6),None)      # corners returned row by row, left to right in every row
    if not retval:
        print('[*] Chessboard not detected in picture ' + str(file))
        continue

    cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    cv2.drawChessboardCorners(img, (9,6), corners, retval)
    cv2.imshow('Chessboard', img)
    cv2.waitKey(1000)

    objectPoints.append(points_3d)
    imagePoints.append(corners)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, gray.shape[::-1],None,None)
np.savetxt('./data/camera' + str(camera_index) + '/intrinsics.txt', K)
np.savetxt('./data/camera' + str(camera_index) + '/distortion.txt', dist)

print(K)