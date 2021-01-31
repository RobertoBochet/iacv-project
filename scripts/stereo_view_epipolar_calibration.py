import cv2
import numpy as np
import matplotlib.pyplot as plt

""" 
This script takes the frame in the middle of video1 and computes the epipolar distance 
of the aruco corners in the frames of video2 around its middle frame, and plots such distance
to visualize how it changes as the two frames become more and more out of sync
"""


chessboard_square_size = 25
subvideo_length = 8 # seconds to analyze around video mid frame

dictionary = cv2.aruco.custom_dictionary(5, 3)
parameters = cv2.aruco.DetectorParameters_create()
K1 = np.loadtxt('./data/camera1/intrinsics.txt')
dist1 = np.loadtxt('./data/camera1/distortion.txt')
K2 = np.loadtxt('./data/camera2/intrinsics.txt')
dist2 = np.loadtxt('./data/camera2/distortion.txt')
c1 = cv2.VideoCapture()
c2 = cv2.VideoCapture()
c1.open('./video/video1.avi')
c2.open('./video/video2.avi')

def chessboard_points_3d():
    points_3d = np.zeros((6*9,3), np.float32)
    for row in range(6):
        for col in range(9):
            points_3d[9*row+col] = (row*chessboard_square_size, col*chessboard_square_size, 0)
    
    return points_3d

def stereo_read(c1, c2):
    c1.grab()
    c2.grab()
    _, img1 = c1.retrieve()
    _, img2 = c2.retrieve()
    return img1, img2

def draw_parametric_lines(img,lines):
    r,c,_ = img.shape
    for l in lines:
        l = l[0]
        x0,y0 = map(int, [0, -l[2]/l[1] ])
        x1,y1 = map(int, [c, -(l[2]+l[0]*c)/l[1] ])
        cv2.line(img, (x0,y0), (x1,y1), (0,255,0), 1, cv2.LINE_AA)

    return img

img1, img2 = stereo_read(c1, c2)

chessboard_found1, corners1 = cv2.findChessboardCorners(img1, (9,6),None)
chessboard_found2, corners2 = cv2.findChessboardCorners(img2, (9,6),None)
object_points = chessboard_points_3d()
_, _, _, _, _, R, T, E, F = cv2.stereoCalibrate([object_points], [corners1], [corners2], K1, dist1, K2, dist2, None, flags=cv2.CALIB_FIX_INTRINSIC)
print(np.linalg.norm(T))

epipolar_loss = []
frames2 = []

for i in range(int(20/2 - subvideo_length/2)*30):
    stereo_read(c1, c2)

for i in range(subvideo_length*30):
    i1, i2 = stereo_read(c1, c2)
    frames2.append(i2)
    if i == 2*30:   # the minimum of the epipolar loss should be after 2 seconds of playback (x=60 in the plot)
        img1 = i1   # fix the frame of camera1 at the middle of the video

img1 = cv2.equalizeHist(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
cv2.imshow('img1', img1)
# cv2.waitKey(0)

corners1, ids1, rejected1 = cv2.aruco.detectMarkers(img1, dictionary, parameters=parameters, cameraMatrix=K1, distCoeff=dist1)
if len(corners1) > 0:

    for img2 in frames2:
        img2_aruco = cv2.equalizeHist(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
        corners2, ids2, rejected2 = cv2.aruco.detectMarkers(img2_aruco, dictionary, parameters=parameters, cameraMatrix=K2, distCoeff=dist2)

        lines = cv2.computeCorrespondEpilines(corners1[0], 1, F)
        draw_parametric_lines(img2, lines)

        if len(corners2) > 0:
            loss = 0
            for corner1,corner2 in zip(corners1[0][0], corners2[0][0]):
                print(corner1.T)
                loss = loss + np.absolute(np.hstack((corner2, 1)) @ F @ np.hstack((corner1, 1)).reshape(3,1))**.5
                """ Why the square root? Since the match between two frames happens when loss -> 0, it enhances the peaks:
                    the root stretches values < 1 proportionally to how close to 0 they are
                """

            epipolar_loss.append(loss)

        else:
            epipolar_loss.append(None)

        cv2.imshow('img2', img2)
        cv2.imshow('img1', img1)
        cv2.waitKey(33)
    
    t = np.array(range(len(epipolar_loss)))
    plt.plot(t, np.array(epipolar_loss))
    plt.show()
else:
    print('marker in img1 not found!')
