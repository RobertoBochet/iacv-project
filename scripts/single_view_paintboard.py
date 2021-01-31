import cv2
import numpy as np

marker_square_size = 22 # mm

dictionary = cv2.aruco.custom_dictionary(5, 3)
parameters = cv2.aruco.DetectorParameters_create()
K = np.loadtxt('./data/camera1/intrinsics.txt')
dist = np.loadtxt('./data/camera1/distortion.txt')
R = np.loadtxt('./data/camera1/R.txt')
tvec = np.loadtxt('./data/camera1/t.txt')

v = cv2.VideoCapture()
v.open(0)
v.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)        # 720p


while True:
    _, img = v.read()
    corners, ids, rejected = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters, cameraMatrix=K, distCoeff=dist)

    if len(corners) > 0:
        img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
        marker_rvecs, marker_tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_square_size, K, dist)

        tip_position_marker = np.array([[0], [-(150-15)], [0]], float)  # position of tip wrt top-left corner of marker, in marker ref. frame
        marker_R, _ = cv2.Rodrigues(marker_rvecs[0])    # converts rvec in a rotation matrix
        tip_position_camera = marker_R @ tip_position_marker + marker_tvecs[0].reshape(3,1)     # reshape needed to make it a column vector instad of row
        tip_position_world = R.T @ (tip_position_camera - tvec.reshape(3,1))

        # print(tip_position_camera)
        tip_image = K @ tip_position_camera
        tip_image = tip_image[0:2]/tip_image[2]     # homogeneous -> cartesian

        cv2.drawMarker(img, (int(tip_image[0]), int(tip_image[1])), (0,255,0))

        print(tip_position_world)
        
        cv2.imshow("pencil", img)
        cv2.waitKey(1)