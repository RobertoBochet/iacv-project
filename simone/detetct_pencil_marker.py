import cv2
import numpy as np

camera_index = 1

dictionary = cv2.aruco.custom_dictionary(5, 3)
parameters = cv2.aruco.DetectorParameters_create()
K = np.loadtxt('./data/camera'+ str(camera_index) + '/intrinsics.txt')
dist = np.loadtxt('./data/camera'+ str(camera_index) + '/distortion.txt')

v = cv2.VideoCapture()
v.open(0)
v.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)        # 720p

while True:
    _, img = v.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters, cameraMatrix=K, distCoeff=dist)
    # print(ids)
    img = cv2.aruco.drawDetectedMarkers(img, corners)
    
    cv2.imshow("pencil", img)
    cv2.waitKey(16)