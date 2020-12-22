import cv2
import numpy as np

dictionary = cv2.aruco.custom_dictionary(5, 3)
parameters = cv2.aruco.DetectorParameters_create()
K = np.fromfile("./data/intrinsics.txt")
dist = np.fromfile("./data/distortion.txt")

v = cv2.VideoCapture()
v.open(0)
v.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)        # 720p

border = int((1280 - 720*4/3.0) / 2)         # 16:9 -> 4:3

while True:
    _, img_orig = v.read()
    img = img_orig[:, border:1280-border]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters, cameraMatrix=K, distCoeff=dist)
    print(ids)
    img = cv2.aruco.drawDetectedMarkers(img, corners, ids)

    cv2.imshow("pencil", img)
    cv2.waitKey(1)