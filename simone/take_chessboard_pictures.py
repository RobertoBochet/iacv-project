import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

v = cv2.VideoCapture()
v.open(0)
v.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 720p

i=0
while True:
    _, img = v.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    chessboard_found, corners = cv2.findChessboardCorners(gray, (9,6),None)
    if chessboard_found:
        img_orig = img.copy()
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        cv2.drawChessboardCorners(img, (9,6), corners, chessboard_found)
    else:
        img_orig = img
    
    cv2.imshow("img", img)

    key = cv2.waitKey(16)
    if key == 32:                       # press space to take a picture
        cv2.imwrite("./chessboard/"+str(i)+".jpg", img_orig)
        i = i+1
    elif key == 120:                    # x to quit
        quit()
