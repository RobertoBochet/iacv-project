import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from filterpy.kalman import KalmanFilter

""" If the cameras have moved run stereo_calibration.py before this script! """

chessboard_square_size = 25

dictionary = cv2.aruco.custom_dictionary(5, 3)
parameters = cv2.aruco.DetectorParameters_create()
K1 = np.loadtxt('./data/camera1/intrinsics.txt')
dist1 = np.loadtxt('./data/camera1/distortion.txt')
K2 = np.loadtxt('./data/camera2/intrinsics.txt')
dist2 = np.loadtxt('./data/camera2/distortion.txt')
R1 = np.loadtxt('./data/camera1/R.txt')
t1 = np.loadtxt('./data/camera1/t.txt')
R2 = np.loadtxt('./data/camera2/R.txt')
t2 = np.loadtxt('./data/camera2/t.txt')
t1 = t1.reshape(3,1)
t2 = t2.reshape(3,1)

c1 = cv2.VideoCapture()
c2 = cv2.VideoCapture()
c1.open('./video/video1.avi')
c2.open('./video/video2.avi')
# c1.open(0)
# c1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)        # 720p
# c2.open(1)
# c2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)        # 720p
# c2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# c2.set(cv2.CAP_PROP_AUTOFOCUS, 0)
# c2.set(cv2.CAP_PROP_FOCUS, 0)

f = KalmanFilter (dim_x=3, dim_z=3)
# init the filter on the center of the board..
f.x = np.array([[100.],     # x
                [62.5],     # y
                [10.]])      # z
# ..with a large variance
f.P = np.array([[1000.,    0.,    0.],
                [   0., 1000.,    0.],
                [   0.,    0., 1000.]])
# the new tip position will be somewhat close to the old one -> no explicit dynamics
f.F = np.array([[1.,0.,0.],
                [0.,1.,0.],
                [0.,0.,1.]])
f.H = np.array([[1.,0.,0.],
                [0.,1.,0.],
                [0.,0.,1.]])
f.R = 10*np.array([ [1.,0.,0.],
                    [0.,1.,0.],
                    [0.,0.,1.]])
f.Q = np.array([[1.,0.,0.],
                [0.,1.,0.],
                [0.,0.,1.]])
f.test_matrix_dimensions()
f.predict()


""" I had to copy this fucker from github because cv2.triangulatePoints doesn't want to work ffs """
def triangulate_nviews(P, ip):
    """
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices.
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as len of P
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]


def triangulate_points(P1, P2, x1, x2):
    """
    Two-view triangulation of points in
    x1,x2 (nx3 homog. coordinates).
    Similar to openCV triangulatePoints.
    """
    if not len(x2) == len(x1):
        raise ValueError("Number of points don't match.")
    X = [triangulate_nviews([P1, P2], [x[0], x[1]]) for x in zip(x1, x2)]
    return np.array(X)


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
    r,c,_ = img1.shape
    for l in lines:
        l = l[0]
        x0,y0 = map(int, [0, -l[2]/l[1] ])
        x1,y1 = map(int, [c, -(l[2]+l[0]*c)/l[1] ])
        cv2.line(img, (x0,y0), (x1,y1), (0,255,0), 1, cv2.LINE_AA)

    return img

def millis():
    return int(time.time() * 1000.0)


img1, img2 = stereo_read(c1, c2)

chessboard_found1, corners1 = cv2.findChessboardCorners(img1, (9,6),None)
chessboard_found2, corners2 = cv2.findChessboardCorners(img2, (9,6),None)

# object_points = chessboard_points_3d()
# _, _, _, _, _, R, T, E, F = cv2.stereoCalibrate([object_points], [corners1], [corners2], K1, dist1, K2, dist2, None, flags=cv2.CALIB_FIX_INTRINSIC)
# print(np.linalg.norm(T))
cv2.waitKey(0)

epipolar_loss = []
paintboard = np.zeros((125, 200, 1), dtype='uint8') # each pixel is 1 mm in the real world
paintboard_cursor = copy.deepcopy(paintboard)

cv2.namedWindow('paintboard', cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)

center = 0
old_row = -1
old_col = -1
previous_frame_ok = False

while True:
    timestamp = millis()

    img1, img2 = stereo_read(c1, c2)
    img1, img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1, img2 = cv2.equalizeHist(img1), cv2.equalizeHist(img2)
    corners1, ids1, rejected1 = cv2.aruco.detectMarkers(img1, dictionary, parameters=parameters, cameraMatrix=K1, distCoeff=dist1)
    corners2, ids2, rejected2 = cv2.aruco.detectMarkers(img2, dictionary, parameters=parameters, cameraMatrix=K2, distCoeff=dist2)

    if len(corners1) > 0:
        # lines = cv2.computeCorrespondEpilines(corners1[0], 1, F)
        # print(corners1, lines)
        # draw_parametric_lines(img2, lines)

        if len(corners2) > 0:
            previous_frame_ok = True
            points3d = triangulate_points(K1 @ np.hstack((R1, t1)), K2 @ np.hstack((R2, t2)), np.hstack((corners1[0][0], np.ones((4,1)))), np.hstack((corners2[0][0], np.ones((4,1)))))

            # points3d = triangulate_points(np.hstack((K1, np.zeros((3,1)))), K2 @ np.hstack((R, T)), np.hstack((corners1[0][0], np.ones((4,1)))), np.hstack((corners2[0][0], np.ones((4,1)))))
            # points3d = cv2.triangulatePoints(np.hstack((K1, np.zeros((3,1)))), K2 @ np.hstack((R, T)), corners1[0], corners2[0])
            # points3d = cv2.triangulatePoints(K1 @ np.hstack((R1, t1)), K2 @ np.hstack((R2, t2)), corners1[0][0].T, corners2[0][0].T)
            cartesian3d = []
            image_points = []
            for point in points3d:
                # cartesian3d.append(point[0:3]/point[3])
                cartesian = (point[0:3]/point[3]).reshape(3,1)

                image_point = K1 @ ((R1 @ cartesian) + t1)
                image_points.append(image_point[0:2]/image_point[2])
            
            # rvec1, _ = cv2.Rodrigues(R1)
            # image_points, _ = cv2.projectPoints(np.array(cartesian3d), rvec1, t1, K1, dist1)

            for point in image_points:
                cv2.drawMarker(img1, (int(point[0]), int(point[1])), (255,0,0))

            center = 0.25*(points3d[0][0:3]+points3d[1][0:3]+points3d[2][0:3]+points3d[3][0:3])
            center_delta_y = 0.5*(points3d[3][0:3] - points3d[0][0:3]) + 0.5*(points3d[2][0:3] - points3d[1][0:3])
            center_delta_x = 0.5*(points3d[1][0:3] - points3d[0][0:3]) + 0.5*(points3d[3][0:3] - points3d[2][0:3])
            y_dir = center_delta_y / np.linalg.norm(center_delta_y)
            x_dir = center_delta_x / np.linalg.norm(center_delta_x)

            tip = center - 15 * y_dir   # tip is on the top side to experiment a bit
            # print(tip[2])
            # tip = center + 135*y_dir
            tip = tip.reshape(3,1)
            center = center.reshape(3,1)

            """ KF filtering """
            f.update(tip)
            f.predict()
            tip = f.x

            """ project back to image plane to visualize the tracking """
            image_center1 = K1 @ ((R1 @ center) + t1)
            image_center1 = image_center1[0:2]/image_center1[2]
            image_center2 = K2 @ ((R2 @ center) + t2)
            image_center2 = image_center2[0:2]/image_center2[2]
            cv2.drawMarker(img1, (int(image_center1[0]), int(image_center1[1])), (0,0,255))

            image_tip = K1 @ ((R1 @ tip) + t1)
            image_tip = image_tip[0:2]/image_tip[2]
            cv2.drawMarker(img1, (int(image_tip[0]), int(image_tip[1])), (0,0,255))

            row = 125 - int(tip[0])
            col = 200 - int(tip[1])
            print(tip[2])
            if row >= 0 and col >= 0 and row < 125 and col < 200:   # if inside the chessboard area
                if tip[2] <= 5:     # touching
                    if old_row != -1 and old_col != -1:
                        cv2.line(paintboard, (old_col, old_row), (col, row), (255))
                    else:
                        paintboard[row, col] = 255
                    old_row = row
                    old_col = col
                else:               # not touching
                    old_row = -1
                    old_col = -1
                
                paintboard_cursor = copy.deepcopy(paintboard)
                paintboard_cursor[row, col] = 100
            else:
                old_row = -1
                old_col = -1
        else:
            old_row = -1
            old_col = -1
    else:
        old_row = -1
        old_col = -1

    # print(millis() - timestamp)

    cv2.imshow('img2', img2)
    cv2.imshow('img1', img1)
    cv2.imshow('paintboard', paintboard_cursor)
    key = cv2.waitKey(1)
    if key == 32:
        paintboard = np.zeros((125, 200, 1), dtype='uint8')
