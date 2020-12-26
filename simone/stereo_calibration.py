import cv2
import numpy as np
import time

chessboard_square_size = 25 # mm

ar_verts = 3 * chessboard_square_size * np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                                                    [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1],
                                                    [0, 0.5, 1.5], [1, 0.5, 1.5]])
ar_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 8), (5, 8), (6, 9), (7, 9), (8, 9)]

def millis():
    return int(time.time() * 1000.0)

def chessboard_points_3d():
    points_3d = np.zeros((6*9,3), np.float32)
    for row in range(6):
        for col in range(9):
            points_3d[9*row+col] = (row*chessboard_square_size, col*chessboard_square_size, 0)
    
    return points_3d

def extrinsics_calibration(camera_index, video_path):

    K = np.loadtxt('./data/camera'+ str(camera_index) + '/intrinsics.txt')
    dist = np.loadtxt('./data/camera'+ str(camera_index) + '/distortion.txt')

    cap = cv2.VideoCapture()
    cap.open(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)        # 720p

    while True:
        timestamp = millis()
        _, img = cap.read()
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        chessboard_found, corners = cv2.findChessboardCorners(img, (9,6),None)
        if chessboard_found:
            _, rvec, tvec, _ = cv2.solvePnPRansac(chessboard_points_3d(), corners, K, dist)
            print(np.linalg.norm(tvec))     # prints the distance from the first corner to the camera origin, in millimeters
            
            R, _ = cv2.Rodrigues(rvec)
            x_ax = np.array([[2], [0], [0]])
            y_ax = np.array([[0], [2], [0]])
            z_ax = np.array([[0], [0], [2]])

            u = K @ ((R @ x_ax) + tvec)
            v = K @ ((R @ y_ax) + tvec)
            z = K @ ((R @ z_ax) + tvec)

            # u, _ = cv2.projectPoints(x_ax, rvec, tvec, K, dist)
            verts = cv2.projectPoints(ar_verts, rvec, tvec, K, dist)[0].reshape(-1, 2)

            for i, j in ar_edges:
                (x0, y0), (x1, y1) = verts[i], verts[j]
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 3, cv2.LINE_AA)

            u = u[0:2]/u[2]
            v = v[0:2]/v[2]
            z = z[0:2]/z[2]

            cv2.line(img, (int(corners[0,0,0]), int(corners[0,0,1])), (int(u[0]), int(u[1])), (255, 0 ,0), 3, cv2.LINE_AA)
            cv2.line(img, (int(corners[0,0,0]), int(corners[0,0,1])), (int(v[0]), int(v[1])), (0, 255, 0), 3, cv2.LINE_AA)
            cv2.line(img, (int(corners[0,0,0]), int(corners[0,0,1])), (int(z[0]), int(z[1])), (0, 0, 255), 3, cv2.LINE_AA)

            # cv2.drawChessboardCorners(img, (9,6), corners, chessboard_found)
            cv2.imshow('img', img)

        deltaTime = millis() - timestamp

        key = cv2.waitKey(max(1, 33 - deltaTime))  # adaptive loop frequency
        if key == 32:
            break

    np.savetxt('./data/camera'+ str(camera_index) + '/R.txt', R)
    np.savetxt('./data/camera'+ str(camera_index) + '/t.txt', tvec)

    return R, tvec
    

R1, t1 = extrinsics_calibration(1, 0)
R2, t2 = extrinsics_calibration(2, 1)