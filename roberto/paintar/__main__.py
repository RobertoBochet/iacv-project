#!/usr/bin/env python3
import time
from pathlib import Path

import cv2.cv2 as cv
import numpy as np

from .tracker.estimators import Position3DEstimator
from .camera import Camera, StereoCamera
from . import utilities as ut

PATH_VIDEO = Path("../simone/video")
PATH_VIDEO_1 = PATH_VIDEO / "video1.avi"
PATH_VIDEO_2 = PATH_VIDEO / "video2.avi"
PATH_VIDEO_AR = PATH_VIDEO / "ar.mp4"

SKIP_FRAMES = 30 * 10
TEST_FRAMES = 1500

PATH_PARAMETERS = Path("../simone/data")
PATH_PARAMETERS_1 = PATH_PARAMETERS / "camera1"
PATH_PARAMETERS_2 = PATH_PARAMETERS / "camera1"

K1 = np.loadtxt(PATH_PARAMETERS_1 / "intrinsics.txt")
K2 = np.loadtxt(PATH_PARAMETERS_2 / "intrinsics.txt")
DIST1 = np.loadtxt(PATH_PARAMETERS_1 / "distortion.txt")
DIST2 = np.loadtxt(PATH_PARAMETERS_2 / "distortion.txt")

CHESSBOARD_SIZE = (9, 6)
CHESSBOARD_SQUARE_SIZE = 0.0245

ARUCO_PEN_TIP_OFFSET = np.array([0, -0.15, 0, 1])
ARUCO_PEN_ID = 0
ARUCO_PEN_SIZE = 0.022
ARUCO_DICT = cv.aruco.custom_dictionary(1, 3)
ARUCO_PARAM = cv.aruco.DetectorParameters_create()

if __name__ == "__main__":
    chessboard = ut.Chessboard(CHESSBOARD_SIZE, CHESSBOARD_SQUARE_SIZE)

    cam1 = Camera(PATH_VIDEO_1.as_posix(), K1, DIST1)
    cam2 = Camera(PATH_VIDEO_2.as_posix(), K2, DIST2)
    # cam1 = Camera(PATH_VIDEO_AR.as_posix(), K1, DIST1)

    stereo_cam = StereoCamera(cam1, cam2)

    while True:
        if stereo_cam.calibrate_geometry(chessboard):
            break

    # tr = Tracker(stereo_cam, aruco_dict=ARUCO_DICT, aruco_param=ARUCO_PARAM)

    for _ in range(SKIP_FRAMES):
        stereo_cam.grab()

    KALMAN_INITIAL_X = np.array([[500.], [500.]])
    KALMAN_INITIAL_P = np.array([[1., 0.], [0., 1.]]) * 1000.

    f1 = Position3DEstimator()
    f2 = Position3DEstimator()

    aruco_counter = 0

    while True:
        t_i = time.time()

        stereo_cam.grab()

        img1, img2 = stereo_cam.retrieve()

        # c1, c2 = tr.triangulate_marker(ARUCO_PEN_ID, False)

        # cv.aruco.drawDetectedMarkers(img1, c1, None)
        # cv.aruco.drawDetectedMarkers(img2, c2, None)

        # cv.imshow("main", img1)

        aruco_counter += 1

        db1 = np.zeros(1, dtype=np.uint8)
        db2 = np.zeros(1, dtype=np.uint8)

        a1 = stereo_cam.cam1.find_aruco_pose(ARUCO_PEN_ID, ARUCO_PEN_SIZE, grab=False,
                                             aruco_dict=ARUCO_DICT, aruco_param=ARUCO_PARAM, debug_buffer=db1)
        a2 = stereo_cam.cam2.find_aruco_pose(ARUCO_PEN_ID, ARUCO_PEN_SIZE, grab=False,
                                             aruco_dict=ARUCO_DICT, aruco_param=ARUCO_PARAM, debug_buffer=db2)

        if aruco_counter > 10:
            f1.reset()
            f2.reset()
            print("kalman reset")

        if a1 is not None and a2 is not None:
            aruco_counter = 0
            img1 = db1
            img2 = db2

            p1 = ut.proj_normalization(a1 @ ARUCO_PEN_TIP_OFFSET)
            p2 = ut.proj_normalization(a2 @ ARUCO_PEN_TIP_OFFSET)

            f1.update(ut.proj2cart(p1))
            f2.update(ut.proj2cart(p2))

            p1 = ut.proj2cart(stereo_cam.cam1.m_c @ p1)
            p2 = ut.proj2cart(stereo_cam.cam2.m_c @ p2)

            cv.drawMarker(img1, tuple(p1.astype(np.uint)),
                          (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10)
            cv.drawMarker(img2, tuple(p2.astype(np.uint)),
                          (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10)

            p1 = ut.proj2cart(stereo_cam.cam1.m_c @ ut.cart2proj(f1.x))
            p2 = ut.proj2cart(stereo_cam.cam2.m_c @ ut.cart2proj(f2.x))

            cv.drawMarker(img1, tuple(p1.reshape(2).astype(np.uint)),
                          (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=20)
            cv.drawMarker(img2, tuple(p2.reshape(2).astype(np.uint)),
                          (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=20)

        # cv.imshow("main", img1)
        # cv.imshow("img2", img2)
        cv.imshow("main", np.concatenate((img1, img2), axis=1))
        cv.waitKey(1)

        if cv.getWindowProperty('main', cv.WND_PROP_VISIBLE) < 1:
            break

