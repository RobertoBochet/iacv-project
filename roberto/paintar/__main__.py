#!/usr/bin/env python3
import time
from pathlib import Path

import cv2.cv2 as cv
import numpy as np

from . import Tracker
from .camera import Camera, StereoCamera
from .utilities import Chessboard, proj2cart

PATH_VIDEO = Path("../simone/video")
PATH_VIDEO_1 = PATH_VIDEO / "video1.avi"
PATH_VIDEO_2 = PATH_VIDEO / "video2.avi"
PATH_VIDEO_AR = PATH_VIDEO / "ar.mp4"

SKIP_FRAMES = 30 * 6
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

ARUCO_PEN_ID = 0
ARUCO_PEN_SIZE = 0.02
ARUCO_DICT = cv.aruco.custom_dictionary(1, 3)
ARUCO_PARAM = cv.aruco.DetectorParameters_create()

if __name__ == "__main__":
    chessboard = Chessboard(CHESSBOARD_SIZE, CHESSBOARD_SQUARE_SIZE)

    cam1 = Camera(PATH_VIDEO_1.as_posix(), K1, DIST1)
    cam2 = Camera(PATH_VIDEO_2.as_posix(), K2, DIST2)
    # cam1 = Camera(PATH_VIDEO_AR.as_posix(), K1, DIST1)

    stereo_cam = StereoCamera(cam1, cam2)

    while True:
        if stereo_cam.calibrate_geometry(chessboard):
            break

    tr = Tracker(stereo_cam, aruco_dict=ARUCO_DICT, aruco_param=ARUCO_PARAM)

    for _ in range(SKIP_FRAMES):
        stereo_cam.grab()

    while True:
        t_i = time.time()

        stereo_cam.grab()

        img1, img2 = stereo_cam.retrieve()

        # c1, c2 = tr.triangulate_marker(ARUCO_PEN_ID, False)

        # cv.aruco.drawDetectedMarkers(img1, c1, None)
        # cv.aruco.drawDetectedMarkers(img2, c2, None)

        # cv.imshow("main", img1)

        db1 = np.zeros(1, dtype=np.uint8)

        a1 = stereo_cam.cam1.find_aruco_pose(ARUCO_PEN_ID, ARUCO_PEN_SIZE, grab=False,
                                             aruco_dict=ARUCO_DICT, aruco_param=ARUCO_PARAM, debug_buffer=db1)
        a2 = stereo_cam.cam2.find_aruco_pose(ARUCO_PEN_ID, ARUCO_PEN_SIZE, grab=False,
                                             aruco_dict=ARUCO_DICT, aruco_param=ARUCO_PARAM)

        if a1 is not None and a2 is not None:
            img1 = db1

            pb1 = np.linalg.inv(stereo_cam.cam1.a) @ a1 @ np.array([0, 0, 0, 1])

            p1 = stereo_cam.cam1.m @ pb1
            #
            # print(pa1)
            #
            # ma1 = stereo_cam.cam1.k @ a1[:-1]
            #
            # p1 = ma1 @ np.array([-0.02, 0, 0, 1])
            # p2 = ma1 @ np.array([0, -0.02, 0, 1])
            # p3 = ma1 @ np.array([0, 0, -0.02, 1])

            cv.drawMarker(img1, tuple(proj2cart(p1).astype(np.uint8)),
                          (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10)
            # cv.drawMarker(img1, tuple(proj2cart(p2).astype(np.uint8)),
            #               (255, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10)
            # cv.drawMarker(img1, tuple(proj2cart(p3).astype(np.uint8)),
            #               (0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=10)

            # img1 = cv.aruco.drawDetectedMarkers(img1, [np.array([ar1], dtype=np.float32)])
            # img2 = cv.aruco.drawDetectedMarkers(img2, [np.array([ar2], dtype=np.float32)])
            # a, _ = cv.Rodrigues(a1[:3, :3])

            # img1 = cv.aruco.drawAxis(img1, stereo_cam.cam1.k, stereo_cam.cam1.dist,
            #                         a.reshape((1, 1, 3)), a1[3, :3].reshape((1, 1, 3)), 0.05)

            # A = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], dtype=np.float) + np.array(
            #     [[0, 0, 1] for _ in range(4)], dtype=np.float)
            #
            # h1, _ = cv.findHomography(A, ar1)
            # h2, _ = cv.findHomography(A, ar2)
            #
            # p1 = (proj2cart(h1.dot(np.array([0, -14, 1])))).astype(np.int)
            # p2 = (proj2cart(h2.dot(np.array([0, -14, 1])))).astype(np.int)
            #
            # cv.circle(img1, tuple(p1), 40, (255, 0, 0), -1)
            # cv.circle(img2, tuple(p2), 40, (255, 0, 0), -1)
            #
            # cv.drawMarker(img1, tuple(p1), (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10)
            # cv.drawMarker(img2, tuple(p2), (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10)

            # ar = stereo_cam.triangulate_aruco(ARUCO_PEN_ID, grab=False, aruco_dict=ARUCO_DICT, aruco_param=ARUCO_PARAM)

            # print(ar.shape)

            # print("processing time: ", time.time() - t_i, "s")

            # print(stereo_cam.cam1._r.shape)
            # print(stereo_cam.cam1._r)
            # print(stereo_cam.cam1._t)
        cv.imshow("main", img1)
            # cv.imshow("img2", img2)
        #cv.imshow("main", np.concatenate((img1, img2), axis=1))
        cv.waitKey(1)

        #if a1 is not None and a2 is not None:
        #    cv.waitKey(100)

        if cv.getWindowProperty('main', cv.WND_PROP_VISIBLE) < 1:
            break
        # if the window is closed, breaks the loop
        # if cv.getWindowProperty('img1', cv.WND_PROP_VISIBLE) < 1 or \
        #         cv.getWindowProperty('img2', cv.WND_PROP_VISIBLE) < 1:
        #     break

        # stereo_cam.grab()
    #
    # img1, img2 = stereo_cam.retrieve()
    #
    # a=stereo_cam.cam1.find_aruco(ARUCO_PEN_ID, grab=False,
    #                                            aruco_dict=ARUCO_DICT, aruco_param=ARUCO_PARAM)
    #
    # print(a)
