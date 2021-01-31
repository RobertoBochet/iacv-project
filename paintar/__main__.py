#!/usr/bin/env python3
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from . import utilities as ut
from .camera import Camera, StereoCamera
from .tracker import Tracker

PATH_VIDEO = Path("./video")
PATH_VIDEO_1 = PATH_VIDEO / "video1.avi"
PATH_VIDEO_2 = PATH_VIDEO / "video2.avi"
PATH_VIDEO_AR = PATH_VIDEO / "ar.mp4"

SKIP_FRAMES = 30 * 10 * 0
TEST_FRAMES = 1500

PATH_PARAMETERS = Path("./data")
PATH_PARAMETERS_1 = PATH_PARAMETERS / "camera1"
PATH_PARAMETERS_2 = PATH_PARAMETERS / "camera2"

K1 = np.loadtxt(PATH_PARAMETERS_1 / "intrinsics.txt")
K2 = np.loadtxt(PATH_PARAMETERS_2 / "intrinsics.txt")
DIST1 = np.loadtxt(PATH_PARAMETERS_1 / "distortion.txt")
DIST2 = np.loadtxt(PATH_PARAMETERS_2 / "distortion.txt")
R1 = np.loadtxt(PATH_PARAMETERS_1 / "R.txt")
R2 = np.loadtxt(PATH_PARAMETERS_2 / "R.txt")
T1 = np.loadtxt(PATH_PARAMETERS_1 / "t.txt") / 1000
T2 = np.loadtxt(PATH_PARAMETERS_2 / "t.txt") / 1000

CHESSBOARD_SIZE = (9, 6)
CHESSBOARD_SQUARE_SIZE = 0.0245

ARUCO_PEN_TIP_OFFSET = np.array([0, -0.15, 0, 1])
ARUCO_PEN_ID = 0
ARUCO_PEN_SIZE = 0.022
ARUCO_DICT = cv.aruco.custom_dictionary(1, 3)

if __name__ == "__main__":
    chessboard = ut.Chessboard(CHESSBOARD_SIZE, CHESSBOARD_SQUARE_SIZE)

    cam1 = Camera(PATH_VIDEO_1.as_posix(), K1, DIST1, R1, T1)
    cam2 = Camera(PATH_VIDEO_2.as_posix(), K2, DIST2, R2, T2)
    # cam1 = Camera(PATH_VIDEO_AR.as_posix(), K1, DIST1)

    stereo_cam = StereoCamera(cam1, cam2)

    # while True:
    #     if stereo_cam.calibrate_geometry(chessboard):
    #         break

    tr = Tracker(stereo_cam,
                 aruco_dict=ARUCO_DICT,
                 aruco_pen_size=ARUCO_PEN_SIZE,
                 aruco_pen_tip_offset=ARUCO_PEN_TIP_OFFSET)

    for _ in range(SKIP_FRAMES):
        stereo_cam.grab()

    stereo_cam.grab()

    while True:
        t_i = time.time()

        if not stereo_cam.grab():
            break

        if not tr.loop(False):
            break
