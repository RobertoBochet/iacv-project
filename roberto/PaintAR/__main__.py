#!/usr/bin/env python3

from pathlib import Path

import cv2.cv2 as cv
import numpy as np

from .camera import Camera, StereoCamera
from . import Tracker
from .utilities import Chessboard

PATH_VIDEO = Path("../simone/video")
PATH_VIDEO_1 = PATH_VIDEO / "video1.avi"
PATH_VIDEO_2 = PATH_VIDEO / "video2.avi"

SKIP_FRAMES = 600
TEST_FRAMES = 1500

PATH_PARAMETERS = Path("../simone/data")
PATH_PARAMETERS_1 = PATH_PARAMETERS / "camera1"
PATH_PARAMETERS_2 = PATH_PARAMETERS / "camera1"

K1 = np.loadtxt(PATH_PARAMETERS_1 / "intrinsics.txt")
K2 = np.loadtxt(PATH_PARAMETERS_2 / "intrinsics.txt")
DIST1 = np.loadtxt(PATH_PARAMETERS_1 / "distortion.txt")
DIST2 = np.loadtxt(PATH_PARAMETERS_2 / "distortion.txt")

CHESSBOARD_SIZE = (9, 6)
CHESSBOARD_SQUARE_SIZE = 25

ARUCO_DICT = cv.aruco.custom_dictionary(5, 3)
ARUCO_PARAM = cv.aruco.DetectorParameters_create()

if __name__ == "__main__":
    chessboard = Chessboard(CHESSBOARD_SIZE, CHESSBOARD_SQUARE_SIZE)

    cam1 = Camera(PATH_VIDEO_1.as_posix(), K1, DIST1)
    cam2 = Camera(PATH_VIDEO_2.as_posix(), K2, DIST2)

    stereo_cam = StereoCamera(cam1, cam2)

    stereo_cam.calibrate(chessboard)

    tr = Tracker(stereo_cam, aruco_dict=ARUCO_DICT, aruco_param=ARUCO_PARAM)

    print(ARUCO_DICT)
    print(type(ARUCO_DICT))
    print(K1)
