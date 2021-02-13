#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import cv2 as cv

from paintar.camera import Camera
from paintar.utilities import draw_axis

PATH_IMAGES = Path("./data/geometric_calibration")
PATH_PARAMETERS = Path("./data/parameters")

PATH_IMAGE_1 = PATH_IMAGES / "camera1.jpg"
PATH_IMAGE_2 = PATH_IMAGES / "camera2.jpg"

PATH_PARAMETERS_1 = PATH_PARAMETERS / "camera1"
PATH_PARAMETERS_2 = PATH_PARAMETERS / "camera2"

K1 = np.loadtxt(PATH_PARAMETERS_1 / "intrinsics.txt")
K2 = np.loadtxt(PATH_PARAMETERS_2 / "intrinsics.txt")
DIST1 = np.loadtxt(PATH_PARAMETERS_1 / "distortion.txt")
DIST2 = np.loadtxt(PATH_PARAMETERS_2 / "distortion.txt")
R1 = np.loadtxt(PATH_PARAMETERS_1 / "R.txt")
R2 = np.loadtxt(PATH_PARAMETERS_2 / "R.txt")
T1 = np.loadtxt(PATH_PARAMETERS_1 / "t.txt")
T2 = np.loadtxt(PATH_PARAMETERS_2 / "t.txt")

ARUCO_PEN_TIP_OFFSET = np.array([0, -0.139, 0, 1])
ARUCO_PEN_ID = 0
ARUCO_PEN_SIZE = 0.022
ARUCO_DICT = cv.aruco.custom_dictionary(1, 3)

T_CANVAS = np.block([
    [
        np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
        ]).T,
        np.array([[.15, .25, 0.002]]).T
    ], [0, 0, 0, 1]
])

if __name__ == "__main__":
    cam1 = Camera(k=K1, dist=DIST1, r=R1, t=T1)

    img = cv.imread(PATH_IMAGE_1.as_posix())

    draw_axis(img, cam1.m)
    draw_axis(img, cam1.m @ T_CANVAS, draw_z=False)

    cv.imshow("reference_frame", img)
    cv.waitKey(0)
