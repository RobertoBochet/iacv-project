#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import cv2 as cv

from paintar.camera import Camera
from paintar.utilities import Chessboard

PATH_DATA = Path("./data") / "demo2"

PATH_PARAMETERS = PATH_DATA / "parameters"
PATH_IMAGES = PATH_DATA / "geometric_calibration"

PATH_PARAMETERS_CAMERA_1 = PATH_PARAMETERS / "camera1"
PATH_PARAMETERS_CAMERA_2 = PATH_PARAMETERS / "camera2"

PATH_INTRINSICS_CAMERA_1 = PATH_PARAMETERS_CAMERA_1 / "intrinsics.txt"
PATH_INTRINSICS_CAMERA_2 = PATH_PARAMETERS_CAMERA_2 / "intrinsics.txt"

PATH_DISTORTION_CAMERA_1 = PATH_PARAMETERS_CAMERA_1 / "distortion.txt"
PATH_DISTORTION_CAMERA_2 = PATH_PARAMETERS_CAMERA_2 / "distortion.txt"

PATH_ROTATION_CAMERA_1 = PATH_PARAMETERS_CAMERA_1 / "R.txt"
PATH_ROTATION_CAMERA_2 = PATH_PARAMETERS_CAMERA_2 / "R.txt"

PATH_POSITION_CAMERA_1 = PATH_PARAMETERS_CAMERA_1 / "t.txt"
PATH_POSITION_CAMERA_2 = PATH_PARAMETERS_CAMERA_2 / "t.txt"

PATH_IMAGE_CAMERA_1 = PATH_IMAGES / "camera1.jpg"
PATH_IMAGE_CAMERA_2 = PATH_IMAGES / "camera2.jpg"

K1 = np.loadtxt(PATH_INTRINSICS_CAMERA_1)
K2 = np.loadtxt(PATH_INTRINSICS_CAMERA_2)

DIST1 = np.loadtxt(PATH_DISTORTION_CAMERA_1)
DIST2 = np.loadtxt(PATH_DISTORTION_CAMERA_2)

CHESSBOARD_SIZE = (9, 6)
CHESSBOARD_SQUARE_SIZE = 0.0245

if __name__ == "__main__":
    chessboard = Chessboard(CHESSBOARD_SIZE, CHESSBOARD_SQUARE_SIZE)

    cam1 = Camera(k=K1, dist=DIST1)
    cam2 = Camera(k=K2, dist=DIST2)

    image_1 = cv.imread(str(PATH_IMAGE_CAMERA_1), cv.IMREAD_GRAYSCALE)
    image_2 = cv.imread(str(PATH_IMAGE_CAMERA_2), cv.IMREAD_GRAYSCALE)

    debug_image_1 = np.zeros(1, dtype=np.uint8)
    debug_image_2 = np.zeros(1, dtype=np.uint8)

    cam1.calibrate_extrinsics(chessboard, image=image_1, debug_buffer=debug_image_1)
    cam2.calibrate_extrinsics(chessboard, image=image_2, debug_buffer=debug_image_2)

    np.savetxt(PATH_ROTATION_CAMERA_1, cam1.r)
    np.savetxt(PATH_ROTATION_CAMERA_2, cam2.r)
    np.savetxt(PATH_POSITION_CAMERA_1, cam1.t)
    np.savetxt(PATH_POSITION_CAMERA_2, cam2.t)

    cv.namedWindow("chessboards", cv.WINDOW_NORMAL)
    cv.imshow("chessboards", np.hstack((debug_image_1, debug_image_2)))
    cv.waitKey(0)
