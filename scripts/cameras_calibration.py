#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import cv2 as cv

from paintar.camera import Camera
from paintar.utilities import Chessboard

PATH_DATA = Path("./data")

PATH_IMAGES = PATH_DATA / "chessboard"
PATH_PARAMETERS = PATH_DATA / "demo2" / "parameters"

PATH_IMAGES_CAMERA_1 = PATH_IMAGES / "camera1"
PATH_IMAGES_CAMERA_2 = PATH_IMAGES / "camera2"

PATH_PARAMETERS_CAMERA_1 = PATH_PARAMETERS / "camera1"
PATH_PARAMETERS_CAMERA_2 = PATH_PARAMETERS / "camera2"

PATH_INTRINSICS_CAMERA_1 = PATH_PARAMETERS_CAMERA_1 / "intrinsics.txt"
PATH_INTRINSICS_CAMERA_2 = PATH_PARAMETERS_CAMERA_2 / "intrinsics.txt"

PATH_DISTORTION_CAMERA_1 = PATH_PARAMETERS_CAMERA_1 / "distortion.txt"
PATH_DISTORTION_CAMERA_2 = PATH_PARAMETERS_CAMERA_2 / "distortion.txt"

CHESSBOARD_SIZE = (9, 6)
CHESSBOARD_SQUARE_SIZE = 0.0245

if __name__ == "__main__":
    chessboard = Chessboard(CHESSBOARD_SIZE, CHESSBOARD_SQUARE_SIZE)

    cam1 = Camera()
    cam2 = Camera()

    path_images_1 = PATH_IMAGES_CAMERA_1.glob("*.jpg")
    path_images_2 = PATH_IMAGES_CAMERA_2.glob("*.jpg")

    images_1 = np.array([cv.imread(str(path), cv.IMREAD_GRAYSCALE) for path in path_images_1])
    images_2 = np.array([cv.imread(str(path), cv.IMREAD_GRAYSCALE) for path in path_images_2])

    cam1.calibrate(images_1, chessboard)
    cam2.calibrate(images_2, chessboard)

    PATH_PARAMETERS_CAMERA_1.mkdir(parents=True, exist_ok=True)
    PATH_PARAMETERS_CAMERA_2.mkdir(parents=True, exist_ok=True)

    np.savetxt(PATH_INTRINSICS_CAMERA_1, cam1.k)
    np.savetxt(PATH_INTRINSICS_CAMERA_2, cam2.k)
    np.savetxt(PATH_DISTORTION_CAMERA_1, cam1.dist)
    np.savetxt(PATH_DISTORTION_CAMERA_2, cam2.dist)
