#!/usr/bin/env python3
from pathlib import Path

import cv2 as cv
import numpy as np

from paintar.camera import Camera, StereoCamera
from paintar.syncer import synchronize, estimate_delay

PATH_DATA = Path("./data") / "demo3-desync"

PATH_VIDEO = PATH_DATA / "video"
PATH_VIDEO_1 = PATH_VIDEO / "video1.avi"
PATH_VIDEO_2 = PATH_VIDEO / "video2.avi"

PATH_PARAMETERS = PATH_DATA / "parameters"

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

ARUCO_PEN_ID = 0
ARUCO_DICT = cv.aruco.custom_dictionary(1, 3)

if __name__ == "__main__":
    cam1 = Camera(PATH_VIDEO_1.as_posix(), k=K1, dist=DIST1, r=R1, t=T1)
    cam2 = Camera(PATH_VIDEO_2.as_posix(), k=K2, dist=DIST2, r=R2, t=T2)

    stereo_cam = StereoCamera(cam1, cam2)

    dt = estimate_delay(stereo_cam,
                        aruco_dict=ARUCO_DICT,
                        aruco_id=ARUCO_PEN_ID,
                        max_delay=3,
                        plots=True)

    synchronize(stereo_cam, dt)

    while stereo_cam.grab():
        img = np.hstack(stereo_cam.retrieve())
        cv.imshow("video", img)
        cv.waitKey(1)
