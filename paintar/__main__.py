#!/usr/bin/env python3
import time
from pathlib import Path

from paintar.syncer import synchronize, estimate_delay
from ._cv import cv
import numpy as np

from paintar.canvas import Canvas
from . import utilities as ut
from .camera import Camera, StereoCamera

PATH_VIDEO = Path("./video")
PATH_VIDEO_1 = PATH_VIDEO / "video1_2.avi"
PATH_VIDEO_2 = PATH_VIDEO / "video2_2.avi"
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
        np.array([[.15, .25, 0.0015]]).T
    ], [0, 0, 0, 1]
])

if __name__ == "__main__":
    chessboard = ut.Chessboard(CHESSBOARD_SIZE, CHESSBOARD_SQUARE_SIZE)

    cam1 = Camera(PATH_VIDEO_1.as_posix(), K1, DIST1, R1, T1)
    cam2 = Camera(PATH_VIDEO_2.as_posix(), K2, DIST2, R2, T2)

    stereo_cam = StereoCamera(cam1, cam2)

    # to test de-sync
    # for _ in range(40):
    #    cam1.grab()
    if False:
        dt = estimate_delay(stereo_cam,
                            aruco_dict=ARUCO_DICT,
                            aruco_id=ARUCO_PEN_ID,
                            max_delay=3.,
                            plots=True)

        cam1.release()
        cam2.release()

        cam1.open(PATH_VIDEO_1.as_posix())
        cam2.open(PATH_VIDEO_2.as_posix())

        synchronize(stereo_cam, dt)

    # while True:
    #     if stereo_cam.calibrate_geometry(chessboard):
    #         break

    canvas = Canvas(stereo_cam,
                    aruco_dict=ARUCO_DICT,
                    aruco_pen_size=ARUCO_PEN_SIZE,
                    aruco_pen_tip_offset=ARUCO_PEN_TIP_OFFSET,
                    debug_image=True,
                    t=T_CANVAS,
                    size=(200, 300),
                    resolution=1e3,
                    drawing_threshold=(0.001, 0.003),
                    brush_size=3,
                    interpolate=True,
                    plots=True,
                    projects=True)

    print(canvas.size_meters)

    for _ in range(SKIP_FRAMES):
        stereo_cam.grab()

    while True:
        t_i = time.time()

        if not stereo_cam.grab():
            break

        if not canvas.loop(False):
            break

        cv.waitKey(1)
