#!/usr/bin/env python3
from pathlib import Path

from paintar.syncer import synchronize, estimate_delay
from ._cv import cv
import numpy as np

from paintar.canvas import Canvas
from .camera import Camera, StereoCamera

PATH_VIDEO = Path("./data/video")
PATH_PARAMETERS = Path("./data/parameters")

PATH_VIDEO_1 = PATH_VIDEO / "video1.avi"
PATH_VIDEO_2 = PATH_VIDEO / "video2.avi"

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

SKIP_FRAMES = 30 * 10 * 0
TEST_FRAMES = 1500

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
    cam1 = Camera(PATH_VIDEO_1.as_posix(), k=K1, dist=DIST1, r=R1, t=T1)
    cam2 = Camera(PATH_VIDEO_2.as_posix(), k=K2, dist=DIST2, r=R2, t=T2)

    stereo_cam = StereoCamera(cam1, cam2)

    # to test de-sync
    # for _ in range(40):
    #    cam1.grab()
    if True:
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
                    size=(400, 600),
                    resolution=2e3,
                    drawing_threshold=(0.001, 0.002),
                    brush_size=2,
                    interpolate=True,
                    plot=True,
                    project=True)

    for _ in range(SKIP_FRAMES):
        stereo_cam.grab()

    while True:
        if not stereo_cam.grab():
            break

        if not canvas.loop(False):
            break

        cv.waitKey(1)
