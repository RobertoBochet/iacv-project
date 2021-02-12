#!/usr/bin/env python3
import logging
from pathlib import Path

from _log import logger_setup
from paintar.syncer import synchronize, estimate_delay
from ._cv import cv
import numpy as np

from paintar.canvas import Canvas
from .camera import Camera, StereoCamera

PATH_DATA = Path("./data") / "demo1"

PATH_VIDEO = PATH_DATA / "video"
PATH_PARAMETERS = PATH_DATA / "parameters"

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
T_CANVAS = np.loadtxt(PATH_PARAMETERS / "t_canvas.txt")

SKIP_FRAMES = 30 * 10 * 0
TEST_FRAMES = 1500

ARUCO_PEN_TIP_OFFSET = np.array([0, -0.139, 0, 1])
ARUCO_PEN_ID = 0
ARUCO_PEN_SIZE = 0.022
ARUCO_DICT = cv.aruco.custom_dictionary(1, 3)

SHOW_DEBUG_IMAGE = True
SHOW_CANVAS = True
SHOW_PROJECTION = True

if __name__ == "__main__":
    logger_setup(logging.DEBUG)

    cam1 = Camera(PATH_VIDEO_1.as_posix(), k=K1, dist=DIST1, r=R1, t=T1)
    cam2 = Camera(PATH_VIDEO_2.as_posix(), k=K2, dist=DIST2, r=R2, t=T2)

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
                    debug_image=SHOW_DEBUG_IMAGE,
                    t=T_CANVAS,
                    size=(400, 600),
                    resolution=2e3,
                    drawing_threshold=(0.001, 0.002),
                    brush_size=2,
                    interpolate=True
                    )

    for _ in range(SKIP_FRAMES):
        stereo_cam.grab()

    if SHOW_DEBUG_IMAGE:
        cv.namedWindow("debug image", cv.WINDOW_NORMAL)
        cv.resizeWindow("debug image", 1280, 720 // 2)
    if SHOW_CANVAS:
        cv.namedWindow("canvas", cv.WINDOW_NORMAL)
        cv.resizeWindow("canvas", 600, 400)
    if SHOW_PROJECTION:
        cv.namedWindow("projection", cv.WINDOW_NORMAL)
        cv.resizeWindow("projection", 1280, 720)

    while True:
        if not stereo_cam.grab():
            break

        canvas.loop(False)

        if cv.getWindowProperty("debug image", cv.WND_PROP_VISIBLE) >= 1:
            cv.imshow("debug image", canvas.debug_image)
        if cv.getWindowProperty("canvas", cv.WND_PROP_VISIBLE) >= 1:
            cv.imshow("canvas", canvas.canvas)
        if cv.getWindowProperty("projection", cv.WND_PROP_VISIBLE) >= 1:
            cv.imshow("projection", canvas.projection)

        cv.waitKey(1)
