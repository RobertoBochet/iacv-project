from typing import Union

import numpy as np

from ..camera import StereoCamera
from .._cv import cv
from matplotlib import pyplot as plt

from ..utilities import cart2proj


def estimate_delay(stereo_camera: StereoCamera,
                   aruco_id: int,
                   aruco_dict: cv.aruco_Dictionary = None,
                   aruco_param: cv.aruco_DetectorParameters = cv.aruco.DetectorParameters_create(),
                   max_delay: float = 10.,
                   returns_fps: bool = True,
                   plots: bool = False) -> Union[float, int]:
    """
    returns an estimate of the delay between cam1 and cam2
    a positive value means a delay of the cam2 over the cam1
    WARNING this method is destructive (reads several frames and discards them),
    after the use the delay will not corrected and it might be increased
    """

    fps = stereo_camera.cam1.get(cv.CAP_PROP_FPS)

    assert fps == stereo_camera.cam2.get(cv.CAP_PROP_FPS), "video must have the same frame rate"

    max_offset = int(max_delay * fps)

    t = np.array(range(-max_offset, max_offset + 1), dtype=int)
    cost = np.full((2 * max_offset + 1,), np.inf, dtype=float)

    for _ in range(max_offset):
        stereo_camera.cam1.grab()

    # finds the aruco in the image of the first camera
    while True:
        p1 = stereo_camera.cam1.find_aruco(aruco_id=aruco_id,
                                           aruco_dict=aruco_dict,
                                           aruco_param=aruco_param,
                                           grab=False)

        # if the aruco is found stops the research
        if p1 is not None:
            break

        stereo_camera.grab()

    for i in range(2 * max_offset + 1):
        stereo_camera.cam2.grab()

        # finds the aruco in the image of the second camera
        p2 = stereo_camera.cam2.find_aruco(aruco_id=aruco_id,
                                           aruco_dict=aruco_dict,
                                           aruco_param=aruco_param,
                                           grab=False)

        # if no aruco is detected skips the frame
        if p2 is None:
            continue

        c = 0

        for j in range(4):
            c = (cart2proj(p2[j]) @ stereo_camera.f @ cart2proj(p1[j]).reshape(3, 1)) ** 2

        cost[i] = c ** .5

    dt = t[np.argmin(cost)]

    if plots:
        plt.plot(t, cost)
        y_limits = np.array([-0.1, 1.1]) * np.nanmax(cost[cost != np.inf])
        plt.vlines(dt, *y_limits, colors="tab:orange")
        plt.show()

    if not returns_fps:
        dt = dt / fps

    return dt


def synchronize(stereo_camera: StereoCamera, delay: int):
    """
    given a delay restores the sync between the two cameras
    """
    cam = stereo_camera.cam1 if delay < 0 else stereo_camera.cam2

    for _ in range(abs(delay)):
        cam.grab()
