from typing import Union

import numpy as np
from matplotlib import pyplot as plt

from .._cv import cv
from ..camera import StereoCamera
from ..utilities import cart2proj


def estimate_delay(stereo_camera: StereoCamera,
                   aruco_id: int,
                   aruco_dict: cv.aruco_Dictionary = None,
                   aruco_param: cv.aruco_DetectorParameters = cv.aruco.DetectorParameters_create(),
                   max_delay: float = 10.,
                   number_frames: int = 3,
                   delta_frames: int = 10,
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

    frames_span = np.array(range(0, number_frames * delta_frames, delta_frames))

    max_offset = int(max_delay * fps)

    t1 = max_offset

    # finds the aruco in the image of the first camera
    while True:
        p1 = []
        for d in frames_span:
            stereo_camera.cam1.set(cv.CAP_PROP_POS_FRAMES, t1 + d)

            pa = stereo_camera.cam1.find_aruco(aruco_id=aruco_id,
                                               aruco_dict=aruco_dict,
                                               aruco_param=aruco_param,
                                               grab=True)

            if pa is None:
                break

            p1.append(pa)

        if len(p1) == len(frames_span):
            break

        t1 += 1

    p1 = np.vstack(p1)
    p1 = cart2proj(p1)

    t = np.array(range(-max_offset, max_offset + 1), dtype=int)
    cost = np.full(t.shape, np.inf, dtype=float)

    # finds the aruco in the image of the first camera
    for dt in t:
        p2 = []
        for d in frames_span:
            stereo_camera.cam2.set(cv.CAP_PROP_POS_FRAMES, t1 + dt + d)

            pa = stereo_camera.cam2.find_aruco(aruco_id=aruco_id,
                                               aruco_dict=aruco_dict,
                                               aruco_param=aruco_param,
                                               grab=True)

            if pa is None:
                break

            p2.append(pa)

        if len(p2) != len(frames_span):
            continue

        p2 = np.vstack(p2)
        p2 = cart2proj(p2)

        c = map(lambda x: x[1] @ stereo_camera.f @ x[0].reshape(3, 1), zip(p1, p2))
        c = map(lambda x: float(x), c)
        c = map(lambda x: abs(x), c)
        c = map(lambda x: x**.5, c)
        c = sum(c)

        cost[dt - t[0]] = c

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
    if delay > 0:
        stereo_camera.cam1.set(cv.CAP_PROP_POS_FRAMES, 0)
        stereo_camera.cam2.set(cv.CAP_PROP_POS_FRAMES, delay)

    else:
        stereo_camera.cam1.set(cv.CAP_PROP_POS_FRAMES, -delay)
        stereo_camera.cam2.set(cv.CAP_PROP_POS_FRAMES, 0)
