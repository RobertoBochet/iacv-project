import cv2.cv2 as cv
import numpy as np
from . import Camera

from ..utilities import Chessboard


class StereoCamera:
    def __init__(self, cam1: Camera, cam2: Camera):
        self._cam1 = cam1
        self._cam2 = cam2

        self._r = None
        self._t = None
        self._e = None
        self._f = None

    @property
    def cam1(self) -> Camera:
        return self._cam1

    @property
    def cam2(self) -> Camera:
        return self._cam2

    @property
    def f(self) -> Camera:
        return self._f

    def calibrate(self, chessboard: Chessboard = Chessboard()) -> None:
        img1, img2 = self.shot()

        chessboard_points = chessboard.get_points()

        # img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        # img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        _, corners1 = cv.findChessboardCorners(img1, chessboard.size, None)
        _, corners2 = cv.findChessboardCorners(img2, chessboard.size, None)

        _, _, _, _, _, self._r, self._t, self._e, self._f = cv.stereoCalibrate([chessboard_points],
                                                                               [corners1], [corners2],
                                                                               self._cam1.k, self._cam1.dist,
                                                                               self._cam2.k, self._cam2.dist,
                                                                               None, flags=cv.CALIB_FIX_INTRINSIC)

    def shot(self) -> tuple[np.array, np.array]:
        self._cam1.grab()
        self._cam1.grab()

        _, img1 = self._cam1.retrieve()
        _, img2 = self._cam1.retrieve()

        return img1, img2
