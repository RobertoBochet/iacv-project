import cv2.cv2 as cv
import numpy as np
from . import Camera

from ..utilities import Chessboard


class StereoCamera:
    def __init__(self, cam1: Camera, cam2: Camera):
        self._cam1 = cam1
        self._cam2 = cam2

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
        raise NotImplemented

    def calibrate(self, chessboard: Chessboard = Chessboard()) -> None:
        self._cam1.calibrate_geometry(chessboard)
        self._cam2.calibrate_geometry(chessboard)
        # img1, img2 = self.shot()
        #
        # chessboard_points = chessboard.get_points()
        #
        # # img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        # # img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        #
        # _, corners1 = cv.findChessboardCorners(img1, chessboard.size, None)
        # _, corners2 = cv.findChessboardCorners(img2, chessboard.size, None)
        #
        # _, _, _, _, _, self._r, self._t, self._e, self._f = cv.stereoCalibrate([chessboard_points],
        #                                                                        [corners1], [corners2],
        #                                                                        self._cam1.k, self._cam1.dist,
        #                                                                        self._cam2.k, self._cam2.dist,
        #                                                                        None, flags=cv.CALIB_FIX_INTRINSIC)

    def shot(self, grab: bool = True) -> tuple[np.array, np.array]:
        if grab:
            self.grab()

        return self.retrieve()

    def grab(self) -> None:
        self._cam1.grab()
        self._cam2.grab()

    def retrieve(self) -> tuple[np.array, np.array]:
        _, img1 = self._cam1.retrieve()
        _, img2 = self._cam2.retrieve()

        return img1, img2

    def triangulate_points(self, x1, x2):
        """
        Two-view triangulation of points in
        x1,x2 (nx3 homog. coordinates).
        Similar to openCV triangulatePoints.
        """
        assert len(x2) == len(x1), "Number of points don't match."
        return np.array([self.triangulate_point(x[0], x[1]) for x in zip(x1, x2)])

    def triangulate_point(self, x1, x2):
        m = np.zeros([3*2, 4+2])
        for i, (x, p) in enumerate([(x1, self._cam1.p), (x2, self._cam2.p)]):
            m[3*i:3*i+3, :4] = p
            m[3*i:3*i+3, 4+i] = -x
        v = np.linalg.svd(m)[-1]
        x = v[-1, :4]
        return x / x[3]
