import cv2 as cv
import numpy as np

from . import Camera
from ..utilities import Chessboard, cart2proj


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

    @property
    def is_calibrated(self) -> bool:
        return self._cam1.is_calibrated and self._cam2.is_calibrated

    def calibrate_geometry(self, chessboard: Chessboard = Chessboard(), grab: bool = True) -> bool:
        """
        calibrates the geometrical parameters R, T of the cameras
        """
        if grab:
            self.grab()

        ret1 = self._cam1.calibrate_geometry(chessboard, grab=False)
        ret2 = self._cam2.calibrate_geometry(chessboard, grab=False)

        return ret1 and ret2

    def shot(self, grab: bool = True) -> tuple[np.array, np.array]:
        """
        shortcut for (optionally) grab and retrieve
        """
        if grab:
            self.grab()

        return self.retrieve()

    def grab(self) -> bool:
        """
        puts cams' images simultaneously in their frame buffers
        """
        ret1 = self._cam1.grab()
        ret2 = self._cam2.grab()

        return ret1 and ret2

    def retrieve(self) -> tuple[np.array, np.array]:
        """
        retrieves the cams' images from their frame buffers
        """
        _, img1 = self._cam1.retrieve()
        _, img2 = self._cam2.retrieve()

        return img1, img2

    def retrieve_undistorted(self) -> tuple[np.array, np.array]:
        """
        retrieves the undistorted cams' images from their frame buffers
        """
        return self._cam1.retrieve_undistorted(), self._cam2.retrieve_undistorted()

    def triangulate_points(self, x1, x2):
        """
        given two sets of points from the two cam in P^2 returns the corresponding set of points in P^3
        """
        assert len(x2) == len(x1), "Number of points don't match."
        return np.array([self.triangulate_point(x[0], x[1]) for x in zip(x1, x2)])

    def triangulate_point(self, x1, x2):
        """
        given two points from the two cam in P^2 returns the corresponding point in P^3
        """
        A = np.zeros([3 * 2, 4 + 2])
        for i, (x, p) in enumerate([(x1, self._cam1.m), (x2, self._cam2.m)]):
            A[3 * i:3 * i + 3, :4] = p
            A[3 * i:3 * i + 3, 4 + i] = -x
        v = np.linalg.svd(A)[-1]
        x = v[-1, :4]
        return x / x[3]

    def triangulate_aruco(self, aruco_id: int, grab: bool = True,
                          aruco_dict: cv.aruco_Dictionary = None,
                          aruco_param: cv.aruco_DetectorParameters = None) -> np.array:
        """
        searches a specific aruco in the two views and returns the 4 corners in P^3
        """
        if grab:
            self.grab()

        ar1 = self._cam1.find_aruco(aruco_id, grab=False,
                                    aruco_dict=aruco_dict, aruco_param=aruco_param)
        ar2 = self._cam2.find_aruco(aruco_id, grab=False,
                                    aruco_dict=aruco_dict, aruco_param=aruco_param)

        if ar1 is None or ar2 is None:
            # aruco is not found in both the views
            return None

        # moves aruco into P^2 space
        ar1 = cart2proj(ar1)
        ar2 = cart2proj(ar2)

        return self.triangulate_points(ar1, ar2)

        return cv.triangulatePoints(self._cam1.p, self._cam2.p, ar1[0], ar2[0])
