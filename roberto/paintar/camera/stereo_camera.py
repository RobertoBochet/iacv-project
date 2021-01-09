import cv2.cv2 as cv
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
        """
        shortcut for (optionally) grab and retrieve
        """
        if grab:
            self.grab()

        return self.retrieve()

    def grab(self) -> None:
        """
        puts cams' images simultaneously in their frame buffers
        """
        self._cam1.grab()
        self._cam2.grab()

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
        m = np.zeros([3 * 2, 4 + 2])
        for i, (x, p) in enumerate([(x1, self._cam1.p), (x2, self._cam2.p)]):
            m[3 * i:3 * i + 3, :4] = p
            m[3 * i:3 * i + 3, 4 + i] = -x
        v = np.linalg.svd(m)[-1]
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
