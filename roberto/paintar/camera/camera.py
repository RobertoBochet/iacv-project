import cv2.cv2 as cv
import numpy as np

from ..utilities import Chessboard


class Camera(cv.VideoCapture):
    def __init__(self, source: any,
                 k: np.array = None, dist: np.array = None,
                 *args, **kwargs):
        super().__init__(source, *args, **kwargs)

        self._k = k
        self._dist = dist
        self._r = None
        self._t = None

    @property
    def k(self) -> np.array:
        return self._k

    @property
    def dist(self) -> np.array:
        return self._dist

    @property
    def r(self) -> np.array:
        return self._r

    @property
    def t(self) -> np.array:
        return self._t

    @property
    def p(self) -> np.array:
        assert self._k is not None and self._r is not None and self._t is not None, "camera must be calibrated"
        return self._k @ np.hstack((self._r, self._t))

    def shot(self) -> np.array:
        _, img = self.read()
        return img

    def retrieve_undistorted(self, *args, **kwargs):
        _, img = self.retrieve(*args, **kwargs)
        return cv.undistort(img, self._k, self._dist)

    def calibrate(self):
        raise NotImplemented

    def calibrate_geometry(self, chessboard: Chessboard) -> None:
        img = self.shot()
        _, corners = cv.findChessboardCorners(img, chessboard.size, None)
        _, r, self._t, _ = cv.solvePnPRansac(chessboard.get_points(), corners, self._k, self._dist)

        self._r, _ = cv.Rodrigues(r)

    def find_aruco(self, aruco_id: int, grab: bool = True,
                   aruco_dict: cv.aruco_Dictionary = None,
                   aruco_param: cv.aruco_DetectorParameters = None) -> np.array:
        if grab:
            self.grab()

        img = self.retrieve_undistorted()

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.equalizeHist(img)

        corners, ids, _ = cv.aruco.detectMarkers(img, aruco_dict, parameters=aruco_param,
                                                 cameraMatrix=self._k,
                                                 # distCoeff=self._dist
                                                 )
        if ids is None:
            # no aruco detected
            return None

        aruco = list(filter(lambda i: i[0] == aruco_id, zip(list(ids), list(corners))))

        if len(aruco) == 0:
            # no aruco with searched id found
            return None

        if len(aruco) > 1:
            # multiple aruco with searched id found
            return None

        return np.squeeze(aruco[0][1])
