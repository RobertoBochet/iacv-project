import cv2.cv2 as cv
import numpy as np

from ..utilities import Chessboard, forge_isometry, forge_projective_matrix


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
    def a(self) -> np.array:
        """
        provides the geometrical transformation A_b^c (i.e. p^c = A_b^c p^b),
        the transformation of the base frame respect of camera frame
        """
        assert self._r is not None and self._t is not None, "camera must be calibrated"
        return forge_isometry(self._r, self._t)

    @property
    def m(self) -> np.array:
        """
        provides the camera matrix M_b^c (i.e. x^c = M_b^c X^b),
        the transformation from point in P^3 referred to base frame to point in P^2 in camera projection
        """
        assert self._k is not None and self._r is not None and self._t is not None, "camera must be calibrated"
        return forge_projective_matrix(self._k, r=self._r, t=self._t)

    def shot(self) -> np.array:
        _, img = self.read()
        return img

    def retrieve_undistorted(self, *args, **kwargs):
        _, img = self.retrieve(*args, **kwargs)
        return cv.undistort(img, self._k, self._dist)

    def calibrate(self):
        raise NotImplemented

    def calibrate_geometry(self, chessboard: Chessboard, grab: bool = True, debug_buffer: np.array = None) -> bool:
        if grab:
            self.grab()

        _, img = self.retrieve()
        ret, corners = cv.findChessboardCorners(img, chessboard.size, None)

        if not ret:
            return False

        if debug_buffer is not None:
            debug_buffer.resize(img.shape, refcheck=False)
            np.copyto(debug_buffer, img, casting="no")
            cv.drawChessboardCorners(debug_buffer, chessboard.size, corners, True)

        ret, r, t, _ = cv.solvePnPRansac(chessboard.get_points(), corners, self._k, self._dist)

        self._r, _ = cv.Rodrigues(r)
        self._t = t.reshape(3)

        return ret

    def find_aruco(self, aruco_id: int, grab: bool = True,
                   aruco_dict: cv.aruco_Dictionary = None,
                   aruco_param: cv.aruco_DetectorParameters = None) -> np.array:
        if grab:
            self.grab()

        _, img = self.retrieve()

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.equalizeHist(img)

        corners, ids, _ = cv.aruco.detectMarkers(img, aruco_dict, parameters=aruco_param,
                                                 cameraMatrix=self._k,
                                                 distCoeff=self._dist
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

    def find_aruco_pose(self, aruco_id: int, marker_size: float, debug_buffer: np.array = None, **kwargs):
        """
        provides the geometrical transformation A_a^c (i.e. p^c = A_a^c p^a),
        the transformation of the aruco frame respect of camera frame
        """
        aruco = self.find_aruco(aruco_id, **kwargs)

        if aruco is None:
            return None

        r, t, _ = cv.aruco.estimatePoseSingleMarkers([aruco], marker_size, self._k, self._dist)

        if debug_buffer is not None:
            _, img = self.retrieve()
            debug_buffer.resize(img.shape, refcheck=False)
            np.copyto(debug_buffer, img, casting="unsafe")
            cv.aruco.drawAxis(debug_buffer, self._k, self._dist, r, t, 2 * marker_size)

        r, _ = cv.Rodrigues(r[0])
        t = t.reshape(3)

        return forge_isometry(r, t)
