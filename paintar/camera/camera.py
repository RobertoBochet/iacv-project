from typing import Union

from .._cv import cv
import numpy as np

from ..utilities import Chessboard, forge_isometry, forge_projective_matrix, crop_around


class Camera(cv.VideoCapture):
    def __init__(self, source: any,
                 k: np.array = None, dist: np.array = None,
                 r: np.array = None, t: np.array = None,
                 *args, **kwargs):
        super().__init__(source, *args, **kwargs)

        self._frame_in_buffer = False
        self._frame_buffer = None
        self._k = k
        self._dist = dist
        self._r = r
        self._t = t

        map_x, map_y = cv.initUndistortRectifyMap(self._k, self._dist, np.eye(3), self._k, (1280,720), cv.CV_16SC2)
        self._ur_map_x = map_x
        self._ur_map_y = map_y

        self._frame_size = np.array([int(self.get(cv.CAP_PROP_FRAME_HEIGHT)),
                                     int(self.get(cv.CAP_PROP_FRAME_WIDTH))], dtype=int)

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
    def a(self) -> np.ndarray:
        """
        provides the geometrical transformation A_b^c (i.e. p^c = A_b^c p^b),
        the transformation of the base frame respect of camera frame
        """
        assert self._r is not None and self._t is not None, "camera must be calibrated"
        return forge_isometry(self._r, self._t)

    @property
    def m_c(self) -> np.ndarray:
        """
        provides the camera matrix M (i.e. x^c = M X^c),
        the transformation from point in P^3 referred to camera frame to point in P^2 in camera projection
        """
        assert self._k is not None, "camera must be calibrated"
        return forge_projective_matrix(self._k)

    @property
    def m(self) -> np.ndarray:
        """
        provides the camera matrix M_b^c (i.e. x^c = M_b^c X^b),
        the transformation from point in P^3 referred to base frame to point in P^2 in camera projection
        """
        assert self._k is not None and self._r is not None and self._t is not None, "camera must be calibrated"
        return forge_projective_matrix(self._k, r=self._r, t=self._t)

    @property
    def is_calibrated(self) -> bool:
        return self._k is not None and self._t is not None and self._r is not None

    @property
    def frame_size(self) -> np.ndarray:
        """
        returns the frame size
        """
        # TODO needs more test
        return self._frame_size

    def retrieve(self, clone: bool = False, *args, **kwargs):
        if not self._frame_in_buffer:
            _, img = super(Camera, self).retrieve(*args, **kwargs)
            self._frame_buffer = cv.remap(img, self._ur_map_x, self._ur_map_y, cv.INTER_LINEAR);
            self._frame_in_buffer = True

        if clone:
            return np.copy(self._frame_buffer)

        return self._frame_buffer

    def grab(self) -> bool:
        self._frame_in_buffer = False
        return super(Camera, self).grab()

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

        ret, r, t, _ = cv.solvePnPRansac(chessboard.get_points(), corners, self._k, None)

        self._r, _ = cv.Rodrigues(r)
        self._t = t.reshape(3)

        return ret

    def find_aruco(self, aruco_id: int, grab: bool = True,
                   aruco_dict: cv.aruco_Dictionary = None,
                   aruco_param: cv.aruco_DetectorParameters = None) -> Union[None, np.ndarray]:
        if grab:
            self.grab()

        img = self.retrieve()

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.equalizeHist(img)

        corners, ids, _ = cv.aruco.detectMarkers(img, aruco_dict, parameters=aruco_param, cameraMatrix=self._k)
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

    def find_aruco_well(self, *args, **kwargs) -> Union[None, np.ndarray]:
        points = self.find_aruco(*args, **kwargs)

        if points is None:
            return None

        points = points.astype(int)

        img = self.retrieve()

        p1, crop1 = crop_around(img, points[0], 10)
        p2, crop2 = crop_around(img, points[1], 10)
        p3, crop3 = crop_around(img, points[2], 10)
        p4, crop4 = crop_around(img, points[3], 10)

        crop = np.vstack((
            np.hstack((crop1, crop2)),
            np.hstack((crop3, crop4)),
        ))
        # TODO: complete the wip
        # cv.imshow("main", crop)
        # cv.waitKey(0)

    def find_aruco_pose(self, aruco_id: int, marker_size: float, debug_buffer: np.array = None, **kwargs):
        """
        provides the geometrical transformation A_a^c (i.e. p^c = A_a^c p^a),
        the transformation of the aruco frame respect of camera frame
        """
        aruco = self.find_aruco(aruco_id, **kwargs)

        if aruco is None:
            return None

        r, t, _ = cv.aruco.estimatePoseSingleMarkers([aruco], marker_size, self._k)

        if debug_buffer is not None:
            if debug_buffer.size == 1:
                _, img = self.retrieve()
                debug_buffer.resize(img.shape, refcheck=False)
                np.copyto(debug_buffer, img, casting="unsafe")
            cv.aruco.drawAxis(debug_buffer, self._k, r, t, 2 * marker_size)

        r, _ = cv.Rodrigues(r[0])
        t = t.reshape(3)

        return forge_isometry(r, t)
