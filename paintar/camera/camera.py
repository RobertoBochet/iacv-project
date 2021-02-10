from functools import cached_property
from typing import Union

from .._cv import cv
import numpy as np

from ..utilities import Chessboard, forge_isometry, forge_projective_matrix, crop_around, draw_axis


class Camera(cv.VideoCapture):
    def __init__(self, *args,
                 k: np.array = None, dist: np.array = None,
                 r: np.array = None, t: np.array = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._frame_in_buffer = False
        self._frame_buffer = None
        self._k = k
        self._dist = dist
        self._r = r
        self._t = t

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

    @cached_property
    def frame_size(self) -> tuple[int, int]:
        """
        returns the frame size
        """
        return (int(self.get(cv.CAP_PROP_FRAME_HEIGHT)),
                int(self.get(cv.CAP_PROP_FRAME_WIDTH)))

    @cached_property
    def undistort_rectify_map(self):
        return cv.initUndistortRectifyMap(self._k, self._dist, np.eye(3), self._k, self.frame_size[::-1], cv.CV_16SC2)

    def retrieve_raw(self, clone: bool = False, *args, **kwargs):
        _, img = super(Camera, self).retrieve(*args, **kwargs)

        if clone:
            return np.copy(self._frame_buffer)

        return img

    def retrieve(self, clone: bool = False, *args, **kwargs):
        if not self._frame_in_buffer:
            _, img = super(Camera, self).retrieve(*args, **kwargs)
            self._frame_buffer = cv.remap(img, *self.undistort_rectify_map, cv.INTER_LINEAR)
            self._frame_in_buffer = True

        if clone:
            return self._frame_buffer.copy()

        return self._frame_buffer

    def grab(self) -> bool:
        self._frame_in_buffer = False
        return super(Camera, self).grab()

    def calibrate(self, images: np.ndarray, chessboard: Chessboard) -> bool:
        img_size = images.shape[1:3]

        images_points = []

        for i in range(len(images)):
            img = images[i]

            if np.ndim(img) == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(img, chessboard.size)

            if not ret:
                continue

            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)

            images_points.append(corners)

        chessboard_points = [chessboard.get_points() for _ in range(len(images_points))]

        ret, k, dist, _, _ = cv.calibrateCamera(chessboard_points, images_points, img_size[::-1], None, None)

        if not ret:
            return False

        self._k = k
        self._dist = dist

        return True

    def calibrate_extrinsics(self, chessboard: Chessboard,
                             image: np.ndarray = None,
                             grab: bool = True,
                             debug_buffer: np.array = None) -> bool:

        if image is None:
            if grab:
                self.grab()

            _, image = self.retrieve_raw()

        if np.ndim(image) == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(image, chessboard.size)

        if not ret:
            return False

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)

        ret, r, t, _ = cv.solvePnPRansac(chessboard.get_points(), corners, self._k, self._dist)

        self._r, _ = cv.Rodrigues(r)
        self._t = t.reshape(3)

        if debug_buffer is not None and ret:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
            debug_buffer.resize(image.shape, refcheck=False)
            np.copyto(debug_buffer, image, casting="unsafe")
            cv.drawChessboardCorners(debug_buffer, chessboard.size, corners, True)
            draw_axis(debug_buffer, self.m)

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
