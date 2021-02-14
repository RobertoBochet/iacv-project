from functools import cached_property
from typing import Union

import numpy as np

from . import Camera
from .._cv import cv
from ..utilities import Chessboard, cart2proj, proj2cart, normalize_points


class StereoCamera:
    """
    implements the code to handle a stereo camera
    """

    def __init__(self, cam1: Camera, cam2: Camera):
        """
        :param cam1: the first camera
        :param cam2: the second camera
        """
        self._cam1 = cam1
        self._cam2 = cam2

    @property
    def cam1(self) -> Camera:
        return self._cam1

    @property
    def cam2(self) -> Camera:
        return self._cam2

    @cached_property
    def e(self) -> Camera:
        """
        returns the essential matrix
        """
        r = self._cam2.r @ self._cam1.r.T
        t = self._cam2.t - r @ self._cam1.t

        return np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0],
        ]) @ r

    @cached_property
    def f(self) -> Camera:
        """
        returns the fundamental matrix
        the same point with the images projection p1, p2 satisfy the relation p2^T * F * p1
        """
        return np.linalg.inv(self._cam2.k.T) @ self.e @ np.linalg.inv(self._cam1.k)

    @property
    def is_calibrated(self) -> bool:
        return self._cam1.is_calibrated and self._cam2.is_calibrated

    def calibrate_extrinsics(self, chessboard: Chessboard = Chessboard(), grab: bool = True) -> bool:
        """
        calibrates the geometrical parameters R, T of the cameras
        """
        if grab:
            self.grab()

        ret1 = self._cam1.calibrate_extrinsics(chessboard, grab=False)
        ret2 = self._cam2.calibrate_extrinsics(chessboard, grab=False)

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

    def retrieve(self, *args, **kwargs) -> tuple[np.array, np.array]:
        """
        retrieves the cams' images from their frame buffers
        """
        img1 = self._cam1.retrieve(*args, **kwargs)
        img2 = self._cam2.retrieve(*args, **kwargs)

        return img1, img2

    def triangulate_points(self, x1, x2):
        """
        given two sets of points from the two cam in P^2 returns the corresponding set of points in P^3
        """
        assert len(x2) == len(x1), "Number of points does not match."
        return np.array([self.triangulate_point(x[0], x[1]) for x in zip(x1, x2)])

    def triangulate_point(self, x1, x2):
        x1 = proj2cart(x1)
        x2 = proj2cart(x2)
        point4d = cv.triangulatePoints(self._cam1.m.astype(np.float), self._cam2.m.astype(np.float),
                                       x1.reshape(2, 1).astype(np.float), x2.reshape(2, 1).astype(np.float))
        return point4d.reshape(4) / point4d[3]

    def triangulate_aruco(self, aruco_id: int, aruco_size: float,
                          grab: bool = True,
                          aruco_dict: cv.aruco_Dictionary = None,
                          aruco_param: cv.aruco_DetectorParameters = None) -> Union[np.ndarray, None]:
        """
        searches a specific aruco in the two views, triangulates it and
        returns a transformation matrix for points in the aruco frame
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

        # triangulates the aruco's points
        p = self.triangulate_points(ar1, ar2)
        p = proj2cart(p)

        # uses Kabsch algorithm
        # finds centroid of the aruco's points
        pc = p.mean(axis=0)

        # defines the points in the aruco reference frame
        # pr = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * aruco_size / 2
        # for svd is a better choice that the average distance from the origin is sqrt(2)
        pr = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]])

        # normalizes the data to reduce the error given by svd and to remove centroid
        _, p = normalize_points(cart2proj(p))
        p = proj2cart(p)

        # computes H = P^T * Q
        h = np.transpose(pr) @ p

        # decomposes H
        u, s, vh = np.linalg.svd(h)
        v = np.transpose(vh)

        # computes the sign for the reference frame
        d = 1 if np.linalg.det(v @ np.transpose(u)) > 0 else -1

        # computes the rotation matrix
        r = np.transpose(vh) @ np.diag([1, 1, d]) @ np.transpose(u)

        # composes the transformation matrix
        t = np.vstack((
            np.hstack((
                r,
                np.transpose(pc[np.newaxis])
            )),
            np.array([0, 0, 0, 1])
        ))

        return t
