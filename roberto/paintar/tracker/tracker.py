import enum

import cv2.cv2 as cv
import numpy as np

from .estimators import Position3DEstimator
from .. import utilities as ut
from ..camera import StereoCamera


class Status(enum.Enum):
    CAMERA_NO_CALIBRATE = 1  # cameras need calibration
    NO_LOCK = 2  # no lock on pen or aruco is acquired
    ARUCO_DETECTED = 3  # aruco is detected but with too low accuracy
    ARUCO_LOCKED = 4  # the aruco is locked in both cameras
    TIP_LOCKED = 5  # pen's tip is locked in both cameras


class Tracker:

    def __init__(self, stereo_cam: StereoCamera,
                 aruco_dict: cv.aruco_Dictionary = None,
                 aruco_param: cv.aruco_DetectorParameters = cv.aruco.DetectorParameters_create(),
                 aruco_pen_tip_offset: np.ndarray = np.array([0, 0, 0, 1]),
                 aruco_pen_id: int = 0,
                 aruco_pen_size: float = 0.01,
                 ):
        self._aruco_pen_size = aruco_pen_size
        self._aruco_pen_id = aruco_pen_id
        self._aruco_pen_tip_offset = aruco_pen_tip_offset
        self._stereo_cam = stereo_cam
        self._aruco_dict = aruco_dict
        self._aruco_param = aruco_param

        self._aruco_estimator_1 = Position3DEstimator()
        self._aruco_estimator_2 = Position3DEstimator()

        self._aruco_locked_threshold = 0.4  # threshold on uncertain to consider aruco position locked
        self._aruco_lost_counter_init = 10  # numbers of iteration before consider lost the aruco
        self._aruco_tip_crop_size1 = 50
        self._aruco_tip_crop_size2 = 50

        self._aruco_lost_counter = 0

    @property
    def is_aruco_detected(self) -> bool:
        raise NotImplemented

    @property
    def is_pen_locked(self) -> bool:
        raise NotImplemented

    @property
    def status(self) -> Status:
        """
        returns the tracker status
        """

        if np.linalg.norm(np.diagonal(self._aruco_estimator_1.P)) < self._aruco_locked_threshold and \
                np.linalg.norm(np.diagonal(self._aruco_estimator_2.P)) < self._aruco_locked_threshold:
            return Status.ARUCO_LOCKED

        if self._aruco_lost_counter > 0:
            return Status.ARUCO_DETECTED

        if self._stereo_cam.is_calibrated:
            return Status.NO_LOCK

        return Status.CAMERA_NO_CALIBRATE

    def loop(self, grab: bool = True):
        if grab:
            self._stereo_cam.grab()

        db1, db2 = self._stereo_cam.retrieve()

        # if the aruco is consider detected,
        # decrease the counter to check if it can still be considered "detected"
        if self.status in {Status.ARUCO_DETECTED, Status.ARUCO_LOCKED}:
            self._aruco_lost_counter -= 1

            # if the counter reaches the zero,
            # resets the kalman estimators
            if self._aruco_lost_counter == 0:
                self._aruco_estimator_1.reset()
                self._aruco_estimator_2.reset()
                self._aruco_lost_counter = -1

        # if the tip is still not found searches the aruco and tracks it
        if self.status in {Status.NO_LOCK, Status.ARUCO_DETECTED, Status.ARUCO_LOCKED}:

            # searches the aruco in the cameras
            a1 = self._stereo_cam.cam1.find_aruco_pose(self._aruco_pen_id, self._aruco_pen_size, grab=False,
                                                       aruco_dict=self._aruco_dict, aruco_param=self._aruco_param,
                                                       debug_buffer=db1
                                                       )
            a2 = self._stereo_cam.cam2.find_aruco_pose(self._aruco_pen_id, self._aruco_pen_size, grab=False,
                                                       aruco_dict=self._aruco_dict, aruco_param=self._aruco_param,
                                                       debug_buffer=db2
                                                       )
            # checks if the aruco is found
            if a1 is not None and a2 is not None:
                # resets the counter to consider lost the aruco
                self._aruco_lost_counter = self._aruco_lost_counter_init

                # calculates the measurement of pen tip exploiting aruco
                p1 = ut.proj_normalization(a1 @ self._aruco_pen_tip_offset)
                p2 = ut.proj_normalization(a2 @ self._aruco_pen_tip_offset)

                # updates the estimations of tip position from aruco
                self._aruco_estimator_1.update(ut.proj2cart(p1))
                self._aruco_estimator_2.update(ut.proj2cart(p2))

                if db1 is not None:
                    p1 = ut.proj2cart(self._stereo_cam.cam1.m_c @ p1)
                    p2 = ut.proj2cart(self._stereo_cam.cam2.m_c @ p2)

                    cv.drawMarker(db1, tuple(p1.astype(np.uint)),
                                  (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10)
                    cv.drawMarker(db2, tuple(p2.astype(np.uint)),
                                  (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=10)

        if self.status == Status.ARUCO_LOCKED:
            # retrieves the aruco estimation positions in cameras projections
            p1 = ut.proj2cart(self._stereo_cam.cam1.m_c @ ut.cart2proj(self._aruco_estimator_1.x))
            p2 = ut.proj2cart(self._stereo_cam.cam2.m_c @ ut.cart2proj(self._aruco_estimator_2.x))

            if db1 is not None:
                cv.drawMarker(db1, tuple(p1.reshape(2).astype(np.uint)),
                              (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=20)
                cv.drawMarker(db2, tuple(p2.reshape(2).astype(np.uint)),
                              (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=20)

            # gets points in format (y, x)
            p1 = p1[::-1]
            p2 = p2[::-1]

            # computes the crops limits on the pen tip
            crop1_limits = ut.crop_around(p1, self._aruco_tip_crop_size1, bounds=self._stereo_cam.cam1.frame_size)
            crop2_limits = ut.crop_around(p2, self._aruco_tip_crop_size2, bounds=self._stereo_cam.cam2.frame_size)

            img1, img2 = self._stereo_cam.retrieve()

            # creates the crops around the estimate tip positions
            crop1 = img1[crop1_limits[0, 0]:crop1_limits[0, 1], crop1_limits[1, 0]:crop1_limits[1, 1]]
            crop2 = img2[crop2_limits[0, 0]:crop2_limits[0, 1], crop2_limits[1, 0]:crop2_limits[1, 1]]

            if db1 is not None:
                db1[0:250, 0:250] = cv.resize(crop1, (250, 250))
                db2[0:250, 0:250] = cv.resize(crop2, (250, 250))

            # TODO find the tip corner

        elif self.status == Status.CAMERA_NO_CALIBRATE:
            pass

        cv.imshow("main", np.concatenate((db1, db2), axis=1))
        cv.waitKey(1)

        if cv.getWindowProperty('main', cv.WND_PROP_VISIBLE) < 1:
            import sys
            sys.exit(0)
