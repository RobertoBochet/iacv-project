import enum
import logging
import time
from typing import Union

import numpy as np
from .._cv import cv

from .estimators import PositionSpeed3DEstimator
from .. import utilities as ut
from ..camera import StereoCamera
from ..utilities import draw_axis

_LOGGER = logging.getLogger(__package__)
_LOGGER_FPS = logging.getLogger("{__package__}.fps")


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
                 debug_image: bool = False,
                 fps_window_size: int = 40
                 ):
        self._aruco_pen_size = aruco_pen_size
        self._aruco_pen_id = aruco_pen_id
        self._aruco_pen_tip_offset = aruco_pen_tip_offset
        self._stereo_cam = stereo_cam
        self._aruco_dict = aruco_dict
        self._aruco_param = aruco_param
        self._debug_image = debug_image

        self._variance_threshold_aruco = 0.005
        self._variance_threshold_feature = 0.003
        self._variance_measure_aruco = 0.005
        self._variance_measure_feature = 0.0005

        self._estimator_tip = PositionSpeed3DEstimator(r_p=self._variance_measure_aruco,
                                                       q_p=0.0001,
                                                       q_s=.00001)

        self._aruco_tip_crop_size1 = 60
        self._aruco_tip_crop_size2 = 60

        self._fps_last_frame = 0.
        self._fps_time = np.zeros((fps_window_size,))

        self._db1 = None
        self._db2 = None

    @property
    def text_info(self) -> str:
        text = ""

        if self.status.value >= Status.ARUCO_DETECTED.value:
            text += "  tip Z: {:+.4f}\n".format(float(self._estimator_tip.pos[2]))
            text += "tip inc: {:.4f}\n".format(self._estimator_tip.var)

        text += " status: {}\n".format(self.status.name)
        text += "    fps: {:.1f}".format(self.fps)
        return text

    @property
    def status(self) -> Status:
        """
        returns the tracker status
        """
        if self._estimator_tip.var < self._variance_threshold_feature:
            return Status.TIP_LOCKED

        if self._estimator_tip.var < self._variance_threshold_aruco:
            return Status.ARUCO_LOCKED

        if not self._estimator_tip.is_reset:
            return Status.ARUCO_DETECTED

        if self._stereo_cam.is_calibrated:
            return Status.NO_LOCK

        return Status.CAMERA_NO_CALIBRATE

    @property
    def tip(self) -> np.ndarray:
        return self._estimator_tip.pos

    @property
    def fps(self) -> float:
        return 1 / self._fps_time.mean()

    @property
    def debug_image(self):
        assert self._debug_image, "debug image must be enabled"

        self._draw_debug_image()

        img = np.concatenate((self._db1, self._db2), axis=1)
        self._write_info(img)

        return img

    def loop(self, grab: bool = True) -> bool:
        self._update_fps()
        _LOGGER_FPS.info(">>{:02.02f}fps".format(self.fps))

        if grab:
            self._stereo_cam.grab()

        if self._debug_image:
            self._db1, self._db2 = self._stereo_cam.retrieve(clone=True)

        if self.status == Status.CAMERA_NO_CALIBRATE:
            raise NotImplemented

        self._estimator_tip.predict()

        ###### almost NO_LOCK ######
        if Status.NO_LOCK.value <= self.status.value < Status.TIP_LOCKED.value:
            self._looking_for_aruco()

        ###### almost ARUCO_DETECTED ######
        if self.status.value >= Status.ARUCO_DETECTED.value:
            pass

        ###### almost ARUCO_LOCKED ######
        if self.status.value >= Status.ARUCO_LOCKED.value:
            self._looking_for_tip()

        ###### almost TIP_LOCKED ######
        if self.status.value >= Status.TIP_LOCKED.value:
            pass

        return True

    def _looking_for_aruco(self):
        # looks for the aruco in the image and tries to estimate its pose
        t = self._stereo_cam.triangulate_aruco(self._aruco_pen_id, self._aruco_pen_size, grab=False,
                                               aruco_dict=self._aruco_dict, aruco_param=self._aruco_param)

        # if aruco pose was estimated successfully, updates the tip position
        if t is not None:
            p = t @ self._aruco_pen_tip_offset
            self._estimator_tip.update(ut.proj2cart(p))

            if self._debug_image:
                # draws the aruco reference frame in the debug images
                draw_axis(self._db1, self._stereo_cam.cam1.m @ t)
                draw_axis(self._db2, self._stereo_cam.cam2.m @ t)

                p1 = ut.proj2cart(self._stereo_cam.cam1.m @ p)
                p2 = ut.proj2cart(self._stereo_cam.cam2.m @ p)

                cv.drawMarker(self._db1, tuple(p1.reshape(2).astype(np.uint)),
                              (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=20)
                cv.drawMarker(self._db2, tuple(p2.reshape(2).astype(np.uint)),
                              (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=20)

    def _looking_for_tip(self):
        p = ut.cart2proj(self._estimator_tip.pos)

        p1 = ut.proj2cart(self._stereo_cam.cam1.m @ p)
        p2 = ut.proj2cart(self._stereo_cam.cam2.m @ p)

        img1, img2 = self._stereo_cam.retrieve(clone=False)

        corner1, crop1 = ut.crop_around(img1, p1, self._aruco_tip_crop_size1, clone=True)
        corner2, crop2 = ut.crop_around(img2, p2, self._aruco_tip_crop_size2, clone=True)

        # requires rejected only if debug is active
        rejected1 = np.empty(0) if self._debug_image else None
        rejected2 = np.empty(0) if self._debug_image else None

        tip1 = self._detect_tip_feature(crop1, rejected=rejected1)
        tip2 = self._detect_tip_feature(crop2, rejected=rejected2)

        if tip1 is not None and tip2 is not None:
            # updates the estimation of tip position from triangulation
            tip = self._stereo_cam.triangulate_point(ut.cart2proj(tip1 + corner1), ut.cart2proj(tip2 + corner2))
            self._estimator_tip.update(ut.proj2cart(tip), R=self._variance_measure_feature)

            if self._debug_image:
                cv.drawMarker(crop1, tuple(tip1.astype(np.uint)),
                              (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=200)
                cv.drawMarker(crop2, tuple(tip2.astype(np.uint)),
                              (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=200)

                if rejected1.size > 0 and rejected2.size > 0:
                    for i in range(rejected1.shape[0]):
                        cv.drawMarker(crop1, tuple(rejected1[i].astype(np.uint)),
                                      (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=20)
                    for i in range(rejected2.shape[0]):
                        cv.drawMarker(crop2, tuple(rejected2[i].astype(np.uint)),
                                      (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=20)

        if self._debug_image:
            crop_res = (250, 250)
            self._db1[0:crop_res[0], 0:crop_res[1]] = cv.resize(crop1, crop_res, cv.INTER_NEAREST)
            self._db2[0:crop_res[0], 0:crop_res[1]] = cv.resize(crop2, crop_res, cv.INTER_NEAREST)

    def _detect_tip_feature(self, tip_area: np.ndarray, rejected: np.ndarray = None) -> Union[np.ndarray, None]:

        hsv = cv.cvtColor(tip_area, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 40]))
        mask = cv.bitwise_not(mask)
        mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)  # needed by bitwise_or
        cv.bitwise_or(mask_rgb, tip_area, tip_area)  # replace everything brighter than V = 40 with pure white

        tip_area_gray = cv.cvtColor(tip_area, cv.COLOR_BGR2GRAY)  # needed by shi-tomasi detector
        features = cv.goodFeaturesToTrack(tip_area_gray, 3, .3,
                                          5)  # top 3 features are returned as an array of [x,y] points

        # if no feature is detected return None
        if features is None:
            return None

        features = np.squeeze(features, axis=1)

        circle_kernel = cv.getStructuringElement(cv.MORPH_RECT, (11, 11))  # basically a 7x7 matrix of 1s
        filtered_area = cv.filter2D(mask, cv.CV_16S, circle_kernel, anchor=(5, 0))
        # brightest -> less dark around the candidate -> high likelihood of being the tip
        # normalize the results from 0 to 255
        filtered_area = cv.normalize(filtered_area, filtered_area, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                     dtype=cv.CV_8U)

        # sort by brightness in the filtered area
        features = np.array(
            sorted(features, key=lambda feature: filtered_area[feature[1].astype(np.uint), feature[0].astype(np.uint)],
                   reverse=True))

        # if rejected elements are required put them in the array
        if rejected is not None and features.shape[0] > 1:
            rej_feat = features[1:]
            rejected.resize(rej_feat.shape, refcheck=False)
            np.copyto(rejected, rej_feat, casting="unsafe")

        # filtered_area = cv.cvtColor(filtered_area, cv.COLOR_GRAY2BGR)

        tip = features[0] + np.array([0, 2])

        return tip

    def _draw_debug_image(self):
        # draws the world frame
        draw_axis(self._db1, self._stereo_cam.cam1.m)
        draw_axis(self._db2, self._stereo_cam.cam2.m)

        # draws the estimation of the tip position
        p = ut.cart2proj(self._estimator_tip.pos)

        p1 = ut.proj2cart(self._stereo_cam.cam1.m @ p)
        p2 = ut.proj2cart(self._stereo_cam.cam2.m @ p)

        p1 = tuple(p1.reshape(2).astype(np.uint))
        p2 = tuple(p2.reshape(2).astype(np.uint))

        cv.drawMarker(self._db1, p1, (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=20)
        cv.drawMarker(self._db2, p2, (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=20)

    def _write_info(self, img: np.ndarray):
        text = self.text_info.splitlines()
        dp = np.array([0, 40])
        pos = np.array([0, img.shape[0]]) + np.array([5, -5]) - dp * len(text)
        for i in range(len(text)):
            pos += dp
            cv.putText(img, text[i], tuple(pos),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    def _update_fps(self):
        self._fps_time = np.roll(self._fps_time, 1)
        t = time.time()
        self._fps_time[0] = t - self._fps_last_frame
        self._fps_last_frame = t
