import enum
from typing import Union

try:
    import cv2.cv2 as cv
except ModuleNotFoundError:
    import cv2 as cv

import numpy as np

from .estimators import Position3DEstimator
from .. import utilities as ut
from ..camera import StereoCamera
from ..utilities import draw_axis


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

        self._aruco_estimator_tip = Position3DEstimator(r=5, q=1)

        self._aruco_threshold_locked = 40  # threshold on uncertain to consider aruco position locked
        self._aruco_lost_counter_init = 10  # numbers of iteration before consider lost the aruco
        self._aruco_tip_crop_size1 = 30
        self._aruco_tip_crop_size2 = 30

        self._aruco_lost_counter = 0

        self._tip_locked = False

    @property
    def is_aruco_detected(self) -> bool:
        raise NotImplemented

    @property
    def is_pen_locked(self) -> bool:
        raise NotImplemented

    @property
    def text_info(self) -> str:
        text = ""
        if self.status in {Status.ARUCO_DETECTED, Status.ARUCO_LOCKED, Status.TIP_LOCKED}:
            text += "aruco_inc: {:.4f}\n".format(
                np.linalg.norm(np.diagonal(self._aruco_estimator_tip.P))
            )

        text += "status: {}".format(self.status.name)
        return text

    @property
    def status(self) -> Status:
        """
        returns the tracker status
        """
        if self._tip_locked == True:
            return Status.TIP_LOCKED

        if np.linalg.norm(np.diagonal(self._aruco_estimator_tip.P)) < self._aruco_threshold_locked:
            return Status.ARUCO_LOCKED

        if not self._aruco_estimator_tip.is_reset:
            return Status.ARUCO_DETECTED

        if self._stereo_cam.is_calibrated:
            return Status.NO_LOCK

        return Status.CAMERA_NO_CALIBRATE

    def loop(self, grab: bool = True):
        if grab:
            self._stereo_cam.grab()

        db1, db2 = self._stereo_cam.retrieve()

        if self.status == Status.CAMERA_NO_CALIBRATE:
            raise NotImplemented

        draw_axis(db1, self._stereo_cam.cam1.m)
        draw_axis(db2, self._stereo_cam.cam2.m)
        ###### almost NO_LOCK ######
        if self.status.value >= Status.NO_LOCK.value and self.status.value < Status.TIP_LOCKED.value:

            t = self._stereo_cam.triangulate_aruco(self._aruco_pen_id, self._aruco_pen_size, grab=False,
                                                   aruco_dict=self._aruco_dict, aruco_param=self._aruco_param)
            if t is not None:
                draw_axis(db1, self._stereo_cam.cam1.m @ t)
                draw_axis(db2, self._stereo_cam.cam2.m @ t)

                p = t @ self._aruco_pen_tip_offset

                if self.status.value < Status.ARUCO_LOCKED.value:
                    self._aruco_estimator_tip.update(ut.proj2cart(p))

                p1 = ut.proj2cart(self._stereo_cam.cam1.m @ p)
                p2 = ut.proj2cart(self._stereo_cam.cam2.m @ p)

                cv.drawMarker(db1, tuple(p1.reshape(2).astype(np.uint)),
                              (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=20)
                cv.drawMarker(db2, tuple(p2.reshape(2).astype(np.uint)),
                              (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=20)

        ###### almost ARUCO_DETECTED ######
        if self.status.value >= Status.ARUCO_DETECTED.value:
            self._aruco_estimator_tip.predict()

            if db1 is not None:
                p = ut.cart2proj(self._aruco_estimator_tip.x)

                p1 = ut.proj2cart(self._stereo_cam.cam1.m @ p)
                p2 = ut.proj2cart(self._stereo_cam.cam2.m @ p)

                cv.drawMarker(db1, tuple(p1.reshape(2).astype(np.uint)),
                              (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=20)
                cv.drawMarker(db2, tuple(p2.reshape(2).astype(np.uint)),
                              (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=20)

        ###### almost ARUCO_LOCKED ######
        if self.status.value == Status.ARUCO_LOCKED.value:
            p = ut.cart2proj(self._aruco_estimator_tip.x)

            p1 = ut.proj2cart(self._stereo_cam.cam1.m @ p)
            p2 = ut.proj2cart(self._stereo_cam.cam2.m @ p)

            # gets points in format (y, x) aka (row, col)
            p1 = p1[::-1]
            p2 = p2[::-1]

            crop1, corner1, crop2, corner2 = self.crop_around_tip(p1, p2)

            rejected1 = np.empty(0)
            rejected2 = np.empty(0)

            tip1, crop1 = self.detect_tip_feature(crop1, rejected=rejected1)
            tip2, crop2 = self.detect_tip_feature(crop2, rejected=rejected2)

            if tip1 is not None and tip2 is not None:
                # updates the estimation of tip position from triangulation
                tip = self._stereo_cam.triangulate_point(ut.cart2proj(tip1 + corner1), ut.cart2proj(tip2 + corner2))
                self._aruco_estimator_tip.update(ut.proj2cart(tip))
                self._tip_locked = True

                if db1 is not None:
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

                    # plots the reprojection of the 3d estimate of the tip on both images
                    # testp1 = ut.proj2cart(self._stereo_cam.cam1.m @ tip)
                    # testp2 = ut.proj2cart(self._stereo_cam.cam2.m @ tip)
                    # cv.drawMarker(db1, tuple(testp1.astype(np.uint)),
                    #                 (255, 0, 0), markerType=cv.MARKER_CROSS, markerSize=10)
                    # cv.drawMarker(db2, tuple(testp2.astype(np.uint)),
                    #                 (255, 0, 0), markerType=cv.MARKER_CROSS, markerSize=10)

                    db1[0:250, 0:250] = cv.resize(crop1, (250, 250))
                    db2[0:250, 0:250] = cv.resize(crop2, (250, 250))

        ###### almost TIP_LOCKED ######
        if self.status.value >= Status.TIP_LOCKED.value:
            # project the direct (ie not through aruco) tip estimation. x is in world (sheet) reference frame
            p1 = ut.proj2cart(self._stereo_cam.cam1.m @ ut.cart2proj(self._aruco_estimator_tip.x))
            p2 = ut.proj2cart(self._stereo_cam.cam2.m @ ut.cart2proj(self._aruco_estimator_tip.x))

            # gets points in format (y, x) aka (row, col)
            p1 = p1[::-1]
            p2 = p2[::-1]

            crop1, corner1, crop2, corner2 = self.crop_around_tip(p1, p2)

            rejected1 = np.empty(0)
            rejected2 = np.empty(0)

            tip1, crop1 = self.detect_tip_feature(crop1, rejected=rejected1)
            tip2, crop2 = self.detect_tip_feature(crop2, rejected=rejected2)

            # updates the estimation of tip position from triangulation
            tip = self._stereo_cam.triangulate_point(ut.cart2proj(tip1 + corner1), ut.cart2proj(tip2 + corner2))
            self._aruco_estimator_tip.update(ut.proj2cart(tip))

            if db1 is not None:
                cv.drawMarker(crop1, tuple(tip1.astype(np.uint)),
                              (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=20)
                cv.drawMarker(crop2, tuple(tip2.astype(np.uint)),
                              (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=20)

                if rejected1.size > 0 and rejected2.size > 0:
                    for i in range(rejected1.shape[0]):
                        cv.drawMarker(crop1, tuple(rejected1[i].astype(np.uint)),
                                      (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=20)
                    for i in range(rejected2.shape[0]):
                        cv.drawMarker(crop2, tuple(rejected2[i].astype(np.uint)),
                                      (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=20)

                db1[0:250, 0:250] = cv.resize(crop1, (250, 250))
                db2[0:250, 0:250] = cv.resize(crop2, (250, 250))

        img = np.concatenate((db1, db2), axis=1)

        self._write_info(img)

        cv.imshow("main", img)
        cv.waitKey(1)

        return not cv.getWindowProperty('main', cv.WND_PROP_VISIBLE) < 1

    def crop_around_tip(self, p1, p2):
        # computes the crops limits on the pen tip
        crop1_limits = ut.crop_around(p1, self._aruco_tip_crop_size1, bounds=self._stereo_cam.cam1.frame_size)
        crop2_limits = ut.crop_around(p2, self._aruco_tip_crop_size2, bounds=self._stereo_cam.cam2.frame_size)

        img1, img2 = self._stereo_cam.retrieve()

        # creates the crops around the estimate tip positions
        crop1 = np.copy(img1[crop1_limits[0, 0]:crop1_limits[0, 1], crop1_limits[1, 0]:crop1_limits[1, 1]])
        crop2 = np.copy(img2[crop2_limits[0, 0]:crop2_limits[0, 1], crop2_limits[1, 0]:crop2_limits[1, 1]])

        # extract x,y coordinates of the top left corner of the cropped area
        corner1 = np.array([crop1_limits[1, 0], crop1_limits[0, 0]])
        corner2 = np.array([crop2_limits[1, 0], crop2_limits[0, 0]])

        return crop1, corner1, crop2, corner2

    def detect_tip_feature(self, tip_area: np.ndarray, rejected: np.ndarray = None) -> tuple[Union[np.ndarray, None], np.ndarray]:

        hsv = cv.cvtColor(tip_area, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 40]))
        mask = cv.bitwise_not(mask)
        maskRGB = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)  # needed by bitwise_or
        tip_area = cv.bitwise_or(maskRGB, tip_area)  # replace everthing brighter than V = 40 with pure white

        tip_area_gray = cv.cvtColor(tip_area, cv.COLOR_BGR2GRAY)  # needed by shi-tomasi detector
        features = cv.goodFeaturesToTrack(tip_area_gray, 3, .3, 5)  # top 3 features are returned as an array of [x,y] points

        # if no feature is detected return None
        if features is None:
            return None

        features = np.squeeze(features, axis=1)

        circle_kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))  # basically a 7x7 matrix of 1s
        filtered_area = cv.filter2D(mask, cv.CV_16S, circle_kernel)
        # brightest -> less dark around the candidate -> high likelihood of being the tip
        # normalize the results from 0 to 255
        filtered_area = cv.normalize(filtered_area, filtered_area, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

        # sort by brightness in the filtered area
        features = np.array(sorted(features, key=lambda feature: filtered_area[feature[1].astype(np.uint), feature[0].astype(np.uint)], reverse=True))

        # if rejected elements are required put them in the array
        if rejected is not None and features.shape[0] > 1:
            rej_feat = features[1:]
            rejected.resize(rej_feat.shape, refcheck=False)
            np.copyto(rejected, rej_feat, casting="unsafe")

        filtered_area = cv.cvtColor(filtered_area, cv.COLOR_GRAY2BGR)
        return features[0], filtered_area

    def _write_info(self, img: np.ndarray) -> np.ndarray:
        text = self.text_info.splitlines()
        dp = np.array([0, 20])
        pos = np.array([0, img.shape[0]]) + np.array([5, -5]) - dp * len(text)
        for i in range(len(text)):
            pos += dp
            cv.putText(img, text[i], tuple(pos),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
