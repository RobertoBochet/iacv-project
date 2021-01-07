import cv2.cv2 as cv

from .camera import StereoCamera


class Tracker:
    def __init__(self, stereo_cam: StereoCamera,
                 aruco_dict: cv.aruco_Dictionary,
                 aruco_param):
        self._stereo_cam = stereo_cam
        self._aurco_dict = aruco_dict
        self._aruco_param = aruco_param

    def triangulate_marker(self):
        img1, img2 = self._stereo_cam.shot()

        img1, img2 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY), cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        img1, img2 = cv.equalizeHist(img1), cv.equalizeHist(img2)

        corners1, ids1, rejected1 = cv.aruco.detectMarkers(img1, self._aruco_dict, parameters=self._aruco_param,
                                                           cameraMatrix=self._stereo_cam.cam1.k,
                                                           distCoeff=self._stereo_cam.cam1.dist)
        corners2, ids2, rejected2 = cv.aruco.detectMarkers(img2, self._aruco_dict, parameters=self._aruco_param,
                                                           cameraMatrix=self._stereo_cam.cam2.k,
                                                           distCoeff=self._stereo_cam.cam2.dist)
