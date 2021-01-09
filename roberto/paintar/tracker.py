import cv2.cv2 as cv

from .camera import StereoCamera


class Tracker:
    def __init__(self, stereo_cam: StereoCamera,
                 aruco_dict: cv.aruco_Dictionary,
                 aruco_param: cv.aruco_DetectorParameters):
        self._stereo_cam = stereo_cam
        self._aruco_dict = aruco_dict
        self._aruco_param = aruco_param

    # def triangulate_marker(self, aruco_id: int, grab: bool = True):
    #     ar1 = self._stereo_cam.cam1.find_aruco(aruco_id, grab=False,
    #                                            aruco_dict=self._aruco_dict, aruco_param=self._aruco_param)
    #     ar2 = self._stereo_cam.cam2.find_aruco(aruco_id, grab=False,
    #                                            aruco_dict=self._aruco_dict, aruco_param=self._aruco_param)
    #
    #     print(ar1)
    #     return None, None
    #
    #     # img1, img2 = self._stereo_cam.shot(grab)
    #     #
    #     # img1, img2 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY), cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    #     # img1, img2 = cv.equalizeHist(img1), cv.equalizeHist(img2)
    #     #
    #     # corners1, ids1, rejected1 = cv.aruco.detectMarkers(img1, self._aruco_dict, parameters=self._aruco_param,
    #     #                                                    cameraMatrix=self._stereo_cam.cam1.k,
    #     #                                                    distCoeff=self._stereo_cam.cam1.dist)
    #     # corners2, ids2, rejected2 = cv.aruco.detectMarkers(img2, self._aruco_dict, parameters=self._aruco_param,
    #     #                                                    cameraMatrix=self._stereo_cam.cam2.k,
    #     #                                                    distCoeff=self._stereo_cam.cam2.dist)
    #     #
    #     # if ids1 is None or ids2 is None:
    #     #     return None, None
    #     #
    #     # ids1 = np.squeeze(ids1, axis=-1)
    #     # ids2 = np.squeeze(ids2, axis=-1)
    #     #
    #     # indx1 = np.where(ids1 == aruco_id)[0]
    #     # indx2 = np.where(ids2 == aruco_id)[0]
    #     #
    #     # if len(indx1) == 0 or len(indx2) == 0:
    #     #     # no aruco detect in both cameras
    #     #     return None, None
    #     #
    #     # if len(indx1) > 1 or len(indx2) > 1:
    #     #     # multiple aruco with searched id found
    #     #     return None, None
    #     #
    #     # return [corners1[indx1[0]]], [corners2[indx2[0]]]
