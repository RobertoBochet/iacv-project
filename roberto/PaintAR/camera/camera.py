import cv2.cv2 as cv
import numpy as np


class Camera(cv.VideoCapture):
    def __init__(self, source: any,
                 k: np.array = None, dist: np.array = None,
                 *args, **kwargs):
        super().__init__(source, *args, **kwargs)

        self._k = k
        self._dist = dist

    @property
    def k(self) -> np.array:
        return self._k

    @property
    def dist(self) -> np.array:
        return self._dist

    def calibrate(self):
        raise NotImplemented
