from functools import cached_property
import numpy as np
from .._cv import cv

from ..tracker import Tracker, Status
from ..utilities import cart2proj, proj2cart
from ..camera import StereoCamera


class Canvas(Tracker):
    def __init__(self,
                 stereo_cam: StereoCamera,
                 size: np.ndarray = np.array([100, 100]),
                 limits: np.ndarray = np.array([.5, .5]),
                 brush_size: int = 0,
                 interpolate: bool = False,
                 *args, **kwargs):
        super(Canvas, self).__init__(stereo_cam, *args, **kwargs)

        self._interpolate = interpolate
        self._size = size
        self._brush_size = brush_size

        self._canvas = None
        self._limits = None

        self._drawing_z_limit = 0.003

        self.limits = limits

        self.clear()

    @property
    def limits(self) -> np.ndarray:
        return self._limits

    @limits.setter
    def limits(self, limits: np.ndarray):
        assert limits.shape == (2, 2) or limits.shape == (2,), "limits have to be one or two points"
        if limits.size == 4:
            self._limits = limits
        else:
            self._limits = np.vstack((np.array([0, 0]), limits))

        # resets t
        if "t" in self.__dict__:
            del self.__dict__["t"]

    @cached_property
    def t(self) -> np.ndarray:
        s = np.linalg.norm(self._size) / np.linalg.norm(self.limits[1] - self.limits[0])
        t = ((- self.limits[0])[np.newaxis]).T

        return np.vstack((
            np.hstack((
                s * np.eye(2), t
            )),
            np.array([0, 0, 1])))

    @property
    def is_drawing(self):
        return self.status is Status.TIP_LOCKED and self._estimator_tip.pos[2] < self._drawing_z_limit

    def clear(self):
        self._canvas = np.zeros(tuple(self._size))

    def loop(self, grab: bool = True) -> bool:
        super(Canvas, self).loop(grab)

        if self.is_drawing:
            p = cart2proj(self.tip[0:2])

            p_c = proj2cart(self.t @ p)

            p_c = p_c.T[0]
            p_c = np.around(p_c)
            p_c = tuple(p_c.astype(int))

            cv.circle(self._canvas, p_c, radius=self._brush_size, color=(1,), thickness=-1)

        cv.imshow("canvas", self._canvas)
        cv.waitKey(1)

        return not cv.getWindowProperty("canvas", cv.WND_PROP_VISIBLE) < 1
