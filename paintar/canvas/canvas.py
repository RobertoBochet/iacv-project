from functools import cached_property
from typing import Union

import numpy as np
import scipy as sp

from .._cv import cv
from ..camera import StereoCamera
from ..tracker import Tracker, State
from ..utilities import cart2proj, proj2cart, draw_axis


class Canvas(Tracker):
    """
    implements the canvas behaviour extends the `Tracker` class
    """

    def __init__(self,
                 stereo_cam: StereoCamera,
                 size: tuple = (100, 100),
                 t_c: np.ndarray = None,
                 reverse_z: bool = True,
                 resolution: float = 1e3,
                 drawing_threshold: Union[float, tuple[float, float]] = 0.001,
                 brush_size: int = 0,
                 interpolate: bool = False,
                 **kwargs):
        """
        :param stereo_cam: A `StereoCamera` instance
        :param size: The canvas' size in pixels
        :param t_c: It is the isometric transformation between world reference frame and the canvas one,
            if `None` the two references frames coincide
        :param reverse_z: If `True` the z values will be reversed for the drawing status computation,
            it is useful because the `cv2` images reference frame has the axis z entering in the image
        :param resolution: It is the canvas pixels number per meters
        :param drawing_threshold: It is the threshold on the z values to define the `is_drawing` status,
            if they are provided two value, these are used as hysteresis cycle parameters
        :param brush_size: The size in pixels of the brush
        :param interpolate: Defines if the points will be interpolated by lines
        :param `**kwargs`: These arguments are given to the `Tracker` class
        """
        super(Canvas, self).__init__(stereo_cam, **kwargs)

        self._interpolate = interpolate
        self._size = size
        self._t_c = t_c
        self._reverse_z = reverse_z
        self._resolution = resolution
        self._brush_size = brush_size

        # if only one threshold is provided uses it for both the hysteresis thresholds
        self._drawing_threshold = drawing_threshold if isinstance(drawing_threshold, tuple) \
            else (drawing_threshold, drawing_threshold)

        self._is_drawing = False

        self._old_pos = None

        # prepares the canvas
        self._canvas = None
        self.clear()

    @property
    def size_meters(self) -> tuple[float, float]:
        """
        the canvas' size in meters
        """
        return (self._size[0] / self._resolution,
                self._size[1] / self._resolution)

    @cached_property
    def _t_c_inv(self) -> np.ndarray:
        return np.linalg.inv(self._t_c)

    @cached_property
    def p1(self) -> np.ndarray:
        """
        the projective matrix from canvas reference frame of the camera1
        """
        return self._p(self._stereo_cam.cam1.m)

    @cached_property
    def p2(self) -> np.ndarray:
        """
        the projective matrix from canvas reference frame of the camera1
        """
        return self._p(self._stereo_cam.cam2.m)

    def _p(self, m: np.ndarray) -> np.ndarray:
        """
        given a projective matrix from world reference frame, returns the projective matrix from canvas reference frame
        """
        m = m @ self._t_c
        p = np.delete(m, 2, 1)
        p = p @ sp.linalg.block_diag(np.eye(2) / self._resolution, 1)
        return p

    @property
    def canvas(self) -> np.ndarray:
        """
        the canvas image
        """
        return self._canvas

    @property
    def projection(self) -> np.ndarray:
        """
        the projection of the canvas in the camera 1 image
        """
        img1 = self._stereo_cam.cam1.retrieve(clone=True)
        a = cv.warpPerspective(self._canvas, self.p1, img1.shape[1::-1], flags=cv.INTER_NEAREST)
        img1[a == 1.] = (0, 0, 0)

        return img1

    @property
    def tip(self) -> np.ndarray:
        """
        the tip position in the canvas frame
        """
        tip = super(Canvas, self).tip

        # if no transformation was provided returns the `Tracker` estimation
        if self._t_c is None:
            return tip

        # moves the tip in the canvas reference frame
        tip_c = self._t_c_inv @ cart2proj(tip)
        return proj2cart(tip_c)

    @property
    def is_drawing(self) -> bool:
        """
        if the pen is drawing
        """
        if self.state is not State.TIP_LOCKED:
            return False

        z = -self.tip[2] if self._reverse_z else self.tip[2]

        # implements the hysteresis cycle
        if self._is_drawing:
            self._is_drawing = z <= self._drawing_threshold[1]
        else:
            self._is_drawing = z <= self._drawing_threshold[0]

        return self._is_drawing

    def clear(self) -> None:
        """
        clears the canvas
        """
        self._canvas = np.zeros(self._size)

    def loop(self, grab: bool = True) -> None:
        """
        processes a single frame

        :param grab: if `True` a grab operation will be performed
        """
        super(Canvas, self).loop(grab)

        if self.is_drawing:
            # changes (x,y) tip coordinates from meters unit to pixels unit
            p_c = self.tip[0:2] * self._resolution

            # converts the coordinates to an `int` `tuple`
            p_c = p_c.T[0]
            p_c = np.around(p_c)
            p_c = tuple(p_c.astype(int))

            # if interpolation is required also the last tip position is used to draw
            if self._interpolate and self._old_pos is not None:
                cv.line(self._canvas, self._old_pos, p_c, (1,), thickness=self._brush_size)
            else:
                cv.circle(self._canvas, p_c, radius=self._brush_size, color=(1,), thickness=-1)

            self._old_pos = p_c
        else:
            self._old_pos = None

    def _draw_debug_image(self) -> None:
        super(Canvas, self)._draw_debug_image()

        if self._t_c is not None:
            draw_axis(self._db1, self._stereo_cam.cam1.m @ self._t_c, draw_z=False)
            draw_axis(self._db2, self._stereo_cam.cam2.m @ self._t_c, draw_z=False)
