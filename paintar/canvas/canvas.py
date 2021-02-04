from functools import cached_property
from typing import Union

import numpy as np

from .._cv import cv

from ..tracker import Tracker, Status
from ..utilities import cart2proj, proj2cart, draw_axis
from ..camera import StereoCamera


class Canvas(Tracker):
    def __init__(self,
                 stereo_cam: StereoCamera,
                 size: tuple = (100, 100),
                 t: np.ndarray = None,
                 reverse_z: bool = True,
                 resolution: float = 1e3,
                 drawing_threshold: Union[float, tuple[float, float]] = 0.001,
                 brush_size: int = 0,
                 interpolate: bool = False,
                 plots: bool = False,
                 **kwargs):
        """
        :param stereo_cam: A `StereoCamera` instance
        :param size: The canvas' size in pixels
        :param t: It is the isometric transformation between world reference frame and the canvas one,
            if `None` the two references frames coincide
        :param reverse_z: If `True` the z values will be reversed for the drawing status computation,
            it is useful because the `cv2` images reference frame has the axis z entering in the image
        :param resolution: It is the canvas pixels number per meters
        :param drawing_threshold: It is the threshold on the z values to define the `is_drawing` status,
            if they are provided two value, these are used as hysteresis cycle parameters
        :param brush_size: The size in pixels of the brush
        :param interpolate: Defines if the points will be interpolated by lines
        :param plots: Defines if the canvas has to be plotted on screen
        :param `**kwargs`: These arguments are given to the `Tracker` class
        """
        super(Canvas, self).__init__(stereo_cam, **kwargs)

        self._interpolate = interpolate
        self._size = size
        self._t = t
        self._reverse_z = reverse_z
        self._resolution = resolution
        self._brush_size = brush_size
        self._plots = plots

        # if only one threshold is provided uses it for both the hysteresis thresholds
        self._drawing_threshold = drawing_threshold if isinstance(drawing_threshold, tuple) \
            else (drawing_threshold, drawing_threshold)

        self._is_drawing = False

        self._old_pos = None

        self._canvas = None
        self.clear()

        self._window_name = "canvas"
        if self._plots:
            cv.namedWindow(self._window_name, cv.WINDOW_NORMAL)
            cv.resizeWindow(self._window_name, *self._size[::-1])
            cv.imshow(self._window_name, self._canvas)

    @property
    def size_meters(self) -> tuple[float, float]:
        """Returns the canvas' size in meters"""
        return (self._size[0] / self._resolution,
                self._size[1] / self._resolution)

    @cached_property
    def _t_inv(self) -> np.ndarray:
        return np.linalg.inv(self._t)

    @property
    def tip(self) -> np.ndarray:
        """Returns the tip position in the canvas frame"""
        tip = super(Canvas, self).tip

        # if no transformation was provided returns the `Tracker` estimation
        if self._t is None:
            return tip

        # moves the tip in the canvas reference frame
        tip_c = self._t_inv @ cart2proj(tip)
        return proj2cart(tip_c)

    @property
    def is_drawing(self) -> bool:
        """Returns if the pen is drawing"""
        if self.status is not Status.TIP_LOCKED:
            return False

        z = -self.tip[2] if self._reverse_z else self.tip[2]

        # implements the hysteresis cycle
        if self._is_drawing:
            self._is_drawing = z <= self._drawing_threshold[1]
        else:
            self._is_drawing = z <= self._drawing_threshold[0]

        return self._is_drawing

    def clear(self) -> None:
        """Clears the canvas"""
        self._canvas = np.zeros(self._size)

    def loop(self, grab: bool = True) -> bool:
        """Processes a single frame

        :param grab: If `True` a grab operation will be done
        :return: `False` if the canvas window was closed
        """
        # if the canvas window was closed, exits the loop
        if self._plots and cv.getWindowProperty(self._window_name, cv.WND_PROP_VISIBLE) < 1:
            return False

        super(Canvas, self).loop(grab)

        if self.is_drawing:
            # changes (x,y) tip coordinates from meters unit to pixels unit
            p_c = self.tip[0:2] * self._resolution

            # converts the coordinates to an `int` `tuple`
            p_c = p_c.T[0]
            p_c = np.around(p_c)
            p_c = tuple(p_c.astype(int))

            if self._interpolate and self._old_pos is not None:
                cv.line(self._canvas, self._old_pos, p_c, (1,), thickness=self._brush_size)
            else:
                cv.circle(self._canvas, p_c, radius=self._brush_size, color=(1,), thickness=-1)

            self._old_pos = p_c
        else:
            self._old_pos = None

        if self._plots:
            cv.imshow(self._window_name, self._canvas)

        return True

    def _draw_debug_image(self):
        super(Canvas, self)._draw_debug_image()

        if self._t is not None:
            draw_axis(self._db1, self._stereo_cam.cam1.m @ self._t)
            draw_axis(self._db2, self._stereo_cam.cam2.m @ self._t)
