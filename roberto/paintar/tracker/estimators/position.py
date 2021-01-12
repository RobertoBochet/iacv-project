from typing import Union

import filterpy.kalman as fpk
import numpy as np


class Position3DEstimator(fpk.KalmanFilter):
    def __init__(self, init_x: Union[np.array, float] = 0., init_p: Union[np.array, float] = 1000.):
        super().__init__(3, 3)

        self._init_x = init_x if isinstance(init_x, np.ndarray) else np.ones((3, 1), dtype=float) * init_x
        self._init_p = init_p if isinstance(init_p, np.ndarray) else np.eye(3, dtype=float) * init_p

        # the new tip position will be somewhat close to the old one -> no explicit dynamics
        self.F = np.eye(3, dtype=float)
        self.H = np.eye(3, dtype=float)
        self.R = np.eye(3, dtype=float) * 5
        self.Q = np.eye(3, dtype=float) * 10

        self.reset()

    def reset(self):
        self.x = self._init_x
        self.P = self._init_p
