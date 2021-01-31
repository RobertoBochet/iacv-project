from typing import Union

import filterpy.kalman as fpk
import numpy as np
import scipy as sp


class PositionSpeed3DEstimator(fpk.KalmanFilter):
    def __init__(self,
                 init_xp: Union[np.array, float] = 0.,
                 init_xs: Union[np.array, float] = 0.,
                 init_p: Union[np.array, float] = 1e3,
                 r_p: Union[np.array, float] = 1.,
                 r_s: Union[np.array, float] = 1.,
                 q_p: Union[np.array, float] = 1.,
                 q_s: Union[np.array, float] = 1.):
        super().__init__(6, 6)

        init_xp = init_xp if isinstance(init_xp, np.ndarray) else np.ones((3, 1), dtype=float) * init_xp
        init_xs = init_xs if isinstance(init_xs, np.ndarray) else np.ones((3, 1), dtype=float) * init_xs

        self._init_x = sp.linalg.block_diag(init_xp, init_xs)
        self._init_p = init_p if isinstance(init_p, np.ndarray) else np.eye(6, dtype=float) * init_p

        self.F = np.array([[1, 0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0],
                           [0, 0, 1, 0, 0, 1],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        self.H = np.diag([1, 1, 1, 0, 0, 0])

        r_p = r_p if isinstance(r_p, np.ndarray) else np.eye(3, dtype=float) * r_p
        r_s = r_s if isinstance(r_s, np.ndarray) else np.eye(3, dtype=float) * r_s
        self.R = sp.linalg.block_diag(r_p, r_s)

        q_p = q_p if isinstance(q_p, np.ndarray) else np.eye(3, dtype=float) * q_p
        q_s = q_s if isinstance(q_s, np.ndarray) else np.eye(3, dtype=float) * q_s
        self.Q = sp.linalg.block_diag(q_p, q_s)

        self.reset()

    def reset(self):
        self.x = self._init_x
        self.P = self._init_p

    def predict(self, **kwargs):
        super(Position3DEstimator, self).predict(**kwargs)

        if self.P > self._init_p:
            self.reset()

    @property
    def is_reset(self) -> bool:
        return np.array_equal(self.x, self._init_x) and np.array_equal(self.P, self._init_p)
