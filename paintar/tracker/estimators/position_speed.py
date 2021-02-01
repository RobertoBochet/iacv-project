from typing import Union

import filterpy.kalman as fpk
import numpy as np
import scipy as sp


class PositionSpeed3DEstimator(fpk.KalmanFilter):
    def __init__(self,
                 init_xp: Union[np.array, float] = 0.,
                 init_xs: Union[np.array, float] = 0.,
                 init_pp: Union[np.array, float] = 1e3,
                 init_ps: Union[np.array, float] = .5,
                 r_p: Union[np.array, float] = 1.,
                 r_s: Union[np.array, float] = 1e3,
                 q_p: Union[np.array, float] = 1.,
                 q_s: Union[np.array, float] = 1.):
        super().__init__(6, 6)

        init_xp = init_xp if isinstance(init_xp, np.ndarray) else np.ones((3, 1), dtype=float) * init_xp
        init_xs = init_xs if isinstance(init_xs, np.ndarray) else np.ones((3, 1), dtype=float) * init_xs

        init_pp = init_pp if isinstance(init_pp, np.ndarray) else np.eye(3, dtype=float) * init_pp
        init_ps = init_ps if isinstance(init_ps, np.ndarray) else np.eye(3, dtype=float) * init_ps

        self._init_x = np.vstack((init_xp, init_xs))
        self._init_p = sp.linalg.block_diag(init_pp, init_ps)

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

        self._is_reset = True

        self.reset()

    def reset(self):
        self.x = self._init_x
        self.P = self._init_p
        self._is_reset = True

    def update(self, z, **kwargs):
        if z.size == 3:
            z = np.hstack((np.squeeze(z), np.array([0, 0, 0])))

        super(PositionSpeed3DEstimator, self).update(z, **kwargs)

        self._is_reset = False

    def predict(self, **kwargs):
        super(PositionSpeed3DEstimator, self).predict(**kwargs)
        self._is_reset = False

        if np.linalg.norm(self.P) > np.linalg.norm(self._init_p):
            self.reset()

    @property
    def is_reset(self) -> bool:
        return self._is_reset
