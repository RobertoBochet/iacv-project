import numpy as np


def cart2proj(x) -> np.array:
    """
    converts one or several points from cartesian 2D/3D to projective P^2/P^3
    """
    if np.ndim(x) == 1:
        return np.hstack((x, [1]))
    else:
        return np.hstack((x, np.ones((x.shape[0], 1))))
