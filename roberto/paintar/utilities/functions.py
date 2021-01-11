import numpy as np


def cart2proj(x) -> np.array:
    """
    converts one or several points from cartesian 2D/3D to projective P^2/P^3
    """
    if np.ndim(x) == 1:
        return np.hstack((x, [1]))
    else:
        return np.hstack((x, np.ones((x.shape[0], 1))))


def proj2cart(x) -> np.array:
    """
    converts one or several points from projective P^2/P^3 to cartesian 2D/3D
    """
    x = proj_normalization(x)
    if np.ndim(x) == 1:
        return x[:-2]
    else:
        return x[:, :-2]


def proj_normalization(x) -> np.array:
    """
    returns the normalized version of a vector or list of vectors in the projective space
    """
    if np.ndim(x) == 1:
        return x / x[-1]
    else:
        return (x.T / x[:, -1]).T
