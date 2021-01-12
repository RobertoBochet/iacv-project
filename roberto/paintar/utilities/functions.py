import numpy as np


def cart2proj(x) -> np.array:
    """
    converts one or several points from cartesian 2D/3D to projective P^2/P^3
    """
    if np.ndim(x) == 1:
        return np.hstack((x, [1]))
    elif np.ndim(x) == 2 and x.shape[1] == 1:
        return np.vstack((x, [1]))
    else:
        return np.hstack((x, np.ones((x.shape[0], 1))))


def proj2cart(x) -> np.array:
    """
    converts one or several points from projective P^2/P^3 to cartesian 2D/3D
    """
    x = proj_normalization(x)
    if np.ndim(x) == 1 or (np.ndim(x) == 2 and x.shape[1] == 1):
        return x[:-1]
    else:
        return x[:, :-1]


def proj_normalization(x) -> np.array:
    """
    returns the normalized version of a vector or list of vectors in the projective space
    """
    if np.ndim(x) == 1 or (np.ndim(x) == 2 and x.shape[1] == 1):
        return x / x[-1]
    else:
        return (x.T / x[:, -1]).T


def forge_isometry(r: np.array, t: np.array) -> np.array:
    """
    forges an isometry from rotation and translation matrices
    """
    assert r.shape[0] == t.shape[0], "R and T must have the same number of rows"
    assert r.shape[0] == r.shape[1], "R must be square matrix"
    assert t.ndim == 1, "T must be a column vector"

    t = t[:, np.newaxis]
    a = np.vstack((
        np.hstack((r, t)),
        np.zeros((1, r.shape[1] + 1))
    ))
    a[-1, -1] = 1

    return a


def forge_projective_matrix(k: np.array, r: np.array = None, t: np.array = None, a: np.array = None) -> np.array:
    """
    forges a projective matrix given camera matrix K, and geometrical transformation (R,T) or A
    """
    assert k.shape == (3, 3), "k must be square matrix (3x3)"

    if a is not None:
        assert a.shape == (4, 4), "shape of A must be (4,4)"
        a = a / a[-1, -1]
        return k @ a[:-1]

    elif r is not None and t is not None:
        assert r.shape == (3, 3), "shape of R must be (3,3)"
        assert t.shape == (3,), "shape of T must be 3"
        return k @ np.hstack((r, t[..., np.newaxis]))

    else:
        return np.hstack((k, np.zeros((3, 1))))
