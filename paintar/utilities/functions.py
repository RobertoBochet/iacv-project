from typing import Union

import numpy as np


def cart2proj(x: np.ndarray) -> np.ndarray:
    """
    converts one or several points from cartesian 2D/3D to projective P^2/P^3
    """
    if np.ndim(x) == 1:
        return np.hstack((x, [1]))
    elif np.ndim(x) == 2 and x.shape[1] == 1:
        return np.vstack((x, [1]))
    else:
        return np.hstack((x, np.ones((x.shape[0], 1))))


def proj2cart(x: np.ndarray) -> np.ndarray:
    """
    converts one or several points from projective P^2/P^3 to cartesian 2D/3D
    """
    x = proj_normalization(x)
    if np.ndim(x) == 1 or (np.ndim(x) == 2 and x.shape[1] == 1):
        return x[:-1]
    else:
        return x[:, :-1]


def proj_normalization(x: np.ndarray) -> np.ndarray:
    """
    returns the normalized version of a vector or list of vectors in the projective space
    """
    if np.ndim(x) == 1 or (np.ndim(x) == 2 and x.shape[1] == 1):
        return x / x[-1]
    else:
        return (x.T / x[:, -1]).T


def forge_isometry(r: np.ndarray, t: np.ndarray) -> np.ndarray:
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


def forge_projective_matrix(k: np.ndarray, r: np.ndarray = None, t: np.ndarray = None,
                            a: np.ndarray = None) -> np.ndarray:
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


def get_crop_around_limits(point: np.ndarray, size: Union[np.ndarray, int], bounds: np.ndarray = None) -> np.ndarray:
    """
    given a point in 2D space, and a size of a rectangle or a circle radius
    returns the limits for a rectangle crop around the point
    if bounds are provided, them limit the crop
    """
    assert point.size == 2, "point must be a 2D point"

    if np.ndim(point) == 2:
        point = point.reshape(2)

    half_size = size * 0.5 if isinstance(size, np.ndarray) else 0.5 * np.array([size, size])

    # the point coordinates are x,y the image coordinate are y,x
    point = point[::-1]
    half_size = half_size[::-1]

    limits = np.vstack((point - half_size, point + half_size)).T

    if bounds is not None:
        if bounds.size == 2:
            bounds = np.vstack((np.zeros(2), bounds)).T

        if limits[0, 0] < bounds[0, 0]:
            limits[0, 0] = bounds[0, 0]
        if limits[1, 0] < bounds[1, 0]:
            limits[1, 0] = bounds[1, 0]
        if limits[0, 1] > bounds[0, 1]:
            limits[0, 1] = bounds[0, 1]
        if limits[1, 1] > bounds[1, 1]:
            limits[1, 1] = bounds[1, 1]

    return limits.astype(np.int)


def crop_around(img: np.ndarray, point: np.ndarray, size: Union[np.ndarray, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    given a point in 2D space, a size of a rectangle or a circle radius and the image
    returns the crop around the point, and the coordinate of the left-up corner
    """
    crop_limit = get_crop_around_limits(point, size, bounds=np.array(img.shape[0:2]))

    y_min = crop_limit[0, 0]
    y_max = crop_limit[0, 1]
    x_min = crop_limit[1, 0]
    x_max = crop_limit[1, 1]

    return np.array([x_min, y_min]), img[y_min:y_max, x_min:x_max]
