try:
    import cv2.cv2 as cv
except ModuleNotFoundError:
    import cv2 as cv
    
import numpy as np

from .functions import proj2cart


def draw_axis(img: np.ndarray, m: np.ndarray, s: float = 0.05, draw_z: bool = True) -> np.ndarray:
    o = np.array([[0, 0, 0, 1]]).T
    x = np.array([[s, 0, 0, 1]]).T
    y = np.array([[0, s, 0, 1]]).T
    z = np.array([[0, 0, s, 1]]).T

    p_o = tuple(np.squeeze(proj2cart(m @ o)).astype(int))
    p_x = tuple(np.squeeze(proj2cart(m @ x)).astype(int))
    p_y = tuple(np.squeeze(proj2cart(m @ y)).astype(int))
    p_z = tuple(np.squeeze(proj2cart(m @ z)).astype(int))

    img = cv.line(img, p_o, p_x, (0, 0, 255), 5)
    img = cv.line(img, p_o, p_y, (0, 255, 0), 5)

    if draw_z:
        img = cv.line(img, p_o, p_z, (255, 0, 0), 5)

    return img
