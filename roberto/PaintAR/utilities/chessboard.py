import numpy as np


class Chessboard:
    def __init__(self, size: tuple[int, int] = (7, 7), square_size: int = 1):
        self._size = size
        self._square_size = square_size

    @property
    def size(self):
        return self._size

    def get_points(self) -> np.array:
        return np.array(
            [(i, j, 0) for i in range(self._size[0]) for j in range(self._size[1])],
            dtype=np.float32) * self._square_size
