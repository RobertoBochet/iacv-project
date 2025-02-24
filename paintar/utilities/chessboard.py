import numpy as np


class Chessboard:
    """
    chessboard model used for the cv.findChessboardCorners function
    """

    def __init__(self, size: tuple[int, int] = (7, 7), square_size: float = 1):
        self._size = size
        self._square_size = square_size

    @property
    def size(self) -> tuple[int, int]:
        """
        it is the size of the chessboard (columns, rows)
        """
        return self._size

    def get_points(self) -> np.array:
        """
        returns a vector of corners points in cartesian 3d world;
        first point (0,0,0) is the top-left corner,
        the next ones are given in order columns(x), rows(y) (1,0,0),(2,0,0),...
        """

        return np.array(
            [(i * self._square_size, j * self._square_size, 0) for i in range(self._size[1]) for j in
             range(self._size[0])], dtype=np.float32)
