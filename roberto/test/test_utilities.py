import unittest
import numpy as np
from paintar import utilities as ut


class Test_cart2proj(unittest.TestCase):
    def test_3(self):
        np.testing.assert_allclose(
            ut.cart2proj(np.array([1., 2., 5.])),
            np.array([1., 2., 5., 1.]))

    def test_3x1(self):
        np.testing.assert_allclose(
            ut.cart2proj(np.array([[1.], [2.], [5.]])),
            np.array([[1.], [2.], [5.], [1.]]))

    def test_Nx3(self):
        np.testing.assert_allclose(
            ut.cart2proj(np.array([[-29., 6., 4.],
                                   [60., -24., 6.]])),
            np.array([[-29., 6., 4., 1.],
                      [60., -24., 6., 1.]]))


class Test_proj2cart(unittest.TestCase):

    def test_3(self):
        np.testing.assert_allclose(
            ut.proj2cart(np.array([1., 2., 5.])),
            np.array([.2, .4]))

    def test_4(self):
        np.testing.assert_allclose(
            ut.proj2cart(np.array([1., 2., 5., 2.])),
            np.array([.5, 1., 2.5]))

    def test_4x1(self):
        np.testing.assert_allclose(
            ut.proj2cart(np.array([[1.], [2.], [5.], [2.]])),
            np.array([[.5], [1.], [2.5]]))

    def test_Nx4(self):
        np.testing.assert_allclose(
            ut.proj2cart(np.array([[-29., 6., 1., 4.],
                                   [60., 48., -24., 6.]])),
            np.array([[-7.25, 1.5, 0.25],
                      [10., 8., -4.]]))


class Test_proj_normalization(unittest.TestCase):
    def test_4(self):
        np.testing.assert_allclose(
            ut.proj_normalization(np.array([1., 2., 5., 2.])),
            np.array([.5, 1., 2.5, 1.]))

    def test_4x1(self):
        np.testing.assert_allclose(
            ut.proj_normalization(np.array([1., 2., 5., 2.]).reshape(4, 1)),
            np.array([.5, 1., 2.5, 1.]).reshape(4, 1))

    def test_Nx4(self):
        np.testing.assert_allclose(
            ut.proj_normalization(np.array([[-29., 6., 1., 4.],
                                            [60., 48., -24., 6.]])),
            np.array([[-7.25, 1.5, 0.25, 1.],
                      [10., 8., -4., 1.]]))


if __name__ == '__main__':
    unittest.main()
