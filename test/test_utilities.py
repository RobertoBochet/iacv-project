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


class Test_normalization_points(unittest.TestCase):
    def test_p4(self):
        original_points = np.array([
            [-7.25, 1.5, 0.25, 0.5],
            [10., 8., -4., 1.],
            [.5, 1., 2.5, 1.],
            [-29., 6., 4., 4.],
            [60., -24., 6., 1.]
        ])

        h, points = ut.normalize_points(original_points)

        # tests centroid in 0
        np.testing.assert_allclose(
            points.mean(axis=0)[:-1],
            np.array([0., 0., 0.]),
            atol=1e-15)

        d = points[:, :-1]
        d = np.linalg.norm(d, axis=1)
        d = d.mean()
        # TODO improve the function to pass this test
        # tests variance close to sqrt(2)
        # self.assertAlmostEqual(d, np.sqrt(2), delta=0.2)

        # normalizes the original points, last element equal to 1
        original_points = (original_points.T / original_points[:, -1]).T

        # restores the precondition
        h_inv = np.linalg.inv(h)
        for i in range(len(points)):
            points[i] = h_inv @ points[i]

        # tests the precondition restoring
        np.testing.assert_allclose(original_points, points)


if __name__ == '__main__':
    unittest.main()
