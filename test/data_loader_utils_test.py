import unittest
import numpy as np
import warnings

import sys
sys.path.append('.')
from data_loader.utils import *

def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class TestUtilsNorm(unittest.TestCase): 
      
    def setUp(self):
        self.oneD = np.arange(10,20)
        self.twoD = np.arange(20).reshape(2,10)
        pass

    def test_max_is_one_and_min_is_zero_oneD(self):
        n, n_max, n_min = norm(self.oneD, normalise='global_max')
        self.assertAlmostEqual(np.max(n), 1)
        n, n_max, n_min = norm(self.oneD, normalise='global_max_min')
        self.assertAlmostEqual(np.max(n), 1)
        self.assertAlmostEqual(np.min(n), 0)
        n, n_max, n_min = norm(self.oneD, normalise='local_max')
        self.assertAlmostEqual(np.max(n), 1)
        n, n_max, n_min = norm(self.oneD, normalise='local_max_min')
        self.assertAlmostEqual(np.max(n), 1)
        self.assertAlmostEqual(np.min(n), 0)

    def test_max_is_one_and_min_is_zero_twoD(self):
        n, n_max, n_min = norm(self.twoD, normalise='global_max')
        self.assertAlmostEqual(np.max(n), 1)
        n, n_max, n_min = norm(self.twoD, normalise='global_max_min')
        self.assertAlmostEqual(np.max(n), 1)
        self.assertAlmostEqual(np.min(n), 0)
        n, n_max, n_min = norm(self.twoD, normalise='local_max')
        self.assertTrue(np.allclose(np.max(n, axis=1), [1,1]))
        n, n_max, n_min = norm(self.twoD, normalise='local_max_min')
        self.assertTrue(np.allclose(np.max(n, axis=1), [1,1]))
        self.assertTrue(np.allclose(np.min(n, axis=1), [0,0]))

    @ignore_warnings
    def test_normalisation_calc_oneD(self):
        n, n_max, n_min = norm(self.oneD, normalise='global_max')
        self.assertTrue(np.allclose((n * (n_max - n_min) + n_min).squeeze(), self.oneD))
        n, n_max, n_min = norm(self.oneD, normalise='global_max_min')
        self.assertTrue(np.allclose((n * (n_max - n_min) + n_min).squeeze(), self.oneD))
        n, n_max, n_min = norm(self.oneD, normalise='local_max')
        self.assertTrue(np.allclose((n * (n_max - n_min) + n_min).squeeze(), self.oneD))
        n, n_max, n_min = norm(self.oneD, normalise='local_max_min')
        self.assertTrue(np.allclose((n * (n_max - n_min) + n_min).squeeze(), self.oneD))

    def test_normalisation_calc_twoD(self):
        n, n_max, n_min = norm(self.twoD, normalise='global_max')
        self.assertTrue(np.allclose((n * (n_max - n_min) + n_min).squeeze(), self.twoD))
        n, n_max, n_min = norm(self.twoD, normalise='global_max_min')
        self.assertTrue(np.allclose((n * (n_max - n_min) + n_min).squeeze(), self.twoD))
        n, n_max, n_min = norm(self.twoD, normalise='local_max')
        self.assertTrue(np.allclose((n * (n_max - n_min) + n_min).squeeze(), self.twoD))
        n, n_max, n_min = norm(self.twoD, normalise='local_max_min')
        self.assertTrue(np.allclose((n * (n_max - n_min) + n_min).squeeze(), self.twoD))

class TestUtilsConv(unittest.TestCase): 
      
    def setUp(self):
        self.oneD = np.arange(10)
        self.twoD = np.arange(20).reshape(2,10)
        pass
  
    @ignore_warnings
    def test_oneD_length(self): 
        out = conv(self.oneD, 5)
        self.assertTrue(out.shape == (6,5,1))
        self.assertTrue(np.allclose(out[0, :,0], np.arange(5)))
        self.assertTrue(np.allclose(out[-1, :,0], np.arange(5,10)))

    @ignore_warnings
    def test_twoD_length(self): 
        out = conv(self.twoD, 5)
        self.assertTrue(out.shape == (12,5,1))
        self.assertTrue(np.allclose(out[0, :,0], np.arange(5)))
        self.assertTrue(np.allclose(out[5, :,0], np.arange(5,10)))
        self.assertTrue(np.allclose(out[-1, :,0], np.arange(15,20)))

    @ignore_warnings
    def test_oneD_length_and_start(self): 
        out = conv(self.oneD, 5, start=2)
        self.assertTrue(out.shape == (4,5,1))
        self.assertTrue(np.allclose(out[0, :,0], np.arange(2,7)))
        self.assertTrue(np.allclose(out[-1, :,0], np.arange(5,10)))

    @ignore_warnings
    def test_twoD_length_and_start(self): 
        out = conv(self.twoD, 5, start=2)
        self.assertTrue(out.shape == (8,5,1))
        self.assertTrue(np.allclose(out[0, :,0], np.arange(2,7)))
        self.assertTrue(np.allclose(out[5, :,0], np.arange(13,18)))
        self.assertTrue(np.allclose(out[-1, :,0], np.arange(15,20)))

    @ignore_warnings
    def test_oneD_length_and_step(self): 
        out = conv(self.oneD, 5, stride=3)
        self.assertTrue(out.shape == (2,5,1))
        self.assertTrue(np.allclose(out[0, :,0], np.arange(5)))
        self.assertTrue(np.allclose(out[-1, :,0], np.arange(3,8)))

    @ignore_warnings
    def test_twoD_length_and_step(self): 
        out = conv(self.twoD, 5, stride=2)
        self.assertTrue(out.shape == (6,5,1))
        self.assertTrue(np.allclose(out[0, :,0], np.arange(5)))
        self.assertTrue(np.allclose(out[2, :,0], np.arange(4,9)))
        self.assertTrue(np.allclose(out[-1, :,0], np.arange(14,19)))

    @ignore_warnings
    def test_oneD_length_and_reshape(self): 
        out = conv(self.oneD, 6, dim=2)
        self.assertTrue(out.shape == (5,3,2))
        self.assertTrue(np.allclose(out[0], np.arange(6).reshape(3,2)))
        self.assertTrue(np.allclose(out[-1], np.arange(4,10).reshape(3,2)))

    @ignore_warnings
    def test_twoD_length_and_reshape(self): 
        out = conv(self.twoD, 6, dim=2)
        self.assertTrue(out.shape == (10,3,2))
        self.assertTrue(np.allclose(out[0], np.arange(6).reshape(3,2)))
        self.assertTrue(np.allclose(out[2], np.arange(2,8).reshape(3,2)))
        self.assertTrue(np.allclose(out[-1], np.arange(14,20).reshape(3,2)))

if __name__ == '__main__': 
    unittest.main()