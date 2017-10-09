import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import aztools as az


class LCurveTest(unittest.TestCase):
    """testing SimLC."""


    def test_LCurve_init(self):
        """Test initializaiton"""
        t = np.arange(6)
        lc = az.LCurve(t, t*0.5, t*0.01)
        assert(lc.nt == len(t))
        assert(lc.dt == 1.)
        assert(lc.iseven)
        lc = az.LCurve(t[[0,2,3,4,5]], t[1:], t[1:]*0.01)
        assert(not lc.iseven)


    def test_make_even(self):
        t  = np.arange(8)*1.
        x  = np.arange(8)*1.
        xe = np.arange(1, 9)*1.
        ind = np.arange(0,8,2); ind[-1] = len(t)-1
        lc = az.LCurve(t[ind], x[ind], xe[ind], 1.0)
        assert(not lc.iseven)
        lc1 = lc.make_even()
        assert(lc1.iseven)
        np.testing.assert_array_equal(t, lc1.time)
        np.testing.assert_array_almost_equal(
            x[ind], lc1.rate[~np.isnan(lc1.rate)])

        # time cannot be cast to even, fail #
        lc.time[1] += 0.5
        with self.assertRaises(ValueError):
            lc.make_even()


