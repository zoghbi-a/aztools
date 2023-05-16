

import unittest

import numpy as np

from aztools import LCurve


class LCurveTest(unittest.TestCase):
    """testing LCurve."""


    def test_init(self):
        """Test initializaiton"""
        tarr = np.arange(6)
        lcrv = LCurve(tarr, tarr*0.5, rerr=tarr*0.01)
        self.assertEqual(lcrv.npoints, len(tarr))
        self.assertEqual(lcrv.deltat, 1.)
        self.assertTrue(lcrv.iseven)
        lcrv = LCurve(tarr[[0,2,3,4,5]], tarr[1:], rerr=tarr[1:]*0.01)
        self.assertFalse(lcrv.iseven)


    def test_make_even(self):
        tarr = np.arange(8)*1.
        xarr = np.arange(8)*1.
        xerr = np.arange(1, 9)*1.
        ind = np.arange(0,8,2)
        ind[-1] = len(tarr)-1
        lcrv = LCurve(tarr[ind], xarr[ind], rerr=xerr[ind], deltat=1.0)
        self.assertFalse(lcrv.iseven)
        lcrv1 = lcrv.make_even()
        self.assertTrue(lcrv1.iseven)
        np.testing.assert_array_equal(tarr, lcrv1.time)
        np.testing.assert_array_almost_equal(
            xarr[ind], lcrv1.rate[~np.isnan(lcrv1.rate)])

        # time cannot be cast to even, fail #
        lcrv.time[1] += 0.5
        with self.assertRaises(ValueError):
            lcrv.make_even()


    def test_rebin_lc(self):
        tarr = np.arange(8, dtype=np.double)
        xarr = np.arange(8, dtype=np.double)
        xerr = np.arange(1, 9)*1.
        lcrv = LCurve(tarr, xarr, rerr=xerr)

        lcrv1 = lcrv.rebin(2, error='poiss')
        np.testing.assert_array_almost_equal(
                lcrv1.time, np.arange(0.5, 8, 2))
        np.testing.assert_array_almost_equal(
                lcrv1.rate, np.arange(0.5, 8, 2))
        np.testing.assert_array_almost_equal(
            lcrv1.rerr, np.sqrt(np.arange(0.5, 8, 2)*2)/2.)


    def test_rebin_lc__w_gaps(self):
        # t: 0  1  2  3  4  5  6  7
        # x: 0  1  2  3  4  5  6  7
        # xe:1  2  3  4  5  6  7  8
        #  : *  -  *  -  -  -  -  *
        tarr = np.arange(8, dtype=np.double)
        xarr = np.arange(8, dtype=np.double)
        xerr = np.arange(1, 9)*1.
        ind = np.array([0, 2, 7])
        lcrv = LCurve(tarr[ind], xarr[ind], rerr=xerr[ind], deltat=1.0)

        lcrv1 = lcrv.rebin(2, min_exp=0.5, error='poiss')
        np.testing.assert_array_almost_equal(
                lcrv1.time, np.array([0.5, 2.5, 6.5]))
        np.testing.assert_array_almost_equal(
                lcrv1.rate, np.array([0., 2., 7]))
        np.testing.assert_array_almost_equal(
                lcrv1.rerr, (np.array([((2.**0.5)+ (7**0.5))/2,
                                     (2.**0.5), (7**0.5)])))

        # when original lc is even #
        lcrv = lcrv.make_even()
        lcrv1 = lcrv.rebin(2, min_exp=0.0, error='poiss')
        np.testing.assert_array_almost_equal(
                lcrv1.time, np.array([0.5, 2.5, 4.5, 6.5]))
        np.testing.assert_array_almost_equal(
                lcrv1.rate, np.array([0., 2., np.nan, 7]))


    def test_interp_small_gaps_1(self):
        # t: 0  1  2  3  4  5  6  7
        # x: 0  1  2  3  4  5  6  7
        # xe:1  2  3  4  5  6  7  8
        #  : *  -  *  -  -  -  -  *
        tarr = np.arange(8, dtype=np.double)
        xarr = np.arange(8, dtype=np.double)
        xerr = np.arange(1, 9)*1.
        ind = np.array([0, 2, 7])
        lcrv = LCurve(tarr[ind], xarr[ind], rerr=xerr[ind], deltat=1.0)

        # not even #
        with self.assertRaises(ValueError):
            lcrv.interp_small_gaps(maxgap=1, noise=None)

        # no noise, 1 gap #
        lcrv = lcrv.make_even()
        lcrv.interp_small_gaps(maxgap=1, noise=None)
        np.testing.assert_array_almost_equal(lcrv.rate,  [0, 1, 2]+[np.nan]*4+[7])
        np.testing.assert_array_almost_equal(lcrv.rerr,  [1, 4, 3]+[np.nan]*4+[8])



    def test_interp_small_gaps_2(self):
        # t: 0  1  2  3  4  5  6  7
        # x: 0  1  2  3  4  5  6  7
        # xe:1  2  3  4  5  6  7  8
        #  : *  -  *  -  -  -  -  *
        tarr = np.arange(8, dtype=np.double)
        xarr = np.arange(8, dtype=np.double)
        xerr = np.arange(1, 9)*1.
        ind = np.array([0, 2, 7])
        lcrv = LCurve(tarr[ind], xarr[ind], rerr=xerr[ind], deltat=1.0)
        lcrv = lcrv.make_even()

        # no noise, all gaps #
        lcrv.interp_small_gaps(maxgap=20, noise=None)
        np.testing.assert_array_almost_equal(lcrv.rate,  [0, 1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_almost_equal(lcrv.rerr,  [1, 4, 3, 4, 4, 4, 4, 8])



    def test_interp_small_gaps_3(self):
        # t: 0  1  2  3  4  5  6  7
        # x: 0  1  2  3  4  5  6  7
        # xe:1  2  3  4  5  6  7  8
        #  : *  -  *  -  -  -  -  *
        tarr = np.arange(8, dtype=np.double)
        xarr = np.arange(8, dtype=np.double)
        xerr = np.arange(1, 9)*1.
        ind = np.array([0, 2, 7])
        lcrv = LCurve(tarr[ind], xarr[ind], rerr=xerr[ind], deltat=1.0)
        lcrv = lcrv.make_even()


        # poisson #
        lcrv.interp_small_gaps(maxgap=1, noise='poiss', seed=123)
        np.random.seed(123)
        prnd = np.random.poisson(1)
        np.testing.assert_array_almost_equal(lcrv.rate,  [0, prnd, 2]+[np.nan]*4+[7])
        np.testing.assert_array_almost_equal(lcrv.rerr,  [1, prnd**0.5, 3]+[np.nan]*4+[8])



    def test_interp_small_gaps_4(self):
        # t: 0  1  2  3  4  5  6  7
        # x: 0  1  2  3  4  5  6  7
        # xe:1  2  3  4  5  6  7  8
        #  : *  -  *  -  -  -  *  -
        # gap at the end of lc #
        tarr = np.arange(8, dtype=np.double)
        xarr = np.arange(8, dtype=np.double)
        xerr = np.arange(1, 9)*1.
        ind = np.array([0, 2, 6, 7])
        lcrv = LCurve(tarr[ind], xarr[ind], rerr=xerr[ind], deltat=1.0)
        lcrv.rate[-1] = np.nan
        lcrv.rerr[-1] = np.nan
        lcrv = lcrv.make_even()


        lcrv.interp_small_gaps(maxgap=1, noise=None)
        np.testing.assert_array_almost_equal(lcrv.rate,  [0, 1, 2]+[np.nan]*3+[6,6])
        np.testing.assert_array_almost_equal(lcrv.rerr,  [1, 11/3, 3]+[np.nan]*3+[7,11/3])



    def test_interp_small_gaps_5(self):
        # t: 0  1  2  3  4  5  6  7
        # x: 0  1  2  3  4  5  6  7
        # xe:1  2  3  4  5  6  7  8
        #  : *  -  *  -  -  -  -  *
        tarr = np.arange(8, dtype=np.double)
        xarr = np.arange(8, dtype=np.double)
        xerr = np.arange(1, 9)*1.
        ind = np.array([0, 2, 7])
        lcrv = LCurve(tarr[ind], xarr[ind], rerr=xerr[ind], deltat=1.0)
        lcrv = lcrv.make_even()


        # norm #
        lcrv.interp_small_gaps(maxgap=1, noise='norm', seed=123)
        np.random.seed(123)
        prnd = np.random.randn(8)[1]*4 + 1
        np.testing.assert_array_almost_equal(lcrv.rate,  [0, prnd, 2]+[np.nan]*4+[7])
        np.testing.assert_array_almost_equal(lcrv.rerr,  [1, 4, 3]+[np.nan]*4+[8])


    def test_calculate_psd(self):
        """test LCurve.calculate_psd"""
        rat = np.random.randn(6) + 10
        psd = LCurve.calculate_psd(rat, 1.0, 'rms')

        np.testing.assert_array_almost_equal(
            psd[0], np.fft.rfftfreq(6, 1.0)[1:-1])
        np.testing.assert_array_almost_equal(
            psd[1], (2./(6.*rat.mean()**2))*np.abs(np.fft.rfft(rat)[1:-1])**2)

        rat = [rat[:3], rat[3:]]
        psd = LCurve.calculate_psd(rat, 1.0, 'rms')
        np.testing.assert_array_almost_equal(
            psd[0], np.sort(np.concatenate([np.fft.rfftfreq(3, 1.0)[1:-1]*2])))
