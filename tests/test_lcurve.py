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

    
    def test_rebin_lc(self):
        t  = np.arange(8, dtype=np.double)
        x  = np.arange(8, dtype=np.double)
        xe = np.arange(1, 9)*1.
        lc = az.LCurve(t, x, xe)
        
        lc1 = lc.rebin(2, error='poiss')
        np.testing.assert_array_almost_equal(
                lc1.time, np.arange(0.5, 8, 2))
        np.testing.assert_array_almost_equal(
                lc1.rate, np.arange(0.5, 8, 2))
        np.testing.assert_array_almost_equal(
            lc1.rerr, np.sqrt(np.arange(0.5, 8, 2)*2)/2.)


    def test_rebin_lc__w_gaps(self):
        # t: 0  1  2  3  4  5  6  7
        # x: 0  1  2  3  4  5  6  7
        # xe:1  2  3  4  5  6  7  8
        #  : *  -  *  -  -  -  -  * 
        t  = np.arange(8, dtype=np.double)
        x  = np.arange(8, dtype=np.double)
        xe = np.arange(1, 9)*1.
        ind = np.array([0, 2, 7])
        lc = az.LCurve(t[ind], x[ind], xe[ind], 1.0)
        
        lc1 = lc.rebin(2, min_exp=0.5, error='poiss')
        np.testing.assert_array_almost_equal(
                lc1.time, np.array([0.5, 2.5, 6.5]))
        np.testing.assert_array_almost_equal(
                lc1.rate, np.array([0., 2., 7]))
        np.testing.assert_array_almost_equal(
                lc1.rerr, (np.array([0., 2., 7])*2)**0.5/2)

        # when original lc is even #
        lc = lc.make_even()
        lc1 = lc.rebin(2, min_exp=0.0, error='poiss')
        np.testing.assert_array_almost_equal(
                lc1.time, np.array([0.5, 2.5, 4.5, 6.5]))
        np.testing.assert_array_almost_equal(
                lc1.rate, np.array([0., 2., np.nan, 7]))


    def test_read_fits_files(self):
        import astropy.io.fits as pyfits
        t  = np.arange(4)
        x  = np.arange(4)
        xe = np.arange(1, 5)*1.
        hdu = pyfits.BinTableHDU.from_columns([
                pyfits.Column(name='TIME', format='E', array=t),
                pyfits.Column(name='RATE', format='E', array=x),
                pyfits.Column(name='ERROR', format='E', array=xe),
            ])
        hdu.name = 'RATE'
        fname = 'tmp.fits'
        hdu.writeto(fname, clobber=True)
        lc, _ = az.LCurve.read_fits_file(fname)
        np.testing.assert_array_almost_equal(t,  lc[0])
        np.testing.assert_array_almost_equal(x,  lc[1])
        np.testing.assert_array_almost_equal(xe, lc[2])
        if os.path.exists(fname): os.remove(fname)
