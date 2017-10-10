import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import aztools as az


class LCurveTest(unittest.TestCase):
    """testing LCurve."""


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



    def test_calculate_psd(self):
        
        r = np.random.randn(6) + 10
        p = az.LCurve.calculate_psd(r, 1.0, 'rms')

        np.testing.assert_array_almost_equal(p[0], 
                np.fft.rfftfreq(6, 1.0)[1:-1])
        np.testing.assert_array_almost_equal(p[1], 
                (2./(6.*r.mean()**2))*np.abs(np.fft.rfft(r)[1:-1])**2)


        r = [r[:3], r[3:]]
        p = az.LCurve.calculate_psd(r, 1.0, 'rms')
        np.testing.assert_array_almost_equal(p[0], 
            np.sort(np.concatenate([np.fft.rfftfreq(3, 1.0)[1:-1]*2])))


    def test_bin_psd__sim(self):
        """Do simple lc simulations and calculated psd
        
        Mostly to test the logavg option. The testing is done
            to compare the errors.
            1- Whenever logavg=True is used, bias correciton needs
            to be applied.

            2- This simulations are very sensitive to red noise leak.
            This is clear from the dependence of the required 'bias'
            on the driving psd slope when using a singel powerlaw model.
            If we control for rednoise leak by using a broken powerlaw
            as a driving psd, things are better.

            3- Leak aside, logavg=True (with bias correction) does better 
            particulalry when using a single long segment, and averaging 
            multiple neighboring frequencies.


        Conclusion:
            Always use logavg=True; if the psd slope is high, use
            a tapering window to reduce read noise leak, Not sure if 
            that works though.

        To plot, comments the return line
        P1 is for simulations with logavg=True, the P2 for logavg=False

        """
        return
        np.random.seed(1234)
        sim = az.SimLC()
        norm = 'var'
        expo = {'var':0, 'leahy':1, 'rms':2}
        #sim.add_model('powerlaw', [1e-1, -2])
        sim.add_model('broken_powerlaw', [1e-1, 0, -2, 1e-2])
        n, mu = 512, 100.


        P1, P2 = [], []
        for i in range(200):
            sim.simulate(4*n, 1.0, mu, norm=norm)
            s,i  = az.misc.split_array(sim.x[:n], 0)
            p0 = az.LCurve.calculate_psd(s, 1.0, norm)
            p1 = az.LCurve.bin_psd(p0[0], p0[1], {'by_n':[20,1]} ,logavg=True)
            p2 = az.LCurve.bin_psd(p0[0], p0[1], {'by_n':[20,1]} ,logavg=False)
            P1.append(p1[:3])
            P2.append(p2[:3])
        P1, P2 = np.array(P1), np.array(P2)


        fm, pm = sim.normalized_psd[0][1:-1], sim.normalized_psd[1][1:-1]
        import pylab as plt
        PP, pp = [P1, P2], [p1, p2]
        for i in range(2):
            plt.subplot(1,2,i+1); plt.loglog(fm, pm)
            P, p = PP[i], pp[i]
            plt.plot(p[0], P[:,1,:].mean(0), color='C2')
            plt.plot(p[0], np.median(P[:,1,:], 0), color='C3')
            plt.title('Log' if i==0 else 'nolog')
            plt.plot(p[0], P[:,1,:].mean(0)+P[:,1,:].std(0), '--', color='C2')
            plt.plot(p[0], P[:,1,:].mean(0)-P[:,1,:].std(0), '--', color='C2')
            plt.plot(p[0], P[:,1,:].mean(0)+P[:,2,:].mean(0), '-.', color='C3')
            plt.plot(p[0], P[:,1,:].mean(0)-P[:,2,:].mean(0), '-.', color='C3')
        plt.show()


