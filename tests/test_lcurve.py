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
                lc1.rerr, (np.array([((2.**0.5)+ (7**0.5))/2, (2.**0.5), (7**0.5)])))

        # when original lc is even #
        lc = lc.make_even()
        lc1 = lc.rebin(2, min_exp=0.0, error='poiss')
        np.testing.assert_array_almost_equal(
                lc1.time, np.array([0.5, 2.5, 4.5, 6.5]))
        np.testing.assert_array_almost_equal(
                lc1.rate, np.array([0., 2., np.nan, 7]))


    def test_interp_small_gaps_1(self):
        # t: 0  1  2  3  4  5  6  7
        # x: 0  1  2  3  4  5  6  7
        # xe:1  2  3  4  5  6  7  8
        #  : *  -  *  -  -  -  -  * 
        t  = np.arange(8, dtype=np.double)
        x  = np.arange(8, dtype=np.double)
        xe = np.arange(1, 9)*1.
        ind = np.array([0, 2, 7])
        lc = az.LCurve(t[ind], x[ind], xe[ind], 1.0)
        
        # not even #
        with self.assertRaises(ValueError):
            lc.interp_small_gaps(maxgap=1, noise=None)

        # no noise, 1 gap #
        lc = lc.make_even()
        lc.interp_small_gaps(maxgap=1, noise=None)
        np.testing.assert_array_almost_equal(lc.rate,  [0, 1, 2]+[np.nan]*4+[7])
        np.testing.assert_array_almost_equal(lc.rerr,  [1, 4, 3]+[np.nan]*4+[8])

    

    def test_interp_small_gaps_2(self):
        # t: 0  1  2  3  4  5  6  7
        # x: 0  1  2  3  4  5  6  7
        # xe:1  2  3  4  5  6  7  8
        #  : *  -  *  -  -  -  -  * 
        t  = np.arange(8, dtype=np.double)
        x  = np.arange(8, dtype=np.double)
        xe = np.arange(1, 9)*1.
        ind = np.array([0, 2, 7])
        lc = az.LCurve(t[ind], x[ind], xe[ind], 1.0)
        lc = lc.make_even()

        # no noise, all gaps #
        lc.interp_small_gaps(maxgap=20, noise=None)
        np.testing.assert_array_almost_equal(lc.rate,  [0, 1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_almost_equal(lc.rerr,  [1, 4, 3, 4, 4, 4, 4, 8])



    def test_interp_small_gaps_3(self):
        # t: 0  1  2  3  4  5  6  7
        # x: 0  1  2  3  4  5  6  7
        # xe:1  2  3  4  5  6  7  8
        #  : *  -  *  -  -  -  -  * 
        t  = np.arange(8, dtype=np.double)
        x  = np.arange(8, dtype=np.double)
        xe = np.arange(1, 9)*1.
        ind = np.array([0, 2, 7])
        lc = az.LCurve(t[ind], x[ind], xe[ind], 1.0)
        lc = lc.make_even()


        # poisson #
        lc.interp_small_gaps(maxgap=1, noise='poiss', seed=123)
        np.random.seed(123)
        pp = np.random.poisson(1)
        np.testing.assert_array_almost_equal(lc.rate,  [0, pp, 2]+[np.nan]*4+[7])
        np.testing.assert_array_almost_equal(lc.rerr,  [1, pp**0.5, 3]+[np.nan]*4+[8])



    def test_interp_small_gaps_4(self):
        # t: 0  1  2  3  4  5  6  7
        # x: 0  1  2  3  4  5  6  7
        # xe:1  2  3  4  5  6  7  8
        #  : *  -  *  -  -  -  *  - 
        # gap at the end of lc #
        t  = np.arange(8, dtype=np.double)
        x  = np.arange(8, dtype=np.double)
        xe = np.arange(1, 9)*1.
        ind = np.array([0, 2, 6, 7])
        lc = az.LCurve(t[ind], x[ind], xe[ind], 1.0)
        lc.rate[-1] = np.nan
        lc.rerr[-1] = np.nan
        lc = lc.make_even()


        lc.interp_small_gaps(maxgap=1, noise=None)
        np.testing.assert_array_almost_equal(lc.rate,  [0, 1, 2]+[np.nan]*3+[6,6])
        np.testing.assert_array_almost_equal(lc.rerr,  [1, 11/3, 3]+[np.nan]*3+[7,11/3])

    

    def test_interp_small_gaps_5(self):
        # t: 0  1  2  3  4  5  6  7
        # x: 0  1  2  3  4  5  6  7
        # xe:1  2  3  4  5  6  7  8
        #  : *  -  *  -  -  -  -  * 
        t  = np.arange(8, dtype=np.double)
        x  = np.arange(8, dtype=np.double)
        xe = np.arange(1, 9)*1.
        ind = np.array([0, 2, 7])
        lc = az.LCurve(t[ind], x[ind], xe[ind], 1.0)
        lc = lc.make_even()


        # norm #
        lc.interp_small_gaps(maxgap=1, noise='norm', seed=123)
        np.random.seed(123)
        pp = np.random.randn(8)[1]*4 + 1
        np.testing.assert_array_almost_equal(lc.rate,  [0, pp, 2]+[np.nan]*4+[7])
        np.testing.assert_array_almost_equal(lc.rerr,  [1, 4, 3]+[np.nan]*4+[8])



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

            4- Tapering might work. The psd need to be renormalized to
            componsate for the reduced variability power (hanning for example
            reduces power by ~0.62; from ratio of rms for random while noise)


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
        sim.add_model('powerlaw', [1e-1, -2])
        #sim.add_model('broken_powerlaw', [1e-1, 0, -2, 1e-2])
        n, mu = 512, 100.


        P1, P2 = [], []
        for i in range(200):
            sim.simulate(4*n, 1.0, mu, norm=norm)
            s,i  = az.misc.split_array(sim.x[:n], 0)
            p0 = az.LCurve.calculate_psd(s, 1.0, norm)
            p1 = az.LCurve.bin_psd(p0[0], p0[1], {'by_n':[10,1]} ,logavg=True)
            p2 = az.LCurve.bin_psd(p0[0], p0[1], {'by_n':[10,1]} ,logavg=False)
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



    def test_calculate_lag(self):
        sim = az.SimLC()
        norm = 'rms'
        expo = 2
        sim.add_model('powerlaw', [1e-4, -2])
        n, mu = 128, 100.

        sim.simulate(n, 1.0, mu, norm=norm)
        
        sim.add_model('constant', 6, lag=True)
        sim.apply_lag(phase=False)

        p = az.LCurve.calculate_lag(sim.y, sim.x, 1.0, fqbin=None, taper=False)
        np.testing.assert_array_almost_equal(sim.normalized_lag[0][1:-1], p[0])
        # do the first 5 before the oscillation kicks in.  
        np.testing.assert_array_almost_equal(np.zeros(5)+6, p[1][:5])



    def test_calculate_lag__sim(self):
        """Do simple lc simulations and calculated lag

        Note:
            1- No poisson noise is included in the following.

            2- Red noise leak can affect lag measurements too. 
            The main effect is that when using steep powerlaw PSD, 
            the lag is underestimated (closer to zero). The effect 
            goes away when using a broken powerlaw PSD. This is the
            wether binning is used or not.

            3- Tapering the light curve with some function (e.g. hanning)
            reduces the bias significantly except at the lowest frequencies.
            where it remains. This is true wether binning is used or not.

            4- The bias in the lowest frequency is reduced by averaging the
            log of the cross spectrum rather than the cross spectrum itself.

        Conclusion: Taper the light curves


        comment the first line to run


        """
        return
        np.random.seed(4567)
        sim = az.SimLC()
        norm = 'var'
        expo = {'var':0, 'leahy':1, 'rms':2}
        sim.add_model('powerlaw', [1e-1, -2])
        #sim.add_model('broken_powerlaw', [1e-1, 0, -2, 1e-2])
        sim.add_model('constant', 2, lag=True)
        #sim.add_model('lorentz', [2, 0, 2e-1], lag=True)
        n, mu = 512, 100.

        def taper(x):
            xm = np.mean(x)
            return (x-xm) * np.hanning(len(x)) + xm

        L = []
        for i in range(200):
            sim.simulate(4*n, 1.0, mu, norm=norm)
            sim.apply_lag(phase=False)
            s,i  = az.misc.split_array(sim.y[:n], 64)
            S,i  = az.misc.split_array(sim.x[:n], 64)
            s = [taper(x) for x in s]
            S = [taper(x) for x in S]
            l = az.LCurve.calculate_lag(s, S, 1.0, {'by_n':[2,1]})
            #l = az.LCurve.calculate_lag(s, S, 1.0)
            L.append(l[:3])
        L = np.array(L)


        fm, lm = sim.normalized_lag[0][1:-1], sim.normalized_lag[1][1:-1]
        import pylab as plt
        fq = l[0]
        plt.semilogx(fm, lm)
        plt.plot(fq, np.mean(L[:,1,:], 0), 'o-', color='C2')
        plt.plot(fq, np.median(L[:,1,:], 0), color='C3')
        plt.plot(fq, L[:,1,:].mean(0)+L[:,1,:].std(0), '--', color='C2')
        plt.plot(fq, L[:,1,:].mean(0)-L[:,1,:].std(0), '--', color='C2')
        plt.plot(fq, L[:,1,:].mean(0)+L[:,2,:].mean(0), '-.', color='C3')
        plt.plot(fq, L[:,1,:].mean(0)-L[:,2,:].mean(0), '-.', color='C3')
        plt.show()

