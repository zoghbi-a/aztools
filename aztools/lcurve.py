
import numpy as np


class LCurve(object):
    """Light curve holder class"""


    def __init__(self, t, r, re, dt=None, fexp=None):
        """Initialize LCurve from array r and optional t, re, dt

        Parameters:
            t:  an array containing the time axis
            r:  an array containing the count rate
            re: an array containing the measurement errors.
            dt: time sampling. If not given, dt = min(diff(t))
            fexp: fraction exposure array. If not given, fexp=np.ones_like(t)
        """

        # check input arrays #
        if not (len(t) == len(r) == len(re)):
            raise ValueError('arrays t, r, re do not match')


        # time sampling #
        if dt is None:
            dt = np.min(np.diff(t))


        # fraction exposure #
        if fexp is None:
            fexp = np.ones_like(t)


        # is the light curve evenly sampled? #
        iseven = np.all(np.isclose(np.diff(t), dt))


        # global variables #
        self.time = np.array(t)
        self.rate = np.array(r)
        self.rerr = np.array(re)
        self.fexp = fexp
        self.dt = dt
        self.iseven = iseven
        self.nt = len(t)



    def make_even(self, fill=np.nan):
        """Make the light curve even in time, filling gaps with fill

        Parameters:
            fill: value to use in gaps.

        Returns:
            a new LCurve object

        """

        if self.iseven:
            print('LCurve is already even')
            return self

        # make sure time axis can be made even #
        itime = (self.time - self.time[0]) / self.dt
        if not np.allclose(itime - np.array(itime, np.int), 0):
            raise ValueError('time axis cannot be made even')


        # do work #
        t_new = np.arange(np.int(itime[-1]) + 1) * self.dt + self.time[0]
        idx = np.in1d(t_new, self.time)
        r, re = [np.zeros_like(t_new) + fill for i in range(2)]
        f = np.zeros_like(t_new)
        r[idx]  = self.rate
        re[idx] = self.rerr
        f[idx]  = self.fexp

        # return a new LCurve object #
        return LCurve(t_new, r, re, self.dt, f)



    def rebin(self, factor, error='poiss', min_exp=0.0):
        """Rebin the light curve to so new_dt = dt*factor
        
        Parameters:
            factor: rebinning factor. dt_new = factor * dt
            error: error type (poiss|norm). 
                If poiss: rerr = sqrt(rate*dt)/dt, otherwise,
                errors are summed quadratically
            min_exp: minimum fractional exposure to leave [0-1]

        Return:
            new binned LCurve

        """

        # check input error type #
        if not error in ['poiss', 'norm']:
            raise ValueError('error need to be poiss|norm')


        # make lc evenly sampled, so we bin arrays easily #
        lc = self.make_even()


        # new sampling time and length #
        factor = np.int(factor)
        dt_new = lc.dt * factor
        nt_new = lc.nt//factor
        nt_scal = nt_new * factor


        # pre-binning #
        t  = lc.time[:nt_scal].reshape(nt_new, factor)
        r  = lc.rate[:nt_scal].reshape(nt_new, factor)
        re = lc.rerr[:nt_scal].reshape(nt_new, factor)
        f  = lc.fexp[:nt_scal].reshape(nt_new, factor)


        # do binning #
        t  = t.mean(1)
        it = np.array([~np.all(_f==0) for _f in f])
        f  = f.mean(1)
        f_ = np.clip(f, 1e-20, np.inf)
        r  = np.nansum(r, 1) * 1./ (factor*f_)
        r[~it] = np.nan

        if error == 'poiss':
            re = np.sqrt(r*dt_new)/dt_new
        else:
            re = np.nansum(re**2, 1)**0.5
            re[~it] = np.nan

        # leave nan values if original lc had nan (i.e it was even)
        if self.iseven: it = np.isfinite(it)

        # filter on fracexp if needed #
        if min_exp > 0:
            it[f < min_exp] = False

        # return a new LCurve object #
        return LCurve(t[it], r[it], re[it], dt_new, f[it])


