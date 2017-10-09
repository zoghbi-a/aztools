
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
        self.time = t
        self.rate = r
        self.rerr = re
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



