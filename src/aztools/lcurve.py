"""A module for handling light curves."""

from itertools import groupby

import numpy as np


class LCurve:
    """Light curve class"""

    def __init__(self,
                 tarr: np.ndarray,
                 rarr: np.ndarray,
                 **kwargs):
        """Initialize LCurve with tarr, rarr and optional rerr, fexp

        Parameters
        ----------
        tarr: np.ndarray
            An array containing the time axis
        rarr: np.ndarray
            An array containing the count rate
            
        Keywords
        --------
        rerr: np.ndarray
            An array containing the measurement errors.
        deltat: float
            Time sampling. If not given, dt = min(diff(t))
        fexp: np.ndarray
            fraction exposure array. If not given, fexp=np.ones_like(t)
        
        """

        # keyword arguments
        rerr = kwargs.get('rerr', None)
        deltat = kwargs.get('deltat', None)
        fexp = kwargs.get('fexp')

        # check input arrays #
        if not len(tarr) == len(rarr) == len(rerr):
            raise ValueError('arrays tarr, rarr, rerr do not match')

        # time sampling #
        if deltat is None:
            deltat = np.min(np.diff(tarr))

        # fraction exposure #
        if fexp is None:
            fexp = np.ones_like(tarr)

        # global variables #
        self.time = np.array(tarr)
        self.rate = np.array(rarr)
        self.rerr = np.array(rerr)
        self.fexp = fexp
        self.deltat = deltat
        self.iseven = np.all(np.isclose(np.diff(tarr), deltat))
        self.npoints = len(tarr)


    def __repr__(self):
        """LCurve as a str"""
        return f'<LCurve :: nt({self.npoints}) :: dt({self.deltat})>'


    def make_even(self, fill: float = np.nan):
        """Make the light curve even in time, filling gaps with fill

        Parameters
        ----------
        fill: float
            The value to use in gaps.

        Return
        ------
        a new LCurve object

        """

        if self.iseven:
            return self

        # make sure time axis can be made even #
        itime = np.round((self.time - self.time[0]) / self.deltat)
        if not np.allclose(itime - np.array(itime, np.int64), 0):
            raise ValueError('time axis cannot be made even')


        # do the work #
        t_new = np.arange(np.int64(itime[-1]) + 1) * self.deltat + self.time[0]
        idx = np.in1d(t_new, self.time)
        rarr = np.zeros_like(t_new) + fill
        rerr = np.zeros_like(t_new) + fill
        fexp = np.zeros_like(t_new)
        rarr[idx] = self.rate
        rerr[idx] = self.rerr
        fexp[idx] = self.fexp

        # return a new LCurve object #
        return LCurve(t_new, rarr, rerr=rerr, deltat=self.deltat, fexp=fexp)


    def rebin(self, factor: int, error: str = 'norm', min_exp: float = 0.0):
        """Rebin the light curve to so new_deltat = deltat*factor
        
        Parameters
        ----------
        factor: int
            Rebinning factor. deltat_new = factor * deltat
        error: str
            Error type (poiss|norm). 
            If poiss: rerr = sqrt(rate*dt)/dt, otherwise,
            errors are summed quadratically
        min_exp: float
            Minimum fractional exposure to leave [0-1]

        Return:
            new binned LCurve

        """

        # check input error type #
        if error not in ['poiss', 'norm']:
            raise ValueError('error need to be poiss|norm')


        # make lc evenly sampled, so we bin arrays easily #
        newlc = self.make_even()


        # new sampling time and length #
        factor = np.int64(factor)
        dt_new = newlc.deltat * factor
        nt_new = newlc.npoints//factor


        # pre-binning #
        tarr = newlc.time[:nt_new*factor].reshape(nt_new, factor)
        rarr = newlc.rate[:nt_new*factor].reshape(nt_new, factor)
        rerr = newlc.rerr[:nt_new*factor].reshape(nt_new, factor)
        fexp = newlc.fexp[:nt_new*factor].reshape(nt_new, factor)

        # rescale the rates to pre-fexp counts/bin #
        carr = rarr * (fexp * newlc.deltat)
        cerr = rerr * (fexp * newlc.deltat)


        # do binning #
        tarr = np.mean(tarr, 1)
        carr = np.nansum(carr, 1)
        if error == 'poiss':
            cerr = np.sqrt(carr)
            cerr[cerr==0] = np.nanmean(cerr[cerr!=0])
        else:
            cerr = np.nansum(cerr**2, 1)**0.5
        fexp = np.mean(fexp, 1)

        # calculate the rate again
        fexps = np.array(fexp)
        itarr = fexps != 0
        fexps[~itarr] = np.nan
        rarr = carr /(dt_new * fexps)
        rerr = cerr/(dt_new * fexps)

        # leave nan values if original lc had nan (i.e it was even)
        if self.iseven:
            itarr = np.ones_like(itarr) == 1

        # filter on fracexp if needed #
        if min_exp > 0:
            itarr[fexp < min_exp] = False

        # return a new LCurve object #
        return LCurve(tarr[itarr], rarr[itarr],
                      rerr=rerr[itarr], deltat=dt_new, fexp=fexp[itarr])


    def interp_small_gaps(self,
                          maxgap: int = None,
                          noise: str = 'poiss',
                          seed: int = None):
        """Interpolate small gaps in the lightcurve if the gap
            is <maxgap; applying noise if requested

        Parameters
        ----------
        maxgap: int
            The maximum length of a gap to be interpolated.
        noise: str:
            poiss|norm|None
        seed: int
            Random seed if noise is requested

        """

        if not self.iseven:
            raise ValueError('lc is not even; make even first')

        # random seed if noise is needed #
        if noise is not None:
            np.random.seed(seed)


        # find gap lengths in the data #
        maxn = self.npoints if maxgap is None else maxgap
        iarr = [list(igrp[1]) for igrp in
                groupby(
                    np.arange(self.npoints),
                    lambda ixarr:np.isfinite(self.rate[ixarr])
                )
               ]
        # indices of non-finite segments #
        iinf = iarr[(1 if np.isfinite(iarr[0][0]) else 0)::2]
        # length of each non-finite segment #
        iinf = [i for i in iinf if len(i)<=maxn]
        iinf = [j for i in iinf for j in i]


        # interpolate all values then keep only those with length<maxn #
        idx = np.isfinite(self.rate)
        yarr = np.interp(self.time, self.time[idx], self.rate[idx])
        yerr = np.zeros_like(yarr)
        mean = np.mean(self.rerr[idx])
        if noise is None:
            # no noise requested; the value is not altered from the interp
            # while the error is the average of all errors
            yerr += mean

        elif noise == 'poiss':
            # apply noise to counts/bin then convert back to counts/s
            ypoiss = np.random.poisson(yarr*self.deltat)
            yarr = ypoiss / self.deltat
            yerr = np.sqrt(ypoiss) / self.deltat
            # reset points where y=0 (hence ye=0)
            yerr[ypoiss == 0] = mean

        elif noise == 'norm':
            yarr += np.random.randn(len(yarr)) * mean
            yerr += mean

        # now fill in the gaps with length<maxn #
        self.rate[iinf] = yarr[iinf]
        self.rerr[iinf] = yerr[iinf]
