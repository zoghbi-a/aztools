"""A module for handling light curves."""

from itertools import groupby
from typing import Union

import numpy as np

from .misc import group_array


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


    @staticmethod
    def calculate_psd(rate: Union[list, np.ndarray],
                      deltat: float,
                      norm: str = 'var',
                      **kwargs):
        """Calculate raw psd from a list of light curves.

        Parameters
        ----------
        rate: np.ndarray or a list of np.ndarray
            An array or a list of arrays of lcurve rates
        deltat: float
            Time bin width of the light curve(s).
        norm: str
            Psd normalization: var|rms|leahy

        Keywords
        --------
        rerr: np.ndarray or a list of np.ndarray
            The measurement error arrays that corresponds to rate.
            If not given, assume, poisson noise.
        bgd: np.ndarray or a list of np.ndarray.
            The background rate arrays that corresponds to source rate. 
            In this case, rate above is assumed background subtracted.
        taper: bool
            Apply Hanning tapering before calculating the psd
            see p388 Bendat & Piersol; the psd needs to be multiplied
            by 8/3 to componsate for the reduced variance.

        Return
        ------
        freq, rpsd, nois. 

        """

        # check input #
        if not isinstance(rate[0], (np.ndarray, list)):
            rate = [rate]

        if norm not in ['var', 'rms', 'leahy']:
            raise ValueError('norm need to be var|rms|leahy')

        # rerr and bgd; for estimating noise level #
        rerr = kwargs.get('rerr', None)
        bgd  = kwargs.get('bgd', 0.0)
        if not isinstance(bgd, (np.ndarray, list)):
            bgd = [bgd for rat in rate]
        if rerr is None:
            # err is sqrt of number of counts/bin
            rerr = [np.sqrt((rat+bak)/deltat) for rat,bak in zip(rate, bgd)]


        # tapering ? #
        taper = kwargs.get('taper', False)
        if taper:
            rate = [(rat-rat.mean()) * np.hanning(len(rat)) + rat.mean()
                    for rat in rate]


        # fft; remove the 0-freq and the nyquist #
        freq = [np.fft.rfftfreq(len(rat), deltat)[1:-1] for rat in rate]
        rfft = [np.fft.rfft(rat)[1:-1] for rat in rate]
        mean = [np.mean(rat) for rat in rate]

        # normalize psd #
        expo = {'var':0, 'leahy':1, 'rms':2}
        rpsd = [(2.*deltat / (len(rat) * mu_**expo[norm])) * np.abs(ratf)**2
                    for rat,ratf,mu_ in zip(rate, rfft, mean)]

        # renormalize rpsd if tapering has been applied #
        if taper:
            rpsd = [rat * 8/3 for rat in rpsd]

        ## ------ noise level ------- ##
        # noise level is: 2*(mu+bgd)/(mu^2) for RMS normalization; eqn A2, Vaughan+03
        # This the special case of poisson noise light curves.
        # Generally: noise = <e^2>/(mu^2 fq_nyq)
        # where <e^2> is the averaged squared error in the measurements
        # which for poisson case: e = sqrt((mu+bgd)*dt)/dt = sqrt((mu+bgd)/dt)
        # --> <e^2> = (mu+bgd)/dt
        # fq_nyq: is the Nyquist frequency: fq_nyq = 1/(2*dt)
        # ==> for poisson case: noise = 2*(mu+bgd)/mu^2
        # For other normalization, we need to renormalize accordingly
        ## -------------------------- ##
        fnyq = 0.5/deltat
        nois = [frq*0+np.mean(ree**2)/(fnyq*mu_**expo[norm])
                    for frq,ree,mu_ in zip(freq, rerr, mean)]

        # flattern lists #
        concat = np.concatenate
        freq = concat(freq)
        isort = np.argsort(freq)
        freq = freq[isort]
        rpsd = concat(rpsd)[isort]
        nois = concat(nois)[isort]

        return freq, rpsd, nois


    @staticmethod
    def bin_psd(freq: np.ndarray,
                rpsd: np.ndarray,
                fqbin: dict,
                noise: bool = None,
                logavg: bool = True):
        """Bin power spectrum.

        Parameters
        ----------
        freq: np.ndarray
            Array of raw frequencies.
        rpsd: np.ndarray
            Array of raw powers.
        fqbin: dict
            Binning dict to be passed to @misc.group_array
            to bin the frequency axis.
        noise: bool
            Array of noise values or None.
        logavg: bool
            Do averaging in log-space, and correct for
            bias. Otherwise it is linear averaging.

        Return
        ------
            freq, psd, psde, desc::dict having some useful info

        """

        # ensure the arrays are compatible #
        if len(freq) != len(rpsd):
            raise ValueError('freq and rpsd are not compatible')

        if noise is None:
            noise = np.zeros_like(freq) + 1e-10

        # group the freq array #
        idx = group_array(freq, do_unique=True, **fqbin)
        fqm = [len(jdx) for jdx in idx]
        fql = [freq[i].min() for i in idx] + [freq[idx[-1].max()]]


        # do the actual binning #
        # the noise in the logavg case is without bias correction
        if logavg:
            frq = [10**np.mean(np.log10(freq[i])) for i in idx]
            psd = [10**np.mean(np.log10(rpsd[i])) for i in idx]
            nse = [10**np.mean(np.log10(noise[i])) for i in idx]
            per = [np.log(10)*psd[ipp]*(0.310/fqm[ipp])**0.5
                   for ipp in range(len(psd))]
        else:
            frq = [np.mean(freq[i]) for i in idx]
            psd = [np.mean(rpsd[i]) for i in idx]
            nse = [np.mean(noise[i]) for i in idx]
            per = [psd[ipp]*(1./fqm[ipp])**0.5
                   for ipp in range(len(psd))]

        frq, psd, psde, nse = np.array(frq), np.array(psd), np.array(per), np.array(nse)

        # bias correction #
        #####################################
        # From the simulations in test_lcurve.py:
        # 1- Whenever logavg=True is used, bias correciton needs
        #    to be applied. Logavg=True does better, most of the
        #    times, particularly when averaging neighboring frequencies
        # bias function: bias_f(2) ~ 0.253 in Papadakis93
        # bias_f = lambda k: -sp.digamma(k/2.)/np.log(10)
        #####################################
        bias = np.zeros_like(psd) + 0.253
        if logavg:
            psd *= 10**bias

        # return #
        desc = {'fql': fql, 'fqm':fqm, 'noise':nse, 'bias':bias}
        return frq, psd, psde, desc
