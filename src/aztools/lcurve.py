"""A module for handling light curves."""

from itertools import groupby
from typing import Union

import numpy as np

from .misc import group_array, read_fits_lcurve

__all__ = ['LCurve']

class LCurve:
    """Light curve class"""

    def __init__(self,
                 tarr: np.ndarray,
                 rarr: np.ndarray,
                 rerr: np.ndarray = None,
                 **kwargs):
        """Initialize LCurve with tarr, rarr and optional rerr, fexp

        Parameters
        ----------
        tarr: np.ndarray
            An array containing the time axis
        rarr: np.ndarray
            An array containing the count rate
        rerr: np.ndarray
            An array containing the measurement errors.
            
        Keywords
        --------
        deltat: float
            Time sampling. If not given, dt = min(diff(t))
        fexp: np.ndarray
            fraction exposure array. If not given, fexp=np.ones_like(t)
        
        """

        # keyword arguments
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


    @staticmethod
    def calculate_lag(xarr: Union[np.ndarray, list],
                      yarr: Union[np.ndarray, list],
                      deltat: float,
                      fqbin: dict = None,
                      **kwargs):
        """Calculate and bin lags from two lists of light curves.
        
        Parameters
        ----------
        xarr: np.ndarray or list
            Array or list of arrays of light curve rates.
        yarr: np.ndarray or list
            Array or list of arrays of reference light curve rates.
        deltat: float
            Time bin width of the light curve.
        fqbin: dict
            Binning dict to be passed to @misc.group_array
            to bin the frequency axis. If None, return raw lag

        Keywords
        --------
        xerr: np.ndarray or a list of np.ndarray
            The measurement error arrays that corresponds to rate.
            If not given, assume, poisson noise.
        xbgd: np.ndarray or a list of np.ndarray.
            The background rate arrays that corresponds to source rate. 
            In this case, rate above is assumed background subtracted.
        yerr: np.ndarray or a list of np.ndarray
            The measurement error arrays that corresponds to yarr.
            If not given, assume, poisson noise.
        ybgd: np.ndarray or a list of np.ndarray.
            The background rate arrays that corresponds to ref_rate. 
            In this case, yarr above is assumed background subtracted.
        phase: bool
            return phase lag instead of time lag
        taper: bool
            Apply Hanning tapering before calculating the fft
            see p388 Bendat & Piersol; the fft need to be multiplied
            by sqrt(8/3) to componsate for the reduced variance. Default: False
        norm: str
            How to normalize the fft during the calculations. None|rms|leahy|var.
            Default is None, so the calculations is done with raw numpy fft


        Return
        ------
        freq, lag, lage, extra
        extra = {'fqm', 'fql', 'xlimit', 'ylimit' ..}
        """

        phase = kwargs.get('phase', False)


        # check input #
        if not isinstance(xarr[0], (np.ndarray, list)):
            xarr = [xarr]
        if not isinstance(yarr[0], (np.ndarray, list)):
            yarr = [yarr]

        # check that lc and reference are compatible #
        for xar,yar in zip(xarr, yarr):
            if len(xar) != len(yar):
                raise ValueError('xarr and yarr are incompatible')


        # error and background values for estimating noise level #
        xbgd = kwargs.get('xbgd', 0.0)
        ybgd = kwargs.get('ybgd', 0.0)
        if not isinstance(xbgd, (np.ndarray, list)):
            xbgd = [xbgd for _ in xarr]
        if not isinstance(ybgd, (np.ndarray, list)):
            ybgd = [ybgd for _ in yarr]

        xerr = kwargs.get('xerr', None)
        yerr = kwargs.get('yerr', None)
        if xerr is None:
            # this is not always correct!
            xerr = [np.sqrt((xar+xbg)/deltat) for xar,xbg in zip(xarr, xbgd)]
        if yerr is None:
            # this is not always correct!
            yerr = [np.sqrt((yar+ybg)/deltat) for yar,ybg in zip(yarr, ybgd)]

        if not isinstance(xerr[0], (np.ndarray, list)):
            xerr = [xerr]
        if not isinstance(yerr[0], (np.ndarray, list)):
            yerr = [yerr]


        # tapering ? #
        taper = kwargs.get('taper', True)
        taper_factor = 1.0
        if taper:
            xarr = [(arr-arr.mean()) * np.hanning(len(arr)) + arr.mean() for arr in xarr]
            yarr = [(arr-arr.mean()) * np.hanning(len(arr)) + arr.mean() for arr in yarr]
            taper_factor = np.sqrt(8/3)


        # normalization ? #
        norm = kwargs.get('norm', None)
        if not norm in [None, 'rms', 'leahy', 'var']:
            raise ValueError('Unknown norm value. It should be None|rms|leahy|var')
        expo = {'var':0, 'leahy':1, 'rms':2}
        def normf(arr):
            return 1.0 if norm is None else (
                2.*deltat / (len(arr) * np.mean(arr)**expo[norm]))**0.5


        # fft; remove the 0-freq and the nyquist #
        # noise level in psd. See comments in @calculate_psd #
        # noise level is: <e^2>/(mu^2 fq_nyq) for rms norm; then renorm accordingly
        fnyq = 0.5/deltat
        freq = [np.fft.rfftfreq(len(_), deltat)[1:-1] for _ in xarr]

        xfft = [np.fft.rfft(_)[1:-1]*taper_factor*normf(_) for _ in xarr]
        xpsd = [np.abs(_)**2 for _ in xfft]
        xnse = [_[0]*0+(np.mean(_[1]**2)*len(_[1])*normf(_[2])**2)/(fnyq*2*deltat)
                    for _ in zip(freq, xerr, xarr)]

        yfft = [np.fft.rfft(_)[1:-1]*taper_factor*normf(_) for _ in yarr]
        ypsd = [np.abs(_)**2 for _ in yfft]
        ynse = [_[0]*0+(np.mean(_[1]**2)*len(_[1])*normf(_[2])**2)/(fnyq*2*deltat)
                    for _ in zip(freq, yerr, yarr)]

        crss  = [_[1]*np.conj(_[0]) for _ in zip(xfft, yfft)]


        # flattern lists #
        concat = np.concatenate
        freq = concat(freq)
        isort = np.argsort(freq)
        freq = freq[isort]
        crss = concat(crss)[isort]
        xpsd = concat(xpsd)[isort]
        ypsd = concat(ypsd)[isort]
        xnse = concat(xnse)[isort]
        ynse = concat(ynse)[isort]


        # save raw values
        raw = {'freq': freq, 'crss':crss,
               'xfft': xfft, 'xpsd': xpsd, 'xnse': xnse,
               'yftt': yfft, 'ypsd': ypsd, 'ynse': ynse}

        # do we need just raw lags? #
        if fqbin is None:
            lag = np.angle(crss) / (1. if phase else 2*np.pi*freq)
            return freq, lag, raw


        # bin the lag #
        idx = group_array(freq, do_unique=True, **fqbin)
        fqm = np.array([len(jdx) for jdx in idx])
        fql = np.array([freq[i].min() for i in idx] + [freq[idx[-1].max()]])


        def mean(arr):
            return np.array([np.mean(arr[jdx]) for jdx in idx])
        def lmean(arr):
            return np.array([10**(np.mean(np.log10(arr[jdx]))) for jdx in idx])

        freq = lmean(freq)
        xpsd = mean(xpsd)
        ypsd = mean(ypsd)
        xnse = mean(xnse)
        ynse = mean(ynse)
        crss = mean(crss)


        # phase lag and its error #
        # g2 is caluclated without noise subtraciton
        # see paragraph after eq. 17 in Nowak+99
        # see eq. 11, 12 in Uttley+14. Nowak (and Uttley too) clearly
        # states that the noise shouldn't be subtracted)
        lag = np.angle(crss)
        nn2 = ((xpsd - xnse)*ynse + (ypsd - ynse)*xnse + xnse*ynse) / fqm
        gg2 = (np.abs(crss)**2) / (xpsd * ypsd)

        # mask out points where coherence is undefined #
        gg2 = np.clip(gg2, 1e-5, 1.0)
        lag_e = np.clip(np.sqrt((1 - gg2) / (2*gg2*fqm)), 0, np.pi)


        # coherence gamma_2 #
        # here we subtract the noise; see eq. 8
        # in Vaughan+97 and related definitions
        coh   = (np.abs(crss)**2 - nn2) / ((xpsd-xnse) * (ypsd-ynse))
        coh = np.clip(coh, 1e-5, 1-1e-5)
        dcoh  = (2/fqm)**0.5 * (1 - coh)/np.sqrt(coh)
        coh_e = coh * (fqm**-0.5) * ((2*nn2*nn2*fqm)/(np.abs(crss)**2 - nn2)**2 +
                (xnse**2/(xpsd-xnse)**2) + (ynse**2/(ypsd-ynse)**2) + (fqm*dcoh/coh**2))**0.5
        coh_e[(coh - coh_e) < 0] = coh[(coh - coh_e) < 0]


        # rms spectrum from psd; error from eq. 14 in Uttley+14 #
        # the rms here is in absolute not fractional units
        dfq = fql[1:]-fql[:-1]
        xmu = np.mean([np.mean(_) for _ in xarr])
        ymu = np.mean([np.mean(_) for _ in yarr])
        rms = xmu * (dfq * np.abs(xpsd - xnse))**0.5
        sigx2 = rms**2
        sigxn2 = dfq * xnse * xmu**2
        rmse = ((2*sigx2*sigxn2 + sigxn2**2) / (2*fqm*sigx2) ) **0.5
        ibad = xpsd<xnse
        rms[ibad] = 0.0
        rmse[ibad] = np.max(np.concatenate((rms, rmse)))


        # covariance: eq. 13, 15 in Uttley+14 #
        # again in absolute not fractional units #
        cov  = ( (np.abs(crss)**2 - nn2) * xmu * xmu * dfq / (ypsd-ynse) )
        ibad = (cov < 0) | (ypsd<=ynse)
        cov  = np.abs(cov)**0.5
        sigy2 = dfq * np.abs(ypsd-ynse) * ymu**2
        sigyn2 = dfq * ynse * ymu**2
        cove = ((sigyn2*cov**2 + sigy2*sigxn2 + sigxn2*sigyn2) / (2*fqm*sigy2))**0.5
        cov[ibad] = 0.0
        cove[ibad] = np.max(np.concatenate((cov, cove)))


        # limits on lag measurements due to poisson noise #
        # equation 30 in Vaughan+2003 #
        xlimit = np.sqrt(np.abs(xnse/(fqm * gg2 * (xpsd-xnse))))
        ylimit = np.sqrt(np.abs(ynse/(fqm * gg2 * (ypsd-ynse))))
        xlimit = np.clip(xlimit, -np.pi, np.pi)
        ylimit = np.clip(ylimit, -np.pi, np.pi)


        # do we need time lag instead of phase lag? #
        if not phase:
            lag    /= (2*np.pi*freq)
            lag_e  /= (2*np.pi*freq)
            xlimit /= (2*np.pi*freq)
            ylimit /= (2*np.pi*freq)


        # return #
        extra = {'fql': fql, 'fqm':fqm, 'xlimit':xlimit, 'yLimit':ylimit,
                'limit_avg':(xlimit+ylimit)/2, 'coh': np.array([coh, coh_e]),
                'xpsd': xpsd, 'xnse': xnse, 'ypsd': ypsd, 'ynse': ynse,
                'cxd': crss, 'nn2': nn2, 'gg2': gg2, 'idx': idx, 'freq': freq,
                'raw': raw, 'rms': np.array([rms, rmse]), 'cov': np.array([cov, cove])}

        return freq, lag, lag_e, extra


    @staticmethod
    def read_pn_lcurve(fits_file, **kwargs):
        """Read pn lcurve fits file.
            This sets values relevant to PN and calls @misc.read_fits_lcurve

        Parameters
        ----------
        fits_file: str
            The name of the files file

        Keywords
        --------
        See @misc.read_fits_lcurve


        Return
        ------
        LCurve object
        """

        # set values relevant to XMM-PN files #
        kwargs.setdefault('min_exp' , 0.7)
        kwargs.setdefault('gti_tbl' , 2)

        data, deltat = read_fits_lcurve(fits_file, **kwargs)
        return LCurve(data[0], data[1], rerr=data[2], deltat=deltat, fexp=data[3])


    @staticmethod
    def read_pca_lcurve(fits_file, **kwargs):
        """Read pca lcurve fits file.
            This sets values relevant to PCA and calls @misc.read_fits_lcurve

        Parameters
        ----------
        fits_file: str
            The name of the files file

        Keywords
        --------
        See @misc.read_fits_lcurve


        Return
        ------
        LCurve object
        """

        # set values relevant to PCA files #
        kwargs.setdefault('min_exp' , 0.99)
        kwargs.setdefault('gti_tbl' , 'STDGTI')

        data, deltat = read_fits_lcurve(fits_file, **kwargs)
        return LCurve(data[0], data[1], rerr=data[2], deltat=deltat, fexp=data[3])


    @staticmethod
    def read_nu_lcurve(fits_file, **kwargs):
        """Read nustar lcurve fits file.
            This sets values relevant to NUSTAR and calls @misc.read_fits_lcurve

        Parameters
        ----------
        fits_file: str
            The name of the files file.

        Keywords
        --------
        See @misc.read_fits_lcurve


        Return
        ------
        LCurve object
        """

        # set values relevant to NUSTAR files #
        kwargs.setdefault('min_exp' , 0.1)
        kwargs.setdefault('gti_tbl' , 'GTI')
        kwargs.setdefault('gti_skip', 3.0)

        data, deltat = read_fits_lcurve(fits_file, **kwargs)
        return LCurve(data[0], data[1], rerr=data[2], deltat=deltat, fexp=data[3])


    @staticmethod
    def read_xis_lcurve(fits_file, **kwargs):
        """Read suzaku xis lcurve fits file.
            This sets values relevant to Suzaku XIS and calls @misc.read_fits_lcurve

        Parameters
        ----------
        fits_file: str
            The name of the files file.

        Keywords
        --------
        See @misc.read_fits_lcurve


        Return
        ------
        LCurve object
        """

        # set values relevant to XIS files #
        kwargs.setdefault('min_exp' , 0.1)
        kwargs.setdefault('gti_tbl' , 'GTI')

        data, deltat = read_fits_lcurve(fits_file, **kwargs)
        return LCurve(data[0], data[1], rerr=data[2], deltat=deltat, fexp=data[3])


    @staticmethod
    def read_ni_lcurve(fits_file, **kwargs):
        """Read nicer lcurve fits file.
            This sets values relevant to NICER and calls @misc.read_fits_lcurve

        Parameters
        ----------
        fits_file: str
            The name of the files file.

        Keywords:
        See @misc.read_fits_lcurve


        Returns:
            LCurve object
        """

        # set values relevant to NICER files #
        kwargs.setdefault('min_exp' , 0.99)
        kwargs.setdefault('gti_tbl' , 'GTI')

        data, deltat = read_fits_lcurve(fits_file, **kwargs)
        return LCurve(data[0], data[1], rerr=data[2], deltat=deltat, fexp=data[3])
