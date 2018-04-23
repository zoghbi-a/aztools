
import numpy as np
import os

from . import misc


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



    def __repr__(self):
        return ('<LCurve :: nt({}) :: dt({})>').format(self.nt, self.dt)



    def make_even(self, fill=np.nan):
        """Make the light curve even in time, filling gaps with fill

        Parameters:
            fill: value to use in gaps.

        Returns:
            a new LCurve object

        """

        if self.iseven:
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



    def rebin(self, factor, error='norm', min_exp=0.0):
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
        iseven = self.iseven
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

        # rescale the rates to pre-fexp counts/bin #
        c  = r  * (f * lc.dt)
        ce = re * (f * lc.dt)


        # do binning #
        t  = np.mean(t, 1)
        c  = np.nansum(c, 1)
        if error == 'poiss':
            ce = np.sqrt(c)
            ce[ce==0] = np.nanmean(ce[ce!=0])
        else:
            ce = np.nansum(ce**2, 1)**0.5

        f  = np.mean(f, 1)
        fs = np.array(f)
        it = (fs != 0)
        fs[~it] = np.nan
        r  = c /(dt_new * fs)
        re = ce/(dt_new * fs)



        # # leave nan values if original lc had nan (i.e it was even)
        if iseven: 
            it = np.ones_like(it) == 1

        # # filter on fracexp if needed #
        if min_exp > 0:
            it[f < min_exp] = False

        # return a new LCurve object #
        return LCurve(t[it], r[it], re[it], dt_new, f[it])



    def interp_small_gaps(self, maxgap=None, noise='poiss', seed=None):
        """Interpolate small gaps in the lightcurve if the gap
            is <maxgap; applying noise if requested

        Parameters:
            maxgap: the maximum length of a gap to be interpolated
            noise: poiss|norm|None
            seed: random seen if noise is requested

        """
        from itertools import groupby

        if not self.iseven:
            raise ValueError('lc is not even; make even first')

        # random seed if noise is needed #
        if noise is not None:
            np.random.seed(seed)


        # find gap lengths in the data #
        maxn = self.nt if maxgap is None else maxgap
        iarr = [list(i[1]) for i in groupby(np.arange(self.nt), 
                    lambda ix:np.isfinite(self.rate[ix]))]
        # indices of non-finite segments #
        iinf = iarr[(1 if np.isfinite(iarr[0][0]) else 0)::2]
        # length of each non-finite segment #
        iinf = [i for i in iinf if len(i)<=maxn]
        iinf = [j for i in iinf for j in i]
        



        # interpolate all values then keep only those with length<maxn #
        idx = np.isfinite(self.rate)
        y  = np.interp(self.time, self.time[idx], self.rate[idx])
        ye = np.zeros_like(y)
        me = np.mean(self.rerr[idx])
        if noise is None:
            # no noise requested; the value is not altered from the interp
            # while the error is the average of all errors
            ye += me

        elif noise == 'poiss':
            # apply noise to counts/bin then convert back to counts/s
            yp = np.random.poisson(y*self.dt)
            y  = yp / self.dt
            ye = np.sqrt(yp) / self.dt
            # reset points where y=0 (hence ye=0)
            ye[yp == 0] = me
        
        elif noise == 'norm':
            y  += np.random.randn(len(y)) * me
            ye += me

        # now update fill in the gaps with length<maxn #
        self.rate[iinf] = y[iinf]
        self.rerr[iinf] = ye[iinf]




    @staticmethod
    def read_fits_file(fits_file, **kwargs):
        """Read LCurve from fits file

        Parameters:
            fits_file: name of the fits file

        Keywords:
            min_exp: minimum fractional exposure to allow. Default 0.0 for all
            rate_tbl: name or number of hdu that contains lc data. Default: RATE
            rate_col: name or number of rate column. Default: RATE
            time_col: name or number of time column. Default: TIME
            rerr_col: name or number of rerr column. Default: ERROR
            fexp_col: name or number of the fracexp column. Default: FRACEXP
            gti_table: name or number of gti extension hdu. Default: GTI 
            dt_key: name of dt keyword in header. Default: TIMEDEL
            gti_skip: how many seconds to skip at the gti boundaries. Default: 0
            verbose. 


        Returns:
            ldata (shape: 4,nt containing, time, rate, rerr, fexp), dt
        """

        # pyfits #
        import astropy.io.fits as pyfits


        # default parameters #
        min_exp  = kwargs.get('min_exp', 0.)
        rate_tbl = kwargs.get('rate_tbl', 'RATE')
        rate_col = kwargs.get('rate_col', 'RATE')
        time_col = kwargs.get('time_col', 'TIME')
        rerr_col = kwargs.get('rerr_col', 'ERROR')
        fexp_col = kwargs.get('fexp_col', 'FRACEXP')
        gti_tbl  = kwargs.get('gti_tbl' , 'GTI')
        dt_key   = kwargs.get('dt_key', 'TIMEDEL')
        gti_skip = kwargs.get('gti_skip', 0.0)
        verbose  = kwargs.get('verbose', False)


        # does file exist? #
        if not os.path.exists(fits_file):
            raise ValueError('file {} does not exist'.format(fits_file))

        # read file #
        with pyfits.open(fits_file) as fs:
            
            # lc data #
            data = fs[rate_tbl].data
            ldata = np.array([  data.field(time_col),
                                data.field(rate_col),
                                data.field(rerr_col)], dtype=np.double)

            
            # start time and time sampling #
            t0 = (fs[rate_tbl].header['TSTART'] if 'TSTART' in 
                        fs[rate_tbl].header.keys() else 0.0)
            dt = (fs[rate_tbl].header[dt_key] if dt_key in 
                        fs[rate_tbl].header.keys() else None)

            # if the time-axis offset, correct it #
            if t0/ldata[0, 1] > 1e6:
                ldata[0] += t0


            # gti #
            try:
                ghdu = fs[gti_tbl]
                lgti  = np.array([ghdu.data.field(0), ghdu.data.field(1)],
                                dtype=np.double)
            except KeyError:
                if verbose:
                    print('No GTI found in {}'.format(fits_file))
                lgti = np.array([[ldata[0, 0]], [ldata[0, -1]]])


            # fractional exposure #
            try:
                lfracexp = data.field(fexp_col)
            except KeyError:
                if verbose:
                    print('cannot read fracexp_col in {}'.format(fits_file))
                lfracexp = np.ones_like(ldata[0])


            # apply gti #
            igti  = np.isfinite(ldata[0])
            for gstart, gstop in lgti.T:
                igti = igti | ( (ldata[0] >= (gstart+gti_skip)) & 
                                (ldata[0] <= (gstop -gti_skip)) )
            igood = igti & (lfracexp >= min_exp)
            ldata = np.vstack([ldata, lfracexp])
            ldata = ldata[:, igood]

        return ldata, dt


    @staticmethod
    def read_pn_lcurve(fits_file, **kwargs):
        """Read pn lcurve fits_file created with xmmlc_lc.
            This sets values relevant to PN and calls read_fits_file

        Parameters:
            fits_file: name of the files file

        Keywords:
            See @LCurve.read_fits_file


        Returns:
            LCurve object
        """

        # set values relevant to XMM-PN files #
        kwargs.setdefault('min_exp' , 0.1)
        kwargs.setdefault('gti_tbl' , 2)
    
        data, dt = LCurve.read_fits_file(fits_file, **kwargs)
        return LCurve(data[0], data[1], data[2], dt, data[3])        


    @staticmethod
    def read_pca_lcurve(fits_file, **kwargs):
        """Read pca lcurve fits_file.
            This sets values relevant to PCA and calls read_fits_file

        Parameters:
            fits_file: name of the files file

        Keywords:
            See @LCurve.read_fits_file


        Returns:
            LCurve object
        """

        # set values relevant to XMM-PN files #
        kwargs.setdefault('min_exp' , 0.99)
        kwargs.setdefault('gti_tbl' , 'STDGTI')
    
        data, dt = LCurve.read_fits_file(fits_file, **kwargs)
        return LCurve(data[0], data[1], data[2], dt, data[3])  


    @staticmethod
    def read_nu_lcurve(fits_file, **kwargs):
        """Read nustar lcurve fits_file.
            This sets values relevant to NUSTAR and calls read_fits_file

        Parameters:
            fits_file: name of the files file

        Keywords:
            See @LCurve.read_fits_file


        Returns:
            LCurve object
        """

        # set values relevant to XMM-PN files #
        kwargs.setdefault('min_exp' , 0.1)
        kwargs.setdefault('gti_tbl' , 'GTI')
        kwargs.setdefault('gti_skip', 3.0)
    
        data, dt = LCurve.read_fits_file(fits_file, **kwargs)
        return LCurve(data[0], data[1], data[2], dt, data[3])  


    @staticmethod
    def calculate_psd(rate, dt, norm='var', **kwargs):
        """Calculate raw psd from a list of light curves.
        
        Parameters:
            rate: array or list of arrays of lcurve rates
            dt: time bin width of the light curve
            norm: psd normalization: var|rms|leahy

        Keywords:
            rerr: array or list of errors on rate. If not give,
                assume, poisson noise.
            bgd: array or list of background rates. In this case,
                rate above is assumed background subtracted.
            taper: apply Hanning tapering before calculating the psd
                see p388 Bendat & Piersol; the psd need to be multiplied
                by 8/3 to componsate for the reduced variance.
        
        Return:
            freq, rpsd, nois. 
        """

        # check input #
        if not isinstance(rate[0], (np.ndarray, list)):
            rate = [rate]

        if not norm in ['var', 'rms', 'leahy']:
            raise ValueError('norm need to be var|rms|leahy')

        # rerr and bgd; for estimating noise level #
        rerr = kwargs.get('rerr', None)
        bgd  = kwargs.get('bgd', 0.0)
        if not isinstance(bgd, (np.ndarray, list)):
            bgd = [bgd for r in rate]
        if rerr is None:
            rerr = [np.sqrt((r+b)/dt) for r,b in zip(rate, bgd)]


        # tapering ? #
        taper = kwargs.get('taper', False)
        if taper:
            rate = [(r-r.mean()) * np.hanning(len(r)) + r.mean() for r in rate]


        # fft; remove the 0-freq and the nyquist #
        freq = [np.fft.rfftfreq(len(r), dt)[1:-1] for r in rate]
        rfft = [np.fft.rfft(r)[1:-1] for r in rate]
        mean = [np.mean(r) for r in rate]

        # normalize psd #
        expo = {'var':0, 'leahy':1, 'rms':2} 
        rpsd = [(2.*dt / (len(r) * mu**expo[norm])) * np.abs(rf)**2
                    for r,rf,mu in zip(rate, rfft, mean)]

        # renormalize rpsd if tapering has been applied #
        if taper:
            rpsd = [r * 8/3 for r in rpsd]

        ## ------ noise level ------- ##
        # noise level is: 2*(mu+bgd)/(mu^2) for RMS normalization
        # This the special case of poisson noise light curves.
        # Generally: noise = <e^2>/(mu^2 fq_nyq)
        # where <e^2> is the averaged squared error in the measurements
        # which for poisson case: e = sqrt((mu+bgd)*dt)/dt = sqrt((mu+bgd)/dt)
        # --> <e^2> = (mu+bgd)/dt
        # fq_nyq: is the Nyquist frequency: fq_nyq = 1/(2*dt)
        # ==> for poisson case: noise = 2*(mu+bgd)/mu^2
        # For other normalization, we need to renormalize accordingly
        ## -------------------------- ##
        fnyq = 0.5/dt
        nois = [ff*0+np.mean(re**2)/(fnyq*mu**expo[norm]) 
                    for ff,re,mu in zip(freq, rerr, mean)]


        # flattern lists #
        _c = np.concatenate
        freq = _c(freq)
        isort = np.argsort(freq)
        freq = freq[isort]
        rpsd = _c(rpsd)[isort]
        nois = _c(nois)[isort]

        return freq, rpsd, nois


    @staticmethod
    def bin_psd(freq, rpsd, fqbin, noise=None, logavg=True):
        """Bin power spectrum.

        Parameters:
            freq: array of frequencies
            rpsd: array of raw powers
            fqbin: binning dict to be passed to @misc.group_array
                to bin the frequency axis
            noise: array of noise.
            logavg: do averaging in log-space, and correct for
                bias. Otherwise it is simple averaging

        Returns:
            fq, psd, psde, desc; with desc having some useful info

        """

        import scipy.special as sp


        # ensure the arrays are compatible #
        if len(freq) != len(rpsd):
            raise ValueError('freq and rpsd are not compatible')

        if noise is None: noise = np.zeros_like(freq)

        # group the freq array #
        nfq = len(freq)
        idx  = misc.group_array(freq, do_unique=True, **fqbin)
        fqm  = [len(i) for i in idx]
        fqL = [freq[i].min() for i in idx] + [freq[idx[-1].max()]]


        # do the actual binning #
        if logavg:
            f  = [10**np.mean(np.log10(freq[i])) for i in idx]
            p  = [10**np.mean(np.log10(rpsd[i])) for i in idx]
            pe = [np.log(10)*p[i]*(0.310/fqm[i])**0.5 for i in range(len(p))]
        else:
            f  = [np.mean(freq[i]) for i in idx]
            p  = [np.mean(rpsd[i]) for i in idx]
            pe = [p[i]*(1./fqm[i])**0.5 for i in range(len(p))]

        # noise; alway nolog #
        n = np.array([np.mean(noise[i]) for i in idx])
        fq, psd, psde = np.array(f), np.array(p), np.array(pe)

        # bias correction #
        #####################################
        # From the simulations in test_lcurve.py:
        # 1- Whenever logavg=True is used, bias correciton needs
        #    to be applied. Logavg=True does better, most of the
        #    times, particularly when averaging neighboring frequencies
        # 2- Red noise leak is important, correct it on case
        #    by case.
        # bias function: bias_f(2) ~ 0.253 in Papadakis93
        # bias_f = lambda k: -sp.digamma(k/2.)/np.log(10)
        #####################################
        bias_f = lambda k: -sp.digamma(k/2.)/np.log(10)
        bias = np.zeros_like(psd) + bias_f(2)
        if logavg: psd *= 10**bias

        # return #    
        desc = {'fqL': fqL, 'fqm':fqm, 'noise':n, 'bias':bias}
        return fq, psd, psde, desc


    @staticmethod
    def calculate_lag(rate, Rate, dt, fqbin=None, **kwargs):
        """Calculate and bin lags from two lists of light curves.
        
        Parameters:
            rate: array or list of arrays of lcurve rates
            Rate: array or list of arrays of Reference lcurve rates
            dt: time bin width of the light curve
            fqbin: binning dict to be passed to @misc.group_array
                to bin the frequency axis. If None, return raw lag

        Keywords:
            rerr: array or list of errors on rate. If not give,
                assume, poisson noise.
            bgd: array or list of background rates. In this case,
                rate above is assumed background subtracted.
            Rerr: array or list of errors on Rate. If not give,
                assume, poisson noise.
            Bgd: array or list of background rates for the reference. 
                In this case, Rate above is assumed background subtracted.
            phase: return phase lag instead of time lag
            mask_coh: mask bins with undefined coherence, setting their
                value to 0 and error to pi
                
        
        Return:
            freq, lag, lage, desc;
            desc = {'fqm', 'fqL', 'limit', 'Limit'}
        """

        phase = kwargs.get('phase', False)
        mask_coh = kwargs.get('mask_coh', False)


        # check input #
        if not isinstance(rate[0], (np.ndarray, list)):
            rate = [rate]
            Rate = [Rate]

        # check that lc and reference are compatible #
        for r1,r2 in zip(rate, Rate):
            if len(r1) != len(r2):
                raise ValueError('rate and Rate are incompatible')


        # rerr and bgd; for estimating noise level #
        rerr = kwargs.get('rerr', None)
        bgd  = kwargs.get('bgd', 0.0)
        if not isinstance(bgd, (np.ndarray, list)):
            bgd = [bgd for r in rate]
        if rerr is None:
            rerr = [np.sqrt((r+b)/dt) for r,b in zip(rate, bgd)]

        # also Rerr and Bgd for the reference lc #
        Rerr = kwargs.get('Rerr', None)
        Bgd  = kwargs.get('Bgd', 0.0)
        if not isinstance(Bgd, (np.ndarray, list)):
            Bgd = [Bgd for r in Rate]
        if Rerr is None:
            Rerr = [np.sqrt((r+b)/dt) for r,b in zip(Rate, Bgd)]


        # fft; remove the 0-freq and the nyquist #
        freq = [np.fft.rfftfreq(len(r), dt)[1:-1] for r in rate]
        rfft = [np.fft.rfft(r)[1:-1] for r in rate]
        Rfft = [np.fft.rfft(r)[1:-1] for r in Rate]
        crss = [R*np.conj(r) for r,R in zip(rfft, Rfft)]
        rpsd = [np.abs(r)**2 for r in rfft]
        Rpsd = [np.abs(r)**2 for r in Rfft]
        


        # noise level in psd. See comments in @calculate_psd #
        fnyq = 0.5/dt
        nois = [ff*0+np.mean(re**2)*len(re)/(fnyq*2*dt) for ff,re in zip(freq, rerr)]
        Nois = [ff*0+np.mean(re**2)*len(re)/(fnyq*2*dt) for ff,re in zip(freq, Rerr)]


        # flattern lists #
        _c = np.concatenate
        freq = _c(freq)
        isort = np.argsort(freq)
        freq = freq[isort]
        crss = _c(crss)[isort]
        rpsd = _c(rpsd)[isort]
        Rpsd = _c(Rpsd)[isort]
        nois = _c(nois)[isort]
        Nois = _c(Nois)[isort]


        # do we need just raw lags? #
        if fqbin is None:
            lag = np.angle(crss) / (1. if phase else 2*np.pi*freq)
            return freq, lag

        
        # bin the lag #
        _a = np.array
        idx = misc.group_array(freq, do_unique=True, **fqbin)
        fqm = _a([len(i) for i in idx])
        fqL = _a([freq[i].min() for i in idx] + [freq[idx[-1].max()]])

        f  = _a([np.mean(freq[i]) for i in idx])
        p  = _a([np.mean(rpsd[i]) for i in idx])
        P  = _a([np.mean(Rpsd[i]) for i in idx])
        n  = _a([np.mean(nois[i]) for i in idx])
        N  = _a([np.mean(Nois[i]) for i in idx])
        c  = _a([np.mean(crss[ii]) for ii in idx])

        # phase lag and its error #
        # g2 is caluclated without noise subtraciton
        # see paragraph after eq. 17 in Nowak+99
        # see eq. 11, 12 in Uttley+14 (is this wrong. Nowak clearly 
        # states that the noise shouldn't be subtracted)
        n2  = ((p - n)*N + (P - N)*n + n*N) / fqm
        lag = np.angle(c)
        g2  = (np.abs(c)**2 - n2) / (p * P)
        dum = (1 - g2) / (2*g2*fqm)
        lag_e = np.sqrt(np.abs(dum))


        # coherence gamma_2 #
        # here we subtract the noise; see eq. 8
        # in Vaughan+97 and related definitions
        coh   = (np.abs(c)**2 - n2) / ((p-n) * (P-N))
        coh[(coh<0) | (coh>1)] = 1e-5
        dcoh  = (2/fqm)**0.5 * (1 - coh)/np.sqrt(coh)
        coh_e = coh * (fqm**-0.5) * ((2*n2*n2*fqm)/(np.abs(c)**2 - n2)**2 + 
                (n**2/(p-n)**2) + (N**2/(P-N)**2) + (fqm*dcoh/coh**2))**0.5
        

        # mask out points where coherence is undefined #
        if mask_coh:
            idx = (dum<0) | (g2<0)
            lag_e[dum<0] = np.pi
            lag[dum<0] = 0.0


        # limits on lag measurements due to poisson noise #
        # equation 30 in Vaughan+2003 #
        limit = np.sqrt(np.abs(n/(fqm * g2 * (p-n))))
        Limit = np.sqrt(np.abs(N/(fqm * g2 * (P-N)))) 


        # do we need time lag instead of phase lag? #
        if not phase:
            lag   /= (2*np.pi*f)
            lag_e /= (2*np.pi*f)
            limit /= (2*np.pi*f)
            Limit /= (2*np.pi*f)


        # return #
        desc = {'fqL': fqL, 'fqm':fqm, 'limit':limit, 'Limit':Limit, 'coh': [coh, coh_e]}

        return f, lag, lag_e, desc




