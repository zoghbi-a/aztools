
import numpy as np
import os


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



