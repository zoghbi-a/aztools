
import numpy as np


class SimLC(object):
    """Class for simulating light curves"""


    def __init__(self, seed=None):
        """Create a simlc object, setting some defaults

        Parameters:
            seed: a seed for the random number generator

        """

        # built-in models #
        self.builtin_models = ['powerlaw', 'broken_powerlaw', 'bending_powerlaw',
                                'step', 'constant', 'lorentz', 'user_array']


        # model containers #
        self.psd_models = []
        self.lag_models = []


        # seed random generator #
        self.rng = np.random.RandomState(seed)



    def add_model(self, model, params, clear=True, lag=False):
        """Add a model to the generating psd/lag models. Adds the model
            to self.psd_models or self.lag_models depending on lag

        Parameters:
            model: a callable model(freq, params), or a string
                from the builtin models
            params: model parameters
            clear: If true, clear all previously defined models
            lag: if true, this is a lag model, else it is a psd
        """

        
        # check the model is callable or in self.builtin_models #
        if isinstance(model, str):
            if not model in self.builtin_models:
                mtxt = ', '.join([x for x in self.builtin_models])
                raise ValueError('model is not in builtin_models\n' + mtxt)
            else:
                model = eval('SimLC.' + model)
        else:
            if not hasattr(model, '__call__'):
                raise ValueError('model is not callable')


        # add given model to the generating list #
        m_container = self.lag_models if lag else self.psd_models

        if clear:
            for k in m_container: m_container.remove(k)
        m_container.append([model, params])



    def calculate_model(self, freq, lag=False):
        """Calculate the psd/lag model using the models
            added with @add_model.

        Parameters:
            freq: frequency array.
            lag: if true, calculate the lag model, else do the psd

        Returns:
            model: array of the same length as freq, containing the model

        Note:
            the normalization of the returned psd/lag is taken from
            the input model parameters without any renormalization

        """

        # check we have models added #
        models = self.lag_models if lag else self.psd_models
        if len(models) == 0:
            raise ValueError('No models added. Use add_model to do so.')


        # calculate the model #
        freq = np.array(freq, np.float)
        mod = np.zeros_like(freq)
        for m,p in models:
            mod += m(freq, p)

        return mod



    def simulate(self, n, dt, mu, norm='var'):
        """Simulate a light curve using the psd model stored in 
            self.psd_models, added with add_model

        Parameters:
            n: number of points in the light curve
            dt: time sampling
            mu: the light curve mean
            norm: string var|rms|leahy

        Returns:
            Nothing. The simulated light curve and time are stored
            in self.t and self.x

        Note:
            The normalized psd is stored in self.normalized_psd

        """

        # make sure the norm is known #
        if not (norm in ['var', 'leahy', 'rms']):
            raise ValueError('norm need to be one of var|rms|leahy')

        # calculate psd #
        freq = np.fft.rfftfreq(n, dt)
        psd = self.calculate_model(freq)
        self.normalized_psd = np.array([freq, psd])

        
        # get correct renoramlization #
        expon = {'var':0, 'leahy':1, 'rms':2}
        renorm = mu**expon[norm] * n/(2.*dt)
        psd *= renorm



        # inverse fft #
        ix = ( self.rng.randn(len(psd)) * np.sqrt(0.5*psd)
             + self.rng.randn(len(psd)) * np.sqrt(0.5*psd)*1j )
        ix[0] = n * mu
        ix[-1] = np.abs(ix[-1])
        self.x = np.fft.irfft(ix)
        self.t = np.arange(n) * dt * 1.0



    def simulate_pdf(self, n, dt, mu, norm='var', pdf='lognorm(s=0.5)'):
        """Simulate a light curve using the psd model stored in 
            self.psd_models, added with add_model, and enforcing
            log-normal distribution.
            This uses the algorithm of Emmanoulopoulos+ (2013) MNRAS 433, 907–927

        Parameters:
            n: number of points in the light curve
            dt: time sampling
            mu: the light curve mean
            norm: string var|rms|leahy
            pdf: a string representing a probability distribution
                from scipy.stats. e.g. lognorm(s=0.5)

        Returns:
            Nothing. The simulated light curve and time are stored
            in self.t and self.x

        Note:
            The normalized psd is stored in self.normalized_psd

        """

        # make sure the norm is known #
        if not (norm in ['var', 'leahy', 'rms']):
            raise ValueError('norm need to be one of var|rms|leahy')

        import scipy.stats as st
        lognorm = eval('st.{}'.format(pdf))


        # calculate psd #
        freq = np.fft.rfftfreq(n, dt)
        psd = self.calculate_model(freq)
        self.normalized_psd = np.array([freq, psd])

        
        # get correct renoramlization #
        expon = {'var':0, 'leahy':1, 'rms':2}
        renorm = mu**expon[norm] * n/(2.*dt)
        psd *= renorm



        # do the algorithm #
        ix = ( self.rng.randn(len(psd)) * np.sqrt(0.5*psd)
             + self.rng.randn(len(psd)) * np.sqrt(0.5*psd)*1j )
        ix[0] = n * mu
        ix[-1] = np.abs(ix[-1])
        # step-1 #
        anorm  = np.abs(ix)

        # step-2; before loop starts #
        xsim   = lognorm.rvs(n)
        diff   = 1.0
        while diff > 1e-4:

            # step-2; main loop #
            psim   = np.angle(np.fft.rfft(xsim))

            # step-3
            xsim_j = np.fft.irfft(anorm * np.exp(1j*psim))

            # step-4
            xsim_j[np.argsort(xsim_j)] = xsim[np.argsort(xsim)]
            
            # check for convergenece & prepare for next loop #
            diff   = np.sum((xsim - xsim_j)**2)/n
            xsim[:]= xsim_j[:]


        # get the final light curve
        psim   = np.angle(np.fft.rfft(xsim))
        self.x = np.fft.irfft(anorm * np.exp(1j*psim))
        self.t = np.arange(n) * dt * 1.0



    def apply_lag(self, phase=False):
        """Apply lag in self.lag_models to the simulate self.c

        Parameters:
            phase: the lags in self.lag_models are in radians
            if true, else in seconds.

        Returns:
            Nothing. creates self.y for the shifted light curve
            and self.normalized_lag for the actual [freq, lag ] used

        Note:
            The lag vs freq is found by calling the functions in
            self.lag_models, filled by calling add_model(..., lag=True)
            The light curve in self.x is assumed to exist (i.e. self.simulate
            should've been called already)

        """

        # has simulate beed run? #
        try:
            self.x
        except:
            raise ValueError('self.x does not exists. Run simulate first')

        freq = self.normalized_psd[0]
        lag  = self.calculate_model(freq, lag=True)
        self.normalized_lag = np.array([freq, lag])
        self.y = SimLC.lag_array(self.x, lag, phase, freq)




    @staticmethod 
    def lag_array(x, lag, phase=False, freq=None):
        """Shift the x by amount lag
        
        Args:
            x: light curve to be shifted
            lag: float or array of length len(n)/2+1
            phase: if True, lag is in radians. Otherwise in seconds
            freq: the frequency axis (used when phase is False)
            

        Returns:
            An array containing the shifted light curve.

        """
        nfq = len(x)/2 + 1
        if not isinstance(lag, np.ndarray): lag = np.repeat(lag, nfq)
        if nfq != len(lag):
            raise ValueError('Lag array does not match given array')
        
        if freq is not None and nfq != len(freq):
            raise ValueError('freq array does not match given array')
        
        phi = lag if phase else lag*2.*np.pi*freq
        xfft = np.fft.rfft(x)
        xfft[1:-1] *= np.exp(-1j*phi[1:-1])
        return np.fft.irfft(xfft)


    @staticmethod
    def add_noise(x, norm=None, seed=None, dt=1.0):
        """Add noise to lcurve x
    
        Parameters:
            norm: if None, add Poisson noise, else
                gaussian noise, with std=norm
            seed: random seed
            dt: used with norm is None. It gives the time samling
                of the light curve. Poisson noise is applied to
                the counts per bin. x in this case is the count rate.
                Counts/bin = Rate/sec * dt
            

        Returns:
            array similar to x with noise added

        """

        if seed is not None:
            np.random.seed(seed)

        if norm is None:
            xn = np.random.poisson(x*dt)/dt
        else:
            xn = np.random.randn(len(x)) * norm + x
        return xn

    
    @staticmethod
    def user_array(freq, params):
        """The model is given by the user directly as an array params

        Parameters:
            freq (np.ndarray): the frequency array
            params (np.ndarray): the model values

        Returns:
            array mod of same length as freq, containing the psd/lag model
        """

        if len(params) != len(freq):
            raise ValueError('params does not match freq: %d vs %d'%(len(params), len(freq)))
        return np.array(params)


    @staticmethod
    def powerlaw(freq, params):
        """A powerlaw model for the psd/lag

        Parameters:
            freq (np.ndarray): the frequency array
            params (list or array: [norm, indx]): parameters of the model
        
        Returns:
            array mod of same length as freq, containing the psd/lag model

        """

        if len(params) != 2:
            raise ValueError('powerlaw needs 2 params: norm, index')
        norm, indx = params
        freq = np.clip(freq, 1e-30, np.inf)
        mod = norm * freq**indx
        return mod


    @staticmethod
    def broken_powerlaw(freq, params):
        """A borken powerlaw model for the psd/lag.

        Parameters:
            freq (np.ndarray): the frequency array
            params (list or array: [norm, indx1, indx2, break]):
                parameters of the model
        
        Returns:
            array mod of same length as freq, containing the psd/lag model

        """

        if len(params) != 4:
            raise ValueError(('broek_powerlaw needs 4 params: norm, '
                      'index1, index2, break'))
        norm, a1, a2, brk = params
        freq = np.clip(freq, 1e-30, np.inf)
        fq_indx = freq < brk
        mod = np.zeros_like(freq)
        mod[fq_indx] = (freq[fq_indx]/brk) ** a1
        mod[~fq_indx] = (freq[~fq_indx]/brk) ** a2
        mod *= norm * brk**a2
        return mod


    @staticmethod
    def bending_powerlaw(freq, params):
        """A bending powerlaw model for the psd/lag.

        Parameters:
            freq (np.ndarray): the frequency array
            params (list or array: [norm, index, break]):
                parameters of the model. The index below the
                break is always 0
        
        Returns:
            array mod of same length as freq, containing the psd/lag model

        """

        if len(params) != 3:
            raise ValueError(('bending_powerlaw needs 3 params: norm, '
                      'index, break'))
        norm, a, brk = params
        freq = np.clip(freq, 1e-30, np.inf)
        mod = (norm/freq) * (1+(freq/brk)**(-a-1))**(-1)
        return mod


    @staticmethod
    def step(freq, params):
        """A step function model for the psd/lag.

        Parameters:
            freq (np.ndarray): the frequency array
            params: a list (or array) of 2 lists (arrays).
                The first holds the frequency bin boundaries.
                The second holds the values of psd|lag
                len(list_1) = len(list_2) + 1
        
        Returns:
            array mod of same length as freq, containing the psd/lag model

        """

        if len(params) != 2:
            raise ValueError('step needs a list of 2 arrays/lists')
        fqL, pars = params
        fqL, pars = np.array(fqL), np.array(pars)
        if len(fqL) != len(pars)+1:
            raise ValueError('fqL and pars do not match')

        ibins = np.digitize(freq, fqL, right=True) - 1
        ibins[ibins==-1] = 0
        ibins[ibins==len(fqL)-1] = len(fqL)-2
        mod = pars[ibins]
        return mod


    @staticmethod
    def constant(freq, params):
        """A constant model for the psd/lag

        Parameters:
            freq (np.ndarray): the frequency array
            params (float): the value of the constant
        
        Returns:
            array mod of same length as freq, containing the psd model

        """
        mod = freq*0 + params
        return mod


    @staticmethod
    def lorentz(freq, params):
        """A lorentzian model for the psd/lag

        Parameters:
            freq (np.ndarray): the frequency array
            params (list or array: [norm, fq_center, fq_sigma]):
                parameters of the model
        
        Returns:
            array mod of same length as freq, containing the psd/lag model

        """
        if len(params) != 3:
            raise ValueError('lorentz needs 3 pars: norm, fq_cent, fq_sig')
        norm, fq_center, fq_sigma = params
        mod = norm * (fq_sigma/(2*np.pi)) / (
                    (freq-fq_center)**2 + (fq_sigma/2)**2 )
        return mod



