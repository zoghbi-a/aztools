"""A module for simulating light curves."""


from typing import Callable, Union

import numpy as np
import scipy.stats as st
from numpy import random

__all__ = ['SimLC']

class SimLC:
    """Class for simulating light curves"""


    def __init__(self, seed=None):
        """Initialize a SimLc instance with some random seed

        Parameters:
        -----------
        seed: int or None
            a seed for the random number generator

        """

        # built-in models and the number of parameter for each one #
        self.builtin_models = {
            'powerlaw': 2,
            'broken_powerlaw': 4,
            'bending_powerlaw': 3,
            'lorentz': 3,
            'constant': 1,
            'user_array': None,
        }


        # model containers #
        self.psd_models = []
        self.lag_models = []

        # other variables
        self.normalized_psd = None
        self.normalized_lag = None
        self.lcurve = None


        # seed random generator #
        self.rng = random.RandomState(seed)


    def add_model(self,
                  model: Union[str, Callable],
                  params: Union[list, np.ndarray],
                  clear: bool = True,
                  lag: bool = False ):
        """Add a model to be used for generating the psd or lag. 
            
        This adds the model to self.psd_models or self.lag_models (if lag=True)

        Parameters:
        -----------
        model: str or Callable 
            a callable method with model(freq, params), or a string
            from the builtin models
        params: list or np.ndarray
            model parameters
        clear: bool
            If True, clear all previously defined models
        lag: bool
            if True, this is a lag model, else it is a psd
        """


        # check the model is callable or in self.builtin_models #
        if isinstance(model, str):

            if model in self.builtin_models:
                if model != 'user_array' and len(params) != self.builtin_models[model]:
                    raise ValueError(f'{model} expects {self.builtin_models[model]} parameter')

                model = getattr(SimLC, model)
            else:
                builtin = ', '.join(self.builtin_models)
                raise ValueError(f'{model} not found in builtin-models: {builtin}')

        else:

            if not isinstance(model, Callable):
                raise ValueError(f'{model} is not callable')


        # add given model to the generating list #
        m_container = self.lag_models if lag else self.psd_models

        if clear:
            m_container.clear()

        m_container.append([model, params])


    def calculate_model(self,
                        freq: Union[list, np.ndarray],
                        lag: bool = False ) -> np.ndarray:
        """Calculate the psd/lag model using the models
            added with @add_model.

        Parameters
        ----------
        freq: np.ndarray
            frequency array.
        lag: bool
            if True, calculate the lag model, else do the psd

        Returns
        -------
        model: np.ndarray
            array of the same length as freq, containing the model

        Note:
            the normalization of the returned psd/lag is taken from
            the input model parameters without any renormalization

        """

        # check we have models added #
        models = self.lag_models if lag else self.psd_models
        if len(models) == 0:
            raise ValueError('No models added. Add models with add_model')


        # calculate the model #
        freq = np.array(freq, np.double)
        total_mod = np.zeros_like(freq)
        for mod,par in models:
            total_mod += mod(freq, par)

        return total_mod


    def simulate(self,
                 npoints: int,
                 deltat: float,
                 lcmean: float,
                 **kwargs):
        """Simulate a light curve using the psd model stored in 
            self.psd_models
            
        The normalized psd is stored in self.normalized_psd

        Parameters
        ----------
        npoints: int
            number of points in the light curve
        deltat: float
            time sampling
        lcmean: float
            the light curve mean
            
        Keywords
        --------
        norm: str
            on of var|rms|leahy

        """
        # get keywords
        norm = kwargs.get('norm', 'var')

        # make sure the norm is known #
        if norm not in ['var', 'leahy', 'rms']:
            raise ValueError('norm need to be one of var|rms|leahy')

        # calculate psd #
        freq = np.fft.rfftfreq(npoints, deltat)
        psd = self.calculate_model(freq)
        self.normalized_psd = np.array([freq, psd])


        # get correct renoramlization #
        expon = {'var':0, 'leahy':1, 'rms':2}
        psd *= lcmean**expon[norm] * npoints/(2.*deltat)


        # inverse fft #
        ixarr = ( self.rng.randn(len(psd)) * np.sqrt(0.5*psd) +
                  self.rng.randn(len(psd)) * np.sqrt(0.5*psd)*1j )
        ixarr[0] = npoints * lcmean
        ixarr[-1] = np.abs(ixarr[-1])
        self.lcurve = [
            np.arange(npoints) * deltat * 1.0,
            np.fft.irfft(ixarr)
        ]


    def simulate_pdf(self,
                     npoints: int,
                     deltat: float,
                     lcmean: float,
                     **kwargs):
        """Simulate a light curve using the psd model enforcing 
        some probability distribution
        
        This uses the algorithm of Emmanoulopoulos+ (2013) MNRAS 433, 907–927
            
        The normalized psd is stored in self.normalized_psd

        Parameters
        ----------
        npoints: int
            number of points in the light curve
        deltat: float
            time sampling
        lcmean: float
            the light curve mean
            
        Keywords
        --------
        norm: str
            on of var|rms|leahy
        pdf: scipy.stats._distn_infrastructure.rv_frozen
            The probability distribution object from scipy.stats that defines
            the desired pdf. e.g. scipy.stats.lognorm(s=0.5)

        """
        # get keywords
        norm = kwargs.get('norm')
        pdf  = kwargs.get('pdf', st.lognorm(s=0.5))

        # make sure the norm is known #
        if norm not in ['var', 'leahy', 'rms']:
            raise ValueError('norm need to be one of var|rms|leahy')

        if not hasattr(pdf, 'rvs'):
            raise ValueError((
                'pdf needs to be an instance of '
                'scipy.stats._distn_infrastructure.rv_frozen'
            ))

        # calculate psd #
        freq = np.fft.rfftfreq(npoints, deltat)
        psd = self.calculate_model(freq)
        self.normalized_psd = np.array([freq, psd])


        # get correct renoramlization #
        expon = {'var':0, 'leahy':1, 'rms':2}
        psd *= lcmean**expon[norm] * npoints/(2.*deltat)


        # do the algorithm #
        ixarr = ( self.rng.randn(len(psd)) * np.sqrt(0.5*psd) +
                  self.rng.randn(len(psd)) * np.sqrt(0.5*psd)*1j )
        ixarr[0] = npoints * lcmean
        ixarr[-1] = np.abs(ixarr[-1])

        # step-1 #
        anorm  = np.abs(ixarr)

        # step-2; before loop starts #
        xsim = pdf.rvs(npoints)
        diff = 1.0
        while diff > 1e-4:

            # step-2 and step-3
            xsim_j = np.fft.irfft(anorm * np.exp(1j*np.angle(np.fft.rfft(xsim))))

            # step-4
            xsim_j[np.argsort(xsim_j)] = xsim[np.argsort(xsim)]

            # check for convergenece & prepare for next loop #
            diff   = np.sum((xsim - xsim_j)**2)/npoints
            xsim[:]= xsim_j[:]

        # final inverse fft #
        self.lcurve = [
            np.arange(npoints) * deltat * 1.0,
            np.fft.irfft(anorm * np.exp(1j*np.angle(np.fft.rfft(xsim))))
        ]


    def apply_lag(self, phase: bool = False):
        """Apply lag in self.lag_models to the simulated light curve

        - This assumes self.simulate has been run.
        - Lag models are added by add_model(..., lag=True)
        - The resulting lag model is stored in self.normalized_lag
        - The generated lagged light curve is added self.lcurve


        Parameters
        ----------
        phase: bool
            If Trye, the lags in self.lag_models are in radians,
            otherwise they are in seconds.

        """

        # has simulate beed run? #
        if self.lcurve is None:
            raise ValueError('self.lcurve does not exists. Call simulate(...) first')

        freq = self.normalized_psd[0]
        lag  = self.calculate_model(freq, lag=True)
        self.normalized_lag = np.array([freq, lag])

        xarr = self.lcurve[1]
        yarr = SimLC.lag_array(xarr, lag, phase, freq)
        if len(self.lcurve) == 3:
            del self.lcurve[-1]
        self.lcurve.append(yarr)


    @staticmethod
    def lag_array(xarr: np.ndarray,
                  lag: Union[np.ndarray, float],
                  phase: bool = False,
                  freq: Union[np.ndarray, None] = None ) -> np.ndarray:
        """Shift the array xarr by a lag
        
        Parameters
        ----------
        xarr: np.ndarray
            light curve array to be shifted
        lag: np.ndarray or float
            If array, it has length len(xarr)/2+1. If float,
            the lag is assumed constant with frequency
        phase: bool
            if True, lag is in radians. Otherwise in seconds
        freq: np.ndarray or None
            The frequency axis (used to convert lag to radians when phase is False)
            

        Returns
        -------
        An array containing the shifted light curve.

        """

        # xarr has to be even length
        if len(xarr) % 2 == 1:
            raise ValueError('xarr has to be even in length')

        nfreq = len(xarr)/2 + 1

        if not isinstance(lag, np.ndarray):
            lag = np.repeat(lag, nfreq)

        if nfreq != len(lag):
            raise ValueError(f'lag array does not match given xarr. Expecting {nfreq}')

        if freq is not None and nfreq != len(freq):
            raise ValueError(f'freq array does not match given xarr. Expecting {nfreq}')

        phi = lag if phase else lag*2.*np.pi*freq
        xfft = np.fft.rfft(xarr)
        xfft[1:-1] *= np.exp(-1j*phi[1:-1])
        return np.fft.irfft(xfft)


    @staticmethod
    def add_noise(xarr: np.ndarray,
                  norm: Union[float, None] = None,
                  seed: Union[int, None] = None,
                  deltat: float = 1.0):
        """Add noise to a lcurve xarr

        Parameters
        ----------
        xarray: np.ndarray
            Input light curve array
        norm: float or None
            if None, add Poisson noise, else gaussian noise
            with std=norm.
        seed: int or None
            Random seed
        deltat: float
            used when norm is None. It gives the time samling
            of the light curve. Poisson noise is applied to
            the counts per bin. xarr in this case is the count rate.
            Counts/bin = Rate/sec * dt
            

        Return
        ------
        An array similar to xarr with noise added

        """

        if seed is not None:
            np.random.seed(seed)

        if norm is None:
            xnoise = np.random.poisson(xarr*deltat)/deltat
        else:
            xnoise = np.random.randn(len(xarr)) * norm + xarr
        return xnoise


    @staticmethod
    def user_array(freq: np.ndarray,
                   params: np.ndarray) -> np.ndarray:
        """Generate a model to be used as psd or lag.
        
        The model is given by the user directly as an array params

        Parameters
        ----------
        freq: np.ndarray
            The frequency array
        params: np.ndarray
            The model values

        Return
        ------
            An array of the model values at freq, containing the psd/lag model
        """

        if len(params) != len(freq):
            raise ValueError(f'params does not match freq: {len(params)} vs {len(freq)}')
        return np.array(params)


    @staticmethod
    def powerlaw(freq: np.ndarray,
                 params: Union[np.ndarray, list]) -> np.ndarray:
        """Generate a powerlaw model to be used as psd or lag.


        Parameters
        ----------
        freq: np.ndarray
            The frequency array
        params: np.ndarray or list
            Parameters of the model as: [norm, indx]

        Return
        ------
            An array of the model values at freq, containing the psd/lag model

        """

        if len(params) != 2:
            raise ValueError('powerlaw needs 2 params: norm, index')
        norm, indx = params
        freq = np.clip(freq, 1e-30, np.inf)
        mod = norm * freq**indx
        return mod


    @staticmethod
    def broken_powerlaw(freq: np.ndarray,
                        params: Union[np.ndarray, list]) -> np.ndarray:
        """Generate a broken powerlaw model to be used as psd or lag.


        Parameters
        ----------
        freq: np.ndarray
            The frequency array
        params: np.ndarray or list
            Parameters of the model as: [norm, indx1, indx2, break]

        Return
        ------
            An array of the model values at freq, containing the psd/lag model

        """

        if len(params) != 4:
            raise ValueError((
                'broek_powerlaw needs 4 params: norm, '
                'index1, index2, break'
            ))
        norm, idx1, idx2, brk = params
        freq = np.clip(freq, 1e-30, np.inf)
        fq_indx = freq < brk
        mod = np.zeros_like(freq)
        mod[fq_indx] = (freq[fq_indx]/brk) ** idx1
        mod[~fq_indx] = (freq[~fq_indx]/brk) ** idx2
        mod *= norm * brk**idx2
        return mod


    @staticmethod
    def bending_powerlaw(freq: np.ndarray,
                         params: Union[np.ndarray, list]) -> np.ndarray:
        """Generate a bending powerlaw model to be used as psd or lag.


        Parameters
        ----------
        freq: np.ndarray
            The frequency array
        params: np.ndarray or list
            Parameters of the model as: [norm, index, break]). 
            The index below the break is always 0.

        Return
        ------
            An array of the model values at freq, containing the psd/lag model

        """

        if len(params) != 3:
            raise ValueError((
                'bending_powerlaw needs 3 params: norm, '
                'index, break'
            ))
        norm, idx, brk = params
        freq = np.clip(freq, 1e-30, np.inf)
        mod = (norm/freq) * (1+(freq/brk)**(-idx-1))**(-1)
        return mod


    @staticmethod
    def lorentz(freq: np.ndarray,
                params: Union[np.ndarray, list]) -> np.ndarray:
        """Generate a lorentz model to be used as psd or lag.


        Parameters
        ----------
        freq: np.ndarray
            The frequency array
        params: np.ndarray or list
            Parameters of the model as: [norm, fq_center, fq_sigma]

        Return
        ------
            An array of the model values at freq, containing the psd/lag model

        """

        if len(params) != 3:
            raise ValueError('lorentz needs 3 pars: norm, fq_cent, fq_sig')
        norm, fq_center, fq_sigma = params
        mod = norm * (fq_sigma/(2*np.pi)) / (
                     (freq-fq_center)**2 + (fq_sigma/2)**2 )
        return mod


    @staticmethod
    def constant(freq: np.ndarray,
                 params: Union[np.ndarray, list]) -> np.ndarray:
        """Generate a constant model to be used as psd or lag.


        Parameters
        ----------
        freq: np.ndarray
            The frequency array
        params: np.ndarray or list
            Parameters of the model as [value]

        Return
        ------
            An array of the model values at freq, containing the psd/lag model

        """
        if len(params) != 1:
            raise ValueError('constant needs 1 parameter')
        mod = freq*0 + params[0]
        return mod
