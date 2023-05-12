"""A module for simulating light curves."""


from typing import Callable, Union

import numpy as np
from numpy.random import RandomState


class SimLC:
    """Class for simulating light curves"""


    def __init__(self, seed=None):
        """Initialize a SimLc instance with some random seed

        Parameters:
        -----------
            seed: int or None
                a seed for the random number generator

        """

        # built-in models #
        self.builtin_models = ['powerlaw', 'broken_powerlaw', 'bending_powerlaw',
                                'step', 'constant', 'lorentz', 'user_array']


        # model containers #
        self.psd_models = []
        self.lag_models = []

        # other variables
        self.normalized_psd = None
        self.tarr = None
        self.xarr = None


        # seed random generator #
        self.rng = RandomState(seed)


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
                 norm: str = 'var'):
        """Simulate a light curve using the psd model stored in 
            self.psd_models

        Parameters
        ----------
            npoints: int
                number of points in the light curve
            deltat: float
                time sampling
            lcmean: float
                the light curve mean
            norm: str
                on of var|rms|leahy

        Note:
            The normalized psd is stored in self.normalized_psd

        """

        # make sure the norm is known #
        if norm not in ['var', 'leahy', 'rms']:
            raise ValueError('norm need to be one of var|rms|leahy')

        # calculate psd #
        freq = np.fft.rfftfreq(npoints, deltat)
        psd = self.calculate_model(freq)
        self.normalized_psd = np.array([freq, psd])


        # get correct renoramlization #
        expon = {'var':0, 'leahy':1, 'rms':2}
        renorm = lcmean**expon[norm] * npoints/(2.*deltat)
        psd *= renorm


        # inverse fft #
        ixarr = ( self.rng.randn(len(psd)) * np.sqrt(0.5*psd) +
                  self.rng.randn(len(psd)) * np.sqrt(0.5*psd)*1j )
        ixarr[0] = npoints * lcmean
        ixarr[-1] = np.abs(ixarr[-1])
        self.xarr = np.fft.irfft(ixarr)
        self.tarr = np.arange(npoints) * deltat * 1.0
