# aztools

`aztools` is a collection of python tools that I have been using over the years in the analysis of X-ray data.

## The aztools package
The `aztools` folder contains the main python package, which contains three parts:
- `lcurve.py` defines a light curve object (`LCurve`), which a holder of the observed time series, and associated functions that manipulate it.
- `sim.py` defines the `SimLC` class for simulating light curves.
- `misc.py` contains many additional tools for doing different things. See the file for more details
- `data.py` contains many methods to process x-ray data from different missions and extract spectra and light curves

Most of the functionality in `aztools.data`, and some `aztools.misc` depend on having `heasoftpy` installed (the high energy software package from NASA Goddard https://heasarc.gsfc.nasa.gov/docs/software/heasoft/).


## Scripts
The `src/scripts` folder contains useful `python` and `bash` scripts. If installing `aztools >= 0.2`, these scripts will be automatically installed when doing `pip install`. 


## The `simulations` folder
This contains codes for checking the functionality of the `aztools` package. It focuses on the statistics testing rather than the code tests.

## `tests` folder:
Simple unit tests for the `aztools` package


# Installation
- With `pip install aztools`

The following python packages are needed to run `aztools`:
- `numpy`
- `scipy`
- `astropy`

and `heasoftpy` if using `aztools.data`

# Documentation
See [full documentation at readthedoc](https://aztools.readthedocs.io/).

