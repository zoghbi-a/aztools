`aztools` is a collection of python tools that I have been using over the years in the analysis of X-ray telescope data.

## The aztools package
The `aztools` folder contains the main python package, which contains three parts:
- `lcurve.py` defines a light curve object (`LCurve`), which a holder of the observed time series, and associated functions that manipulate it.
- `sim.py` defines the `SimLC` class for simulating light curves.
- `misc.py` contains many additional tools for doing different things. See the file for more details


## The `scirpts` folder
This folder contains many `python` and `bash` scripts for processing X-ray data, extracing spectra and light curves. Most of the functionality here depend on having `heasoft` installed (the high energy software package from NASA Goddard https://heasarc.gsfc.nasa.gov/docs/software/heasoft/).


## The `simulations` folder
This contains codes for checking the functionality of the `aztools` package. It focuses on the statistics testing rather than the code tests.

## `tests` folder:
Simple unit tests for the `aztools` packge


# Installation
- With `pip install aztools`

OR

- Copy this to a place where python can find it. This can be done by adding the folder conatinting the `aztools` package to `PYTHONPATH`.

The following python packages are needed to run `aztools`:
- `numpy`
- `scipy`
- `astropy`

# Documentation
The functions are documented individually. They are summarized in the [doc file]

# Examples
To get started, see some usage examples on [simulating light curves and calculating psd].


[doc file]: docs
[simulating light curves and calculating psd]: examples/example1
