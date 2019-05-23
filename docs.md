<h1 id="aztools">aztools</h1>


<h2 id="aztools.LCurve">LCurve</h2>

```python
LCurve(self, t, r, re, dt=None, fexp=None)
```
Light curve holder class
<h3 id="aztools.LCurve.make_even">make_even</h3>

```python
LCurve.make_even(self, fill=nan)
```
Make the light curve even in time, filling gaps with fill

Parameters:
    fill: value to use in gaps.

Returns:
    a new LCurve object


<h3 id="aztools.LCurve.rebin">rebin</h3>

```python
LCurve.rebin(self, factor, error='norm', min_exp=0.0)
```
Rebin the light curve to so new_dt = dt*factor

Parameters:
    factor: rebinning factor. dt_new = factor * dt
    error: error type (poiss|norm).
        If poiss: rerr = sqrt(rate*dt)/dt, otherwise,
        errors are summed quadratically
    min_exp: minimum fractional exposure to leave [0-1]

Return:
    new binned LCurve


<h3 id="aztools.LCurve.interp_small_gaps">interp_small_gaps</h3>

```python
LCurve.interp_small_gaps(self, maxgap=None, noise='poiss', seed=None)
```
Interpolate small gaps in the lightcurve if the gap
is <maxgap; applying noise if requested

Parameters:
maxgap: the maximum length of a gap to be interpolated
noise: poiss|norm|None
seed: random seen if noise is requested


<h3 id="aztools.LCurve.read_fits_file">read_fits_file</h3>

```python
LCurve.read_fits_file(fits_file, **kwargs)
```
Read LCurve from fits file

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

<h3 id="aztools.LCurve.read_pn_lcurve">read_pn_lcurve</h3>

```python
LCurve.read_pn_lcurve(fits_file, **kwargs)
```
Read pn lcurve fits_file created with xmmlc_lc.
This sets values relevant to PN and calls read_fits_file

Parameters:
fits_file: name of the files file

Keywords:
See @LCurve.read_fits_file


Returns:
LCurve object

<h3 id="aztools.LCurve.read_pca_lcurve">read_pca_lcurve</h3>

```python
LCurve.read_pca_lcurve(fits_file, **kwargs)
```
Read pca lcurve fits_file.
This sets values relevant to PCA and calls read_fits_file

Parameters:
fits_file: name of the files file

Keywords:
See @LCurve.read_fits_file


Returns:
LCurve object

<h3 id="aztools.LCurve.read_nu_lcurve">read_nu_lcurve</h3>

```python
LCurve.read_nu_lcurve(fits_file, **kwargs)
```
Read nustar lcurve fits_file.
This sets values relevant to NUSTAR and calls read_fits_file

Parameters:
fits_file: name of the files file

Keywords:
See @LCurve.read_fits_file


Returns:
LCurve object

<h3 id="aztools.LCurve.calculate_psd">calculate_psd</h3>

```python
LCurve.calculate_psd(rate, dt, norm='var', **kwargs)
```
Calculate raw psd from a list of light curves.

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

<h3 id="aztools.LCurve.bin_psd">bin_psd</h3>

```python
LCurve.bin_psd(freq, rpsd, fqbin, noise=None, logavg=True)
```
Bin power spectrum.

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


<h3 id="aztools.LCurve.calculate_lag">calculate_lag</h3>

```python
LCurve.calculate_lag(rate, Rate, dt, fqbin=None, **kwargs)
```
Calculate and bin lags from two lists of light curves.

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
    taper: apply Hanning tapering before calculating the fft
        see p388 Bendat & Piersol; the fft need to be multiplied
        by sqrt(8/3) to componsate for the reduced variance. Default: False



Return:
    freq, lag, lage, desc;
    desc = {'fqm', 'fqL', 'limit', 'Limit'}

<h2 id="aztools.SimLC">SimLC</h2>

```python
SimLC(self, seed=None)
```
Class for simulating light curves
<h3 id="aztools.SimLC.add_model">add_model</h3>

```python
SimLC.add_model(self, model, params, clear=True, lag=False)
```
Add a model to the generating psd/lag models. Adds the model
to self.psd_models or self.lag_models depending on lag

Parameters:
model: a callable model(freq, params), or a string
    from the builtin models
params: model parameters
clear: If true, clear all previously defined models
lag: if true, this is a lag model, else it is a psd

<h3 id="aztools.SimLC.calculate_model">calculate_model</h3>

```python
SimLC.calculate_model(self, freq, lag=False)
```
Calculate the psd/lag model using the models
added with @add_model.

Parameters:
freq: frequency array.
lag: if true, calculate the lag model, else do the psd

Returns:
model: array of the same length as freq, containing the model

Note:
the normalization of the returned psd/lag is taken from
the input model parameters without any renormalization


<h3 id="aztools.SimLC.simulate">simulate</h3>

```python
SimLC.simulate(self, n, dt, mu, norm='var')
```
Simulate a light curve using the psd model stored in
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


<h3 id="aztools.SimLC.simulate_pdf">simulate_pdf</h3>

```python
SimLC.simulate_pdf(self, n, dt, mu, norm='var', pdf='lognorm(s=0.5)')
```
Simulate a light curve using the psd model stored in
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


<h3 id="aztools.SimLC.apply_lag">apply_lag</h3>

```python
SimLC.apply_lag(self, phase=False)
```
Apply lag in self.lag_models to the simulate self.c

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


<h3 id="aztools.SimLC.lag_array">lag_array</h3>

```python
SimLC.lag_array(x, lag, phase=False, freq=None)
```
Shift the x by amount lag

Args:
    x: light curve to be shifted
    lag: float or array of length len(n)/2+1
    phase: if True, lag is in radians. Otherwise in seconds
    freq: the frequency axis (used when phase is False)


Returns:
    An array containing the shifted light curve.


<h3 id="aztools.SimLC.add_noise">add_noise</h3>

```python
SimLC.add_noise(x, norm=None, seed=None, dt=1.0)
```
Add noise to lcurve x

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


<h3 id="aztools.SimLC.user_array">user_array</h3>

```python
SimLC.user_array(freq, params)
```
The model is given by the user directly as an array params

Parameters:
    freq (np.ndarray): the frequency array
    params (np.ndarray): the model values

Returns:
    array mod of same length as freq, containing the psd/lag model

<h3 id="aztools.SimLC.powerlaw">powerlaw</h3>

```python
SimLC.powerlaw(freq, params)
```
A powerlaw model for the psd/lag

Parameters:
    freq (np.ndarray): the frequency array
    params (list or array: [norm, indx]): parameters of the model

Returns:
    array mod of same length as freq, containing the psd/lag model


<h3 id="aztools.SimLC.broken_powerlaw">broken_powerlaw</h3>

```python
SimLC.broken_powerlaw(freq, params)
```
A borken powerlaw model for the psd/lag.

Parameters:
    freq (np.ndarray): the frequency array
    params (list or array: [norm, indx1, indx2, break]):
        parameters of the model

Returns:
    array mod of same length as freq, containing the psd/lag model


<h3 id="aztools.SimLC.bending_powerlaw">bending_powerlaw</h3>

```python
SimLC.bending_powerlaw(freq, params)
```
A bending powerlaw model for the psd/lag.

Parameters:
    freq (np.ndarray): the frequency array
    params (list or array: [norm, index, break]):
        parameters of the model. The index below the
        break is always 0

Returns:
    array mod of same length as freq, containing the psd/lag model


<h3 id="aztools.SimLC.step">step</h3>

```python
SimLC.step(freq, params)
```
A step function model for the psd/lag.

Parameters:
    freq (np.ndarray): the frequency array
    params: a list (or array) of 2 lists (arrays).
        The first holds the frequency bin boundaries.
        The second holds the values of psd|lag
        len(list_1) = len(list_2) + 1

Returns:
    array mod of same length as freq, containing the psd/lag model


<h3 id="aztools.SimLC.constant">constant</h3>

```python
SimLC.constant(freq, params)
```
A constant model for the psd/lag

Parameters:
    freq (np.ndarray): the frequency array
    params (float): the value of the constant

Returns:
    array mod of same length as freq, containing the psd model


<h3 id="aztools.SimLC.lorentz">lorentz</h3>

```python
SimLC.lorentz(freq, params)
```
A lorentzian model for the psd/lag

Parameters:
    freq (np.ndarray): the frequency array
    params (list or array: [norm, fq_center, fq_sigma]):
        parameters of the model

Returns:
    array mod of same length as freq, containing the psd/lag model


<h2 id="aztools.misc">aztools.misc</h2>


<h3 id="aztools.misc.split_array">split_array</h3>

```python
split_array(arr, length, strict=False, *args, **kwargs)
```
Split an array arr to segments of length length.

Parameters:
    arr: array to be split
    length: (int) desired length of the segments
    strict: force all segments to have length length. Some data
        may be discarded

Args:
    any other arrays similar to arr to be split

Keywords:
    overlap: (int) number < length of overlap between segments
    split_at_gaps: Split at non-finite values. Default True.
    min_seg_length: (int) minimum seg length to keep. Use when strict=False
    approx: length is used as an approximation.

Returns:
    (result, indx, ...{other arrays if given})


<h3 id="aztools.misc.group_array">group_array</h3>

```python
group_array(arr, by_n=None, bins=None, **kwargs)
```
Group elements of array arr given one of the criterion

Parameters:
    arr: array to be grouped. Assumed 1d
    by_n: [nstart, nfac] group every
        [nstart, nstart*nfac, nstart*nfac^2 ...]
    bins: array|list of bin boundaries. Values outside
        these bins are discarded

Keywords:
    do_unique: if true, the groupping is done for the
        unique values.
    min_per_bin: minimum number of elements per bin


<h3 id="aztools.misc.write_2d_veusz">write_2d_veusz</h3>

```python
write_2d_veusz(fname, arr, xcent=None, ycent=None, append=False)
```
Write a 2d array to a file for veusz viewing

Parameters:
    fname: name of file to write
    arr: array to write, shape (len(xcent), len(ycent))
    xcent, ycent: central points of axes.
    append: append to file? Default=False

<h3 id="aztools.misc.spec_2_ebins">spec_2_ebins</h3>

```python
spec_2_ebins(spec_file, nbins=1, **kwargs)
```
Find nbins energy bins from spec_file so
that the bins have roughly the same number of
counts per bin, equal snr per bin, and are
eually-separated in log-space

Parameters:
spec_file: spectrum file. background will be
    read from the header. If not defined, assume
    it is 0.
nbins: how many bins to extract.

Keywords:
ebound: [emin, emax] of the limiting energies where to
    do the calculations. e.g. [3., 79] for nustar
efile: name of output file. Default energy.dat
    --> {file}, {file}.snr, {file}.log


Write results to {efile}, {efile}.snr, {feile}.log


<h3 id="aztools.misc.write_pha_spec">write_pha_spec</h3>

```python
write_pha_spec(b1, b2, arr, err, stem)
```
Write spectra to pha spectrum format and call
flx2xsp to create a pha file

Parameters:
b1: lower boundaries of bins
b2: upper boundaries of bins
arr: array of `spectral` data
err: measurement error for arr
stem: stem name for the output spectra



<h3 id="aztools.misc.print_progress">print_progress</h3>

```python
print_progress(step, total, final=False)
```
Print progress in one line

Args:
    step: current step; assumed to start at 0
    total: total number of steps
    final: is this the final step


