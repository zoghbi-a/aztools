"""Miscellaneous Utilities"""

import functools
import glob
import os
import subprocess
from itertools import groupby
from multiprocessing import Pool, cpu_count
from typing import Union

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


import numpy as np
from astropy.io import fits

try:
    import heasoftpy as hsp
except ImportError:
    hsp = None

__all__ = [
    'split_array', 'group_array', 'write_pha_spec', 'write_2d_veusz',
    'set_fancy_plot', 'sync_lcurve', 'lcurve_to_segments', 'read_fits_lcurve',
    'run_cmd_line_tool', 'add_spectra', 'parallelize'
]

def split_array(arr: np.ndarray,
                length: int,
                *args,
                strict: bool = False,
                **kwargs):
    """Split an array arr to segments of length length.

    Parameters
    ----------
    arr: np.ndarray
        The array to be split.
    length: int
        The desired length of the segments
    strict: bool
        If True, force all segments to have length `length`. 
        Some data may be discarded

    Arguments
    ---------
        any other arrays similar to `arr` to be split

    Keywords
    --------
    overlap: int 
        number < length of overlap between segments.
    split_at_gaps: bool
        Split at non-finite values. Default True.
    min_seg_length: int.
        Minimum seg length to keep. Used when strict=False
    approx: int
        length is used as an approximation.

    Returns
    -------
    A tuple with: (result, indx, ...{other arrays if given})

    """


    # get keywords #
    split_at_gaps = kwargs.get('split_at_gaps', True)
    overlap = kwargs.get('overlap', 0)
    min_seg_length = kwargs.get('min_seg_length', 0)
    approx  = kwargs.get('approx', False)
    if strict:
        approx = False


    # make sure overlap makes sense if used #
    if overlap >= length > 0:
        raise ValueError('overlap needs to be < length')


    # index array #
    iarr = np.arange(len(arr))

    # split at gaps ? #
    if split_at_gaps:
        iarr = [list(idx[1]) for idx in groupby(iarr, lambda idxx:np.isfinite(arr[idxx]))]
        iarr = iarr[(0 if np.isfinite(arr[iarr[0][0]]) else 1)::2]
    else:
        iarr = [iarr]


    # split if length>0
    if length > 0:

        # can we use approximate length? #
        if approx:
            ilength = [
                length if len(iidx)<=length else
                np.int64(np.round(len(iidx)/np.round(len(iidx) / length))) for iidx in iarr
            ]
        else:
            ilength = [length] * len(iarr)


        # split every part of iarr #
        iarr = [
            [iidx[idx:idx+idxl] for idx in
            range(0, len(iidx), idxl-overlap)] for iidx,idxl in zip(iarr, ilength)
        ]

        # flatten the list #
        iarr = [jdx for idx in iarr for jdx in idx if not strict or len(jdx)==length]

    # enforce a minimum segment length
    iarr = [idx for idx in iarr if len(idx)>=min_seg_length]


    res = [arr[idx] for idx in iarr]
    others = [[arr[idx] for idx in iarr] for arr in args]
    return (res, iarr) + tuple(others)


def group_array(arr: np.ndarray,
                by_n: Union[np.ndarray, list] = None,
                bins: Union[np.ndarray, list] = None,
                **kwargs):
    """Group elements of array arr given one of the criterion

    Parameters
    ----------
    arr: np.ndarray
        The array to be grouped. Assumed 1d.
    by_n: np.ndarray or list
        [nstart, nfac] creates a new groupd at 
        [nstart, nstart*nfac, nstart*nfac^2 ...]
    bins: np.ndarray or list
        A list of bin boundaries. Values outside 
        these bins are discarded

    Keywords
    --------
    do_unique: bool
        if True, the groupping is done for the 
        unique values.
    min_per_bin: int
        The minimum number of elements per bin.

    """

    if by_n is None and bins is None:
        raise ValueError('No binning defined')


    if by_n is not None:

        # check input #
        if (
            not isinstance(by_n, (list, np.ndarray))
            or len(by_n) != 2
            or by_n[0]<1 or by_n[1]<1
        ):
            raise ValueError('by_n needs to be [nstart>=1, nfac>=1]')


        if kwargs.get('do_unique', True):
            # arru, arri, arrc
            arruic = np.unique(
                arr, return_counts=True, return_inverse=True)
        else:
            # arru, arri, arrc
            arruic = (np.array(arr),
                      np.arange(len(arr)), np.ones_like(arr))
        narru, narr = len(arruic[0]), len(arr)

        num = [0, np.int64(by_n[0])]
        nval, idx = num[1], []
        while True:
            while arruic[2][num[0]:num[1]].sum() > nval and num[1] > (num[0]+1):
                num[1] -= 1
            num[1] += np.int64(arruic[2][num[0]:num[1]].sum() < nval)
            idx.append(range(num[0], min(num[1], narru)))
            num[0], nval = num[1], nval*by_n[1]
            num[1] = np.int64(nval+num[0])
            if num[0] >= narru:
                break
        ind = [np.concatenate([np.arange(narr)[arruic[1]==jdx] for jdx in iidx])
               for iidx in idx]

    else: # use bins

        if not isinstance(bins, (list, np.ndarray)):
            raise ValueError('bins must be a list or array')
        idx = np.digitize(arr, bins) - 1
        ind = [np.argwhere(idx==jdx)[:,0] for jdx in range(len(bins)-1)]


    # enforce minimum per bin if requesed #
    min_per_bin = kwargs.get('min_per_bin', 1)
    new_ind, grp = [], [[]]
    for idx,_ in enumerate(ind):
        grp.append(ind[idx])
        cgrp = np.concatenate(grp)
        if len(cgrp) >= min_per_bin:
            new_ind.append(np.array(cgrp, dtype=np.int64))
            grp = [[]]

    return new_ind


def write_pha_spec(bin1: Union[list, np.ndarray],
                   bin2: Union[list, np.ndarray],
                   arr: np.ndarray,
                   err: np.ndarray,
                   stem: str,
                   **kwargs):
    """Write some data array to pha file so it can be
    used inside xspec

    Parameters
    ----------
    bin1: list of np.ndarray.
        The lower boundaries of bins
    bin2: list of np.ndarray.
        The upper boundaries of bins
    arr: np.ndarray
        The array of data values to be written.
    err: np.ndarray
        The array of measurement error corresponding to arr.

    Keywords
    --------
    stem: str:
        Stem name for the output spectra -> {stem}.pha|rsp


    """
    if hsp is None:
        raise ImportError('write_pha_spec depends on heasoftpy. Install it first')

    verbose = kwargs.get('verbose', True)

    if not len(bin1) == len(bin2) == len(arr) == len(err):
        raise ValueError('bin1, bin2, arr, err need to have the same length')

    delb = np.array(bin2) - np.array(bin1)
    txt = '\n'.join([(f'{_bin1:8.8} {bin2[idx]:8.8} '
                      f'{arr[idx]*delb[idx]:8.8} {err[idx]*delb[idx]:8.8}')
                     for idx,_bin1 in enumerate(bin1)])
    with open(f'{stem}.xsp', 'w', encoding='utf8') as filep:
        filep.write(f'{txt}\n')

    # pylint: disable=no-member
    out = hsp.flx2xsp(infile=f'{stem}.xsp', phafile=f'{stem}.pha',
                      rspfile=f'{stem}.rsp')
    if out.returncode != 0:
        raise RuntimeError(out.output)
    if verbose:
        print(f'{stem}.pha was created successfully')


def write_2d_veusz(fname: str,
                   arr: np.ndarray,
                   xcent: np.ndarray = None,
                   ycent: np.ndarray = None,
                   append: bool = False):
    """Write a 2d array to a file for veusz viewing
    
    Parameters
    ----------
    fname: str
        The name of file to write.
    arr: np.ndarray
        The array to write, its shape is (len(xcent), len(ycent))
    xcent: np.ndarray:
        Central points of X-axis.
    ycent: np.ndarray:
        Central points of Y-axis.
    append: bool
        Append to file? Default=False

    """
    if xcent is None:
        xcent = np.arange(len(arr[0]))
    if ycent is None:
        ycent = np.arange(len(arr))

    if arr.shape == (len(xcent),len(ycent)):
        raise ValueError('arr.shape does not match xcent,ycent')

    thead = '\n\n'
    assert(arr.shape==(len(xcent),len(ycent)))
    thead += f"xcent {' '.join([f'{xcen}' for xcen in xcent])}\n"
    thead += f"ycent {' '.join([f'{ycen}' for ycen in ycent])}\n"

    arr = arr.T
    txt2d = '\n'.join([' '.join([f'{arr[idx,jdx]:3.3e}'
                                 for jdx,_ in enumerate(arr[0])])
                      for idx,_ in enumerate(arr)])
    with open(fname, 'a' if append else 'w', encoding='utf8') as filep:
        filep.write(f'{thead}{txt2d}')


def set_fancy_plot():
    """Some settings for plt that make nicer plots
    
    """
    if plt is None:
        raise ImportError('matplotlib is not available. Cannot use set_fancy_plot')

    plt.rcParams.update({
        'font.size': 14, 
        'font.family': 'serif',

        'lines.linewidth': 1,
        'lines.markersize': 8.0,
        'figure.subplot.wspace': 0.,
        'axes.linewidth': 0.5,
        'axes.formatter.use_mathtext': True,

        'axes.edgecolor': '#111',
        'axes.facecolor': '#fafafa',


        'axes.xmargin': 0.1,
        'xtick.direction': 'in',
        'xtick.major.size': 9.,
        'xtick.major.pad': 5.,
        'xtick.minor.size': 4.,
        'xtick.top': True,
        'xtick.minor.visible': True,
        'xtick.major.width': 0.5,
        'xtick.minor.width': 0.5,

        'axes.ymargin': 0.1,
        'ytick.direction': 'in',
        'ytick.major.size': 9,
        'ytick.major.pad': 5.,
        'ytick.minor.size': 4,
        'ytick.right': True,
        'ytick.minor.visible': True,
        'ytick.major.width': 0.5,
        'ytick.minor.width': 0.5,
    })


def sync_lcurve(lc_list: Union[list, np.ndarray],
                tbase: np.ndarray = None):
    """Synchronize a list of arrays or LCurves
    
    Parameters
    ----------
    lc_list: list
        A list of arrays or a list of LCurve objects.
        if arrays, the shape is (nlcurve, 3 (or 4 with fexp), ntime).
        The 3 is for (time, rate, rerr)
    tbase: np.ndarray
        The time array to use for reference. 
        If not given, use the intersection of all time arrays
        
    Return
    ------
    a list of sync'ed arrays/light curves
    """

    if not isinstance(lc_list, (list, np.ndarray)):
        raise ValueError('lc_list must be a list')

    if hasattr(lc_list[0], 'deltat'): # LCurve
        data = [np.array([lcrv.time, lcrv.rate, lcrv.rerr])
                for lcrv in lc_list]
    else:
        data = [np.array(l) for l in lc_list]

    if tbase is None:
        tbase = data[0][0]
        for dat in data[1:]:
            tbase = tbase[np.in1d(tbase, dat[0])]

    data = [dat[:, np.in1d(dat[0], tbase)] for dat in data]
    return data


def lcurve_to_segments(lcurves,
                       seglen: float,
                       strict: bool = False,
                       **kwargs):
    """Split an LCurve or a list of them to segments. 
    Useful to be used with calculate_psd|lag etc.


    Parameters
    ----------
    lcurves: LCurve
        an LCurve or a list of them
    seglen: float
        segment length in seconds.
    strict: bool
        force all segments to have length length. Some data 
        may be discarded

    Keywords
    --------
    uneven: The light curves are uneven, so the splitting produces 
        segments that have the same number of points. Default: False
    **other arguments to be passed to split_array

    Returns
    -------
    rate, rerr, time, seg_idx
    seg_idx is the indices used to create the segments.

    """
    # Keywords
    uneven = kwargs.get('uneven', False)


    if not isinstance(lcurves, list):
        lcurves = [lcurves]

    # assert the same sampling #
    deltat = lcurves[0].deltat
    for lcrv in lcurves:
        if deltat != lcrv.deltat:
            raise ValueError('deltat of the input LCurve do not match')

    # segments details #
    iseglen = np.int64(seglen/deltat)

    # make sure the LCurve objects are evenly sampled #
    if not uneven:
        lcurves = [lcrv.make_even() for lcrv in lcurves]


    # split the rate arrays #
    splt = [split_array(lcrv.rate, iseglen, lcrv.rerr,
                        lcrv.time, strict=strict, **kwargs) for lcrv in lcurves]

    # flatten the segments into on large list #
    rate = [idx for spt in splt for idx in spt[0]]
    rerr = [idx for spt in splt for idx in spt[2]]
    time = [idx for spt in splt for idx in spt[3]]
    seg_idx = [spt[1] for spt in splt]
    return rate, rerr, time, seg_idx


def read_fits_lcurve(fits_file: str, **kwargs):
    """Read a light cuurve from fits file

    Parameters
    ----------
    fits_file: str
        Name of the fits file.

    Keywords
    --------
    min_exp: float
        minimum fractional exposure to allow. Default 0.0 for all
    rate_tbl: str or int
        Name or number of hdu that contains the light curve data. Default: RATE
    rate_col: str or int
        Name or number of rate column. Default: RATE
    time_col: str or int 
        Name or number of time column. Default: TIME
    rerr_col: str or int 
        Name or number of rerr column. Default: ERROR
    fexp_col: str or int 
        Name or number of the fracexp column. Default: FRACEXP
    gti_table: str or int
        Name or number of gti extension hdu. Default: GTI 
    dt_key: str or int
        Name of time sampling keyword in header. Default: TIMEDEL
    gti_skip: float
        How many seconds to skip at the gti boundaries. Default: 0
    verbose: bool
        Print progress


    Returns
    -------
        lcurve_data (shape: 4,nt containing, time, rate, rerr, fexp), deltat

    """

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
        raise ValueError(f'file {fits_file} does not exist')

    # read file #
    with fits.open(fits_file) as filep:

        # lc data #
        data = filep[rate_tbl].data
        ldata = np.array([  data.field(time_col),
                            data.field(rate_col),
                            data.field(rerr_col)], dtype=np.double)


        # start time and time sampling #
        tstart = filep[rate_tbl].header.get('TSTART', 0.0)
        time_0 = filep[rate_tbl].header.get('timezero', 0.0)
        deltat = filep[rate_tbl].header.get(dt_key, None)
        if deltat is not None:
            tstart += deltat/2
        if time_0 != 0 and tstart/time_0 < 1e5:
            time_0 = 0.0

        # if the time-axis offset, correct it #
        if tstart/ldata[0, 1] > 1e5:
            ldata[0] += tstart + time_0

        # gti #
        try:
            lgti  = np.array([filep[gti_tbl].data.field(0),
                              filep[gti_tbl].data.field(1)], dtype=np.double)
        except KeyError:
            if verbose:
                print(f'No GTI found in {fits_file}')
            lgti = np.array([[ldata[0, 0]], [ldata[0, -1]]])


        # fractional exposure #
        try:
            lfracexp = data.field(fexp_col)
        except KeyError:
            if verbose:
                print(f'cannot read fracexp_col in {fits_file}')
            lfracexp = np.ones_like(ldata[0])


        # apply gti #
        igti  = ldata[0] < 0
        for gstart, gstop in lgti.T:
            igti = igti | ( (ldata[0] >= (gstart+gti_skip)) &
                            (ldata[0] <= (gstop -gti_skip)) )
        igood = igti & (lfracexp >= min_exp) & (np.isfinite(ldata[0]))
        ldata = np.vstack([ldata, lfracexp])
        ldata = ldata[:, igood]

    return ldata, deltat


def run_cmd_line_tool(cmd: str,
                      env: dict = None,
                      allow_fail: bool = True,
                      logfile: str = None):
    """Run a command line tool
    
    Parameters
    ----------
    cmd: str
        The command strign to be run where the parameters are in the string
    env: dict
        Dictionary of environment variables to be used by the task
    allow_fail: bool
        If True and the task fails, return without raising an exception
    logfile: str
        File name to long output/error to. If None, no logs are produced
    
    """

    run_env = os.environ.copy()
    if env is not None:
        run_env.update(**env)

    output = ''
    try:

        with subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True,
                              stderr=subprocess.PIPE, env=run_env) as proc:
            output, error = proc.communicate()


        # Decode the output and error messages
        output = output.decode().strip()
        error = error.decode().strip()
        returncode = proc.returncode

        if returncode != 0:
            raise RuntimeError(error)

    except Exception as exception: # pylint: disable=broad-exception-caught
        if allow_fail:
            raise exception

        error  = str(exception)
        returncode = -1

    full_output = output
    if error != '':
        full_output += f'\n\nERROR:\n{error}'
    if logfile is not None:
        with open(logfile, 'w', encoding='utf8') as filep:
            filep.write(full_output)

    result = {
        'output': output,
        'error': error,
        'returncode': returncode,
        'full_output': full_output
    }
    return result


def add_spectra(speclist: list, outfile: str, **kwargs):
    """Call addspec to combine spectra.
    
    do it multiple times if num of spectra above kwargs['nmax']
    
    Parameters
    ----------
    speclist: list
        list of the names of the spectral files to be added
    outfile: str
        output name root
    
    Keywords:
    ---------
    nmax: int
        add a maximum of nmax spectra at a time
    
    other parameters for addspec (qaddrmf, qsubback, clobber)
    
    """
    if hsp is None:
        raise ImportError('write_pha_spec depends on heasoftpy. Install it first')

    nmax = kwargs.get('nmax', 10)
    nspec = len(speclist)
    nbatch = nspec//nmax

    if nbatch > 1:
        batches = [speclist[i:i+nbatch] for i in range(0, nspec, nbatch)]
        # combine spectra in the batches
        spec = [add_spectra(slist, f'{outfile}_{idx}', **kwargs)
                for idx,slist in enumerate(batches)]

        # combine the batches to produce one file
        out = add_spectra(spec, outfile, **kwargs)
        os.system(f'rm -rf {" ".join(spec)} '
                  f'{" ".join([s.replace("pha", "rsp") for s in spec])}')
        return out

    with open('tmp.add', 'w', encoding='utf8') as filep:
        filep.write('\n'.join(speclist))
    if len(glob.glob(f'{outfile}.???')) != 0:
        os.system(f'rm {outfile}.???')
    qaddrmf = kwargs.get('qaddrmf', 'no')
    qsubback = kwargs.get('qsubback', 'no')
    clobber = kwargs.get('clobber', 'yes')
    out = hsp.addspec( # pylint: disable=no-member
        infil='tmp.add', outfil=outfile,
        qaddrmf=qaddrmf, qsubback=qsubback, clobber=clobber)

    if out.returncode != 0:
        logfile = 'add_spectra.log'
        with open(logfile, 'w', encoding='utf8') as filep:
            filep.write(str(out))
        raise RuntimeError(f'ERROR in addspec; Writing log to {logfile}')

    return f'{outfile}.pha'


def parallelize(func, use_irun=True):
    """A wrapper to make a function run in parallel
    
    Parameters
    ----------
    func: method
        The method to parallelize. It can any args or kwargs.
    use_irun: bool
        If True, pass a keyword argument irun as int to func
        that holds the call number in the sequence of parallel
        calls. See description of the returned function below
        

    Return
    ------
    return a method with parameters that are lists of args/kwargs
    to be passed to func. The returned method takes these special kwargs:
    """

    @functools.wraps(func)
    def _parallelize(*args, **kwargs):

        # remove special keywords:
        irun = kwargs.pop('irun', None)
        nproc = kwargs.pop('nproc', cpu_count())

        # check that arguments are lists of the same length
        arg0 = None
        fargs = {str(arg):arg for arg in args}
        fargs.update(**kwargs)
        for key,arg in fargs.items():
            if not isinstance(arg, list):
                raise ValueError(f'{key} is expected to be a list')

            if arg0 is not None and len(arg) != len(arg0):
                raise ValueError('Expected lists of the same length: '
                                 f'len({arg})!= len({arg0})')
            arg0 = arg
        ntasks = len(arg0)
        nproc = min(nproc, ntasks)

        if use_irun:
            if irun is None:
                irun = [1+it for it in range(ntasks)]
            elif isinstance(irun, int):
                irun = [irun+it for it in range(ntasks)]
            elif isinstance(irun, (list, np.ndarray)):
                if len(irun) != ntasks:
                    raise ValueError(f'Expected irun length of {ntasks}')
            else:
                raise ValueError(f'{irun} should be either an int or a list '
                                 'of length similar to number of calls')
            kwargs['irun'] = irun

        with Pool(nproc) as pool:
            procs = [
                pool.apply_async(
                    func,
                    args=[arg[it] for arg in args],
                    kwds={key:val[it] for key,val in kwargs.items()}
                )
                for it in range(ntasks)
            ]
            results = [proc.get() for proc in procs]

        return results
    # Update the docstring
    _parallelize.__doc__ = (func.__doc__ or '') + '\nExtra keywords:' + _P_EXTRA_DOC

    return _parallelize


_P_EXTRA_DOC = """
    - irun: None, int or list to control the call number:
        - None: generate a sequence [1, ntasks]
        - int n: generate a sequence [n, n+ntasks]
        - list of int: use as a sequence. It has to have
        the correct length
    - nproc: Number of parallel processes to use.
"""
parallelize.__doc__ += _P_EXTRA_DOC
