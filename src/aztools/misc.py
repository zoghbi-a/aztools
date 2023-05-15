"""Miscellaneous Utilities"""

from itertools import groupby
from typing import Union

import numpy as np

try:
    import heasoftpy as hsp
except ImportError:
    hsp = None


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


def set_fancy_plot(plt):
    """Some settings for plt that make nicer plots
    
    Parameters
    ----------
    plt: matplotlib.pyplot
    
    """

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
