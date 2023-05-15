"""Miscellaneous Utilities"""

from itertools import groupby
from typing import Union

import numpy as np


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
