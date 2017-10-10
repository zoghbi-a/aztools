

import numpy as np


def split_array(arr, length, strict=False, **kwargs):
    """Split an array arr to segments of length length.

    Parameters:
        arr: array to be split
        length: (int) desired length of the segments
        strict: force all segments to have length length. Some data 
            may be discarded

    Keywords:
        overlap: (int) number < length of overlap between segments
        split_at_gaps: Split at non-finite values. Default True.


    Returns:
        (result, indx)

    """
    from itertools import groupby


    # defaults #
    split_at_gaps = kwargs.get('split_at_gaps', True)
    overlap = kwargs.get('overlap', 0)


    # make sure overlap makes sense if used #
    if length>0 and overlap >= length:
        raise ValueError('overlap needs to be < length')


    # index array #
    iarr = np.arange(len(arr))

    
    # split at gaps ? #
    if split_at_gaps:
        iarr = [list(i[1]) for i in groupby(iarr, lambda ix:np.isfinite(arr[ix]))]
        iarr = iarr[(0 if np.isfinite(arr[iarr[0][0]]) else 1)::2]
    else:
        iarr = [iarr]


    # split if length>0 #
    if length > 0:

        # split every part of iarr #
        iarr = [[ii[i:i+length] for i in 
                    range(0, len(ii), length-overlap)] for ii in iarr]


        # flatten the list #
        iarr = [j for i in iarr for j in i if not strict or len(j)==length]


    res = [arr[i] for i in iarr]
    return res, iarr


def group_array(arr, by_n=None, bins=None, **kwargs):
    """Group elements of array arr given one of the criterion

    Parameters:
        arr: array to be grouped. Assumed 1d
        by_n: [nstart, nfac] group every 
            [nstart, nstart*nfac, nstart*nfac^2 ...]
        bins: array|list of bin boundaries. Values outside
            these bins are discarded

    Keywords:
        do_unique: if true, the groupping is done for the 
            unique values.

    """

    if by_n is not None:
        
        # check input #
        if (not isinstance(by_n, (list, np.ndarray)) or len(by_n) != 2 
                or by_n[0]<1 or by_n[1]<1):
            raise ValueError('by_n need to be [nstart>=1, fact>=1]')
        else:
            nstart, factor = by_n

        do_unique = kwargs.get('do_unique', True)
        if do_unique:
            arru, arri, arrc = np.unique(
                arr, return_counts=True, return_inverse=True)
        else:
            arru, arri, arrc = (np.array(arr), 
                np.arange(len(arr)), np.ones_like(arr))
        narru, narr = len(arru), len(arr)

        n1, n2 = 0, np.int(nstart)
        nval, idx = n2, []
        while True:
            while arrc[n1:n2].sum() > nval and n2 > (n1+1): n2 -= 1
            if arrc[n1:n2].sum() < nval: n2 += 1 # reverse last step if needed
            idx.append(range(n1, min(n2, narru)))
            n1, nval = n2, nval*factor
            n2 = np.int(nval+n1)
            if n1 >= narru: break
        idxa = np.arange(narr)
        ind = [np.concatenate([idxa[arri==j] for j in i]) for i in idx]

    elif bins is not None:
        
        if not isinstance(bins, (list, np.ndarray)):
            raise ValueError('bins must be a list or array')
        ib = np.digitize(arr, bins) - 1
        ind = [np.argwhere(ib==i)[:,0] for i in range(len(bins)-1)]
    else:
        raise ValueError('No binning defined')

    return ind



