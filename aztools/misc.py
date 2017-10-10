

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
    if overlap is not None and overlap >= length:
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




