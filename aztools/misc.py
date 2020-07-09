

import numpy as np
import os
from itertools import groupby
import sys
import scipy.optimize as opt
import scipy.stats as st


def split_array(arr, length, strict=False, *args, **kwargs):
    """Split an array arr to segments of length length.

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

    """


    # defaults #
    split_at_gaps = kwargs.get('split_at_gaps', True)
    overlap = kwargs.get('overlap', 0)
    min_seg_length = kwargs.get('min_seg_length', 0)
    approx  = kwargs.get('approx', False)
    if strict:
        approx = False


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

        # can we use approximate length? #
        if approx:
            ilength = [length if len(ii)<=length else 
                        np.int(np.round(len(ii)/np.round(len(ii) / length))) for ii in iarr]
        else:
            ilength = [length] * len(iarr)


        # split every part of iarr #
        iarr = [[ii[i:i+il] for i in 
                    range(0, len(ii), il-overlap)] for ii,il in zip(iarr, ilength)]


        # flatten the list #
        iarr = [j for i in iarr for j in i if not strict or len(j)==length]

    # enforce a minimum segment length #  
    iarr = [i for i in iarr if len(i)>=min_seg_length]


    res = [arr[i] for i in iarr]
    others = tuple([[ar[i] for i in iarr] for ar in args])
    return (res, iarr) + others


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
        min_per_bin: minimum number of elements per bin

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

    # enforce minimum per bin if requesed #
    min_per_bin = kwargs.get('min_per_bin', 1)
    new_ind, grp = [], [[]]
    for ib in range(len(ind)):
        grp.append(ind[ib])
        cgrp = np.concatenate(grp)
        if len(cgrp) >= min_per_bin:
            new_ind.append(np.array(cgrp, dtype=np.int))
            grp = [[]]
    ind = new_ind

    return ind


def write_2d_veusz(fname, arr, xcent=None, ycent=None, append=False):
    """Write a 2d array to a file for veusz viewing
    
    Parameters:
        fname: name of file to write
        arr: array to write, shape (len(xcent), len(ycent))
        xcent, ycent: central points of axes.
        append: append to file? Default=False
    """

    thead = '\n\n'
    if xcent is not None and ycent is not None:
        assert( arr.shape==(len(xcent),len(ycent)))
        thead += (  'xcent ' + 
                    ' '.join(['{}'.format(x) for x in xcent]) + '\n')
        thead += (  'ycent ' +
                    ' '.join(['{}'.format(x) for x in ycent]) + '\n')
    arr = arr.T
    txt2d = '\n'.join([
            '{}'.format(' '.join(['{:3.3e}'.format(arr[i, j])
                        for j in range(len(arr[0]))]))
                for i in range(len(arr))
        ])
    with open( fname, 'a' if append else 'w' ) as fp:
        fp.write(thead+txt2d)


def spec_2_ebins(spec_file, nbins=1, **kwargs):
    """Find nbins energy bins from spec_file so
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

    """
    import astropy.io.fits as pyfits

    ebound = kwargs.get('ebound', [2., 10.])
    efile = kwargs.get('efile', 'energy.dat')


    # do we have a file? #
    if not os.path.exists(spec_file):
        raise ValueError('Cannot find file {}'.format(spec_file))

    
    # try reading the background file #
    with pyfits.open(spec_file) as fp:
        try:
            basedir = '/'.join(spec_file.split('/')[:-1]) or '.'
            bgd_file = '{}/{}'.format(basedir, fp[1].header['BACKFILE'])
            print('Background File {} ...'.format(bgd_file))
        except KeyError:
            bgd_file = None
            print('There is no background file. Assuming 0.')

    # try reading the response file #
    with pyfits.open(spec_file) as fp:
        try:
            basedir = '/'.join(spec_file.split('/')[:-1]) or '.'
            rsp_file = '{}/{}'.format(basedir, fp[1].header['RESPFILE'])
            print('Response File {} ...'.format(rsp_file))
        except KeyError:
            raise ValueError('There is no response file. Quitting ...')

    # read counts #
    with pyfits.open(spec_file) as fs:
        cs = fs['SPECTRUM'].data['COUNTS']
        if bgd_file is not None:
            s_scale = fs['SPECTRUM'].header['BACKSCAL']
    if bgd_file is not None:
        with pyfits.open(bgd_file) as fs:
            cb = fs['SPECTRUM'].data['COUNTS']
            b_scale = fs['SPECTRUM'].header['BACKSCAL']
    counts = cs if bgd_file is None else cs - cb*s_scale/b_scale


    # Read response for en to channel conversion #    
    with pyfits.open(rsp_file) as fp:
        Ch = fp['EBOUNDS'].data.field(0)
        Emin = fp['EBOUNDS'].data.field(1)
        Emax = fp['EBOUNDS'].data.field(2)

    i_useful = range(np.argmin(np.abs(Emin-ebound[0])), 
                     np.argmin(np.abs(Emax-ebound[1])) + 1)
    c_useful = counts[i_useful]

    ############################
    # for equal counts per bin #
    c_per_bin = c_useful.sum()*1./nbins
    csum = np.cumsum(c_useful)
    cfac = np.arange(nbins+1) * c_per_bin
    cbin = [np.argmin(np.abs(csum-c)) for c in cfac]
    cbin = np.array(cbin) + i_useful[0]
    with open(efile, 'w') as fp:
        fp.write('\n'.join(['{} {}'.format(
            cbin[i], 0.5*(Emin[cbin[i]]+Emax[cbin[i]])) 
                for i in range(nbins+1)]))
    print('Results written to {}'.format(efile))


    ##############
    # bin by snr #
    c_sum = c_useful.sum()
    nc = len(c_useful)
    snr_per_bin = (c_sum/(nbins+1)) / np.sqrt(c_sum/(nbins+1))
    sbin, snr, i1, i2 = [0], 0, 0, 0
    ss = []
    while i1<nc:
        while snr <= snr_per_bin:
            i2 += 1
            if i2 >= nc-1: break
            dum = c_useful[i1:i2].sum()
            if dum < 0: continue
            snr = dum/np.sqrt(dum)
        sbin.append(i2)
        i1 = i2
        ss.append(snr)
        snr = 0
    sbin = np.array(sbin[:-1]) + i_useful[0]
    with open(efile + '.snr', 'w') as fp:
        fp.write('\n'.join(['{} {}'.format(
            sbin[i], 0.5*(Emin[cbin[i]]+Emax[cbin[i]])) 
            for i in range(nbins+1)]))
    print('Results written to {}'.format(efile + '.snr'))


    ####################
    # bin in log space #
    lebin = np.logspace(np.log10(ebound[0]), np.log10(ebound[1]), nbins+1)
    # re-snap to the detector channels.
    lbin = list(map(lambda en:np.argmin(np.abs(Emin-en)), lebin))

    with open(efile + '.log', 'w') as fp:
        fp.write('\n'.join(['{} {}'.format(
            lbin[i], 0.5*(Emin[cbin[i]]+Emax[cbin[i]])) 
            for i in range(nbins+1)]))
    print('Results written to {}'.format(efile + '.log'))


def write_pha_spec(b1, b2 ,arr ,err ,stem, verbosity=True):
    """Write spectra to pha spectrum format and call 
        flx2xsp to create a pha file

    Parameters:
        b1: lower boundaries of bins
        b2: upper boundaries of bins
        arr: array of `spectral` data
        err: measurement error for arr
        stem: stem name for the output spectra


    """
    if not (len(b1) == len(b2) == len(arr) == len(err)):
        raise ValueError('b1, b2, arr, err need to have the same length')

    de = b2 - b1
    txt = '\n'.join(['{:8.8} {:8.8} {:8.8} {:8.8}'.format(b1[i], b2[i], 
            arr[i]*de[i], err[i]*de[i]) for i in range(len(b1))]) + '\n'
    with open('{}.xsp'.format(stem), 'w') as fp: fp.write(txt)

    cmd = 'export HEADASNOQUERY=;export HEADASPROMPT=/dev/null;'
    cmd += 'flx2xsp {0}.xsp {0}.pha {0}.rsp'.format(stem)
    os.system(cmd)
    if verbosity: print('{}.pha was created successfully'.format(stem))


def print_progress(step, total, final=False):
    """Print progress in one line
    
    Args:
        step: current step; assumed to start at 0
        total: total number of steps
        final: is this the final step

    """

    step += 1
    curr_idx = np.int( step * 50 / total)
    text = '-'*curr_idx + ' '*(49-curr_idx) + '|'
    txt = ( '|{:02g} %| '.format(np.int(100.*step/total)) + text + '\r')
    sys.stdout.write(txt)
    sys.stdout.flush()
    if final:
        sys.stdout.write('\n\n\n')
        

def simple_fit(xval, yval, yerr, mod, **kwargs):
    """Fit a simple constant or linear model to some data, and do the null test

    Parameters:
        xval: values of the independent variable
        yval: values of the dependent variable
        yerr: uncertainty in yval
        mod: a string for a model: 'const' | 'linear'

    Keywords
        verbose: print progress
        p0: starting parameters

    Returns:


    """
    verbose = kwargs.get('verbose', True)
    p0 = kwargs.get('p0', None)


    # built-in models #
    def fcn_1(x, *p) : return x*0 + p[0]
    def fcn_2(x, *p) : return p[0] + p[1] * x
    models = {
        'const':  [1, fcn_1],
        'linear': [2, fcn_2]
    }

    if not mod in models.keys():
        raise ValueError('Unknown model %s'%mod)
    npar = models[mod][0]
    fcn  = models[mod][1]


    # starting parameters #
    if p0 is None:
        p0 = [np.mean(yval)] + ([] if npar==1 else [0.1])
    assert(len(p0) == npar)


    fopt = opt.curve_fit(fcn, xval, yval, p0, sigma=yerr)
    chi2 = np.sum(((yval - fcn(xval, *fopt[0]))/yerr)**2)
    pval = 1 - st.chi2.cdf(chi2, df=len(xval)-npar)
    nsig = -st.norm.ppf(pval/2)
    p, pe = fopt[0], np.diag(fopt[1])**0.5 
    text  = '\n# fit({}): chi2({:6.3}) pval({:6.4}) conf({:6.3}) sigma({:6.3})'.format(
                mod, chi2, pval, 1-pval, nsig)
    text += '\n# ' + (', '.join(['%6.3g (%6.3g)'%(x,xe) for x,xe in zip(p, pe)])) + '\n'
    if verbose:
        print(text)

    return [p, pe], chi2, pval, nsig, text
