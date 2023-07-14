#!/usr/bin/env python
"""Group pha files using optimal binning based on instrument resolution"""

import argparse
import os

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d


def do_ogrppha(args):
    """core of ogrppha
    
    Parameters
    ----------
    args: return of argparse.ArgumentParser.parse_args
    """

    # process input #
    spec_file   = args.spec_file
    out_file    = args.out_file
    min_snr     = args.snr
    min_counts  = args.counts
    osample_fac = max(args.osample_fac, 1)
    use_formula = args.use_formula
    write_chan  = args.write_chan


    # ------------------------------------------- #
    # Read the response and background file names #
    with fits.open(spec_file) as filep:
        src_exposure = filep['SPECTRUM'].header['EXPOSURE']
        try:
            src_c  = np.array(filep['SPECTRUM'].data.field('COUNTS'), np.double)
        except KeyError:
            src_c  = np.array(filep['SPECTRUM'].data.field('RATE'), np.double) * src_exposure

        try:
            src_bsc = filep['SPECTRUM'].header['BACKSCAL']
        except KeyError:
            src_bsc = 1.0

        # this will fail if there is no RESPFILE
        rsp_file = filep['SPECTRUM'].header['RESPFILE']
        if rsp_file.lower() == 'none':
            raise ValueError(f'Response file "{rsp_file}" does not exist')

        try:
            bgd_file = filep['SPECTRUM'].header['BACKFILE']
        except KeyError:
            bgd_file = None
    # ------------------------------------------- #


    # --------------------- #
    # get background counts #
    bgd_c = np.zeros_like(src_c)
    if not bgd_file in [None, 'none', 'NONE']:
        with fits.open(bgd_file) as filep:
            bgd_exposure = filep['SPECTRUM'].header['EXPOSURE']

            try:
                bgd_c  = np.array(filep['SPECTRUM'].data.field('COUNTS'), np.double)
            except KeyError:
                bgd_c  = np.array(filep['SPECTRUM'].data.field('RATE'), np.double) * bgd_exposure

            try:
                bgd_bsc = filep['SPECTRUM'].header['BACKSCAL']
            except KeyError:
                bgd_bsc = 1.0

        bgd_c *= (src_bsc/bgd_bsc) * (src_exposure/bgd_exposure)
    # incase bgd_c is extreme, cap it
    cap = -20
    iextr = (src_c - bgd_c) < cap
    bgd_c[iextr] = src_c[iextr] - cap
    # --------------------- #


    # ------------------------- #
    # energy-channel conversion #
    with fits.open(rsp_file) as filep:
        edata  = filep['EBOUNDS'].data
        chan   = edata.field(0)
        energy = (edata.field(1)+edata.field(2)) / 2.
        try:
            matrix = filep['MATRIX'].data
        except KeyError:
            matrix = filep['SPECRESP MATRIX'].data
        nchan  = len(chan)
        nen    = len(matrix)
    men = np.array([(mat[1]+mat[0])/2 for mat in matrix])
    # ------------------------- #


    # --------------------- #
    # loop through channels #
    ibin = np.zeros(nchan) - 1
    ich, ibin[0] = 0, 1
    smooth = nen * 1. / nchan
    while ich < nchan:

        # get the response curve at ich #
        ien = np.argmin(np.abs(men - energy[ich]))
        istart, ilen = matrix[ien][3], matrix[ien][4]
        if not isinstance(istart, (list, np.ndarray)):
            istart, ilen = [istart], [ilen]
        if len(istart) == 0 or len(ilen) == 0:
            ich += 1
            continue
        #iarr = np.concatenate([np.arange(i1, i1+i2) for i1,i2 in zip(istart, ilen)])
        rarr = matrix[ien][5]
        rarr = gaussian_filter1d(rarr, smooth)


        # work out fwhm at ich #
        if not np.allclose(rarr, 0) and nchan > 100:
            imax = np.argmax(rarr)

            # limits of fwhm in energy grid units
            ic1 = np.argmin(np.abs(rarr[:imax]-rarr[imax]/2.)) if imax !=0 else 0
            ic2 = np.argmin(np.abs(rarr[imax:]-rarr[imax]/2.)) + imax
            width = ic2 - ic1

        else:
            #width = nchan - ich
            width = 1


        # get oversampling factor if we use_formula is requested #
        if use_formula:
            width = np.max([1, width])
            ind = range(ich, ich+width)
            if ind[-1] >= nchan:
                ind = range(ich, nchan)
            counts = np.max([sum(src_c[ind] - bgd_c[ind]), 1e-10] )
            counts *= 1./width # per resolution element
            xarr = np.log(counts*(1 + 0.20 *np.log(width)))
            osample_fac = 1. if xarr<2.119 else (0.08+7./xarr + 1.8/xarr**2)/(1+5.9/xarr)
            osample_fac = 1. / osample_fac
        # ----- #


        # bin width in channel units using oversample_fac
        width = np.int64(np.round(np.max([1, width*1./osample_fac])))
        ind   = range(ich, np.min([ich+width, nchan]) )


        # increase width until snr > min_snr #
        if min_snr:
            snr = sum(src_c[ind] - bgd_c[ind])
            while snr < 1 and ich+width<=nchan:
                width += 1
                ind = range(ich, min(ich+width, nchan) )
                snr = sum(src_c[ind] - bgd_c[ind])
            if snr > 0:
                snr /= np.sqrt(sum(src_c[ind] + bgd_c[ind]))
            while (snr < min_snr) and (ich+width < nchan):
                ind = range(ich, ich+width)
                if ind[-1] >= nchan:
                    ind = range(ich, nchan)
                    break

                snr = sum(src_c[ind] - bgd_c[ind])
                if snr <= 0:
                    width += 1
                    continue
                snr /= np.sqrt(sum(src_c[ind] + bgd_c[ind]))
                width += 1

        # do we have a min_counts requirement? #
        if min_counts:
            counts = sum(src_c[ind] - bgd_c[ind])
            while (counts < min_counts) and (ich+width < nchan):
                ind = range(ich, ich+width)
                if ind[-1] >= nchan:
                    ind = range(ich, nchan)
                    break
                counts = sum(src_c[ind] - bgd_c[ind])
                width += 1
        ich += len(ind)
        if ich < nchan:
            ibin[ich] = 1
    # --------------------- #


    # ------------------ #
    # write the grouping #
    if write_chan:
        # write an ascii file and use grppha #
        ichan = np.arange(nchan)[ibin==1]
        if ichan[-1] < nchan-1:
            ichan = np.append(ichan, nchan)
        bins = []
        for ich in range(len(ichan)-1):
            bins.append([ichan[ich], ichan[ich+1]-1, ichan[ich+1]-ichan[ich]])
        txt = '\n'.join([f'{xval[0]} {xval[1]} {xval[2]}' for xval in bins])
        with open('tmp_chans.dat', 'w', encoding='utf-8') as filep:
            filep.write(txt)
        if os.path.exists(out_file):
            os.system(f'rm {out_file}')
        os.system(f'grppha {spec_file} {out_file} "group tmp_chans.dat&exit"')
    else:
        # modify the file with pyfits #
        if os.path.exists(out_file):
            os.system(f'rm {out_file}')
        os.system(f'grppha {spec_file} {out_file} "group min 20&exit" &> /dev/null')

        with fits.open(out_file) as filep:
            hdu = filep['SPECTRUM']
            orig_cols = hdu.columns
            orig_cols['GROUPING'].array = np.array(ibin, np.int64)
            cols = fits.ColDefs(orig_cols)
            tbl = fits.BinTableHDU.from_columns(cols)
            hdu.header.update(tbl.header.copy())
            tbl.header = hdu.header.copy()
            grp = fits.HDUList([filep[0],tbl])
            grp.writeto(out_file, overwrite=True)
        print(f'Grouped file {out_file} written sucessfully')
    # ------------------ #


def main():
    """Group pha files using optimal binning based on the instrument resoluions.
    
    We use the formula in Kaastra+Bleeker 16,
    with the addition of a signal to noise constraint.
    The input spectrum is assumed to have the RESPFILE and BACKFILE keywords.
    
    
    """

    parser = argparse.ArgumentParser(description=main.__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("spec_file"  , metavar="spec_file", type=str,
                        help="The name of the input spectrum file.")
    parser.add_argument("out_file"  , metavar="out_file", type=str,
                        help="The name of the output spectrum file.")
    parser.add_argument('-s', '--snr', metavar='min_snr',
                        type=float, default=3,
                        help='The minimum signal to noise ration per bin. -1 to ignore')
    parser.add_argument('-c', '--counts', metavar='min_counts',
                        type=float, default=-1,
                        help='The minimum counts per bin. -1 to ignore')
    parser.add_argument('-f', '--osample_fac', metavar='osample_fac',
                        type=float, default=3,
                        help='The oversampling factor in units of the local FWHM')
    parser.add_argument("--use_formula", action='store_true', default=False,
                        help=('Calculate the oversampling factor using formula 36-37 '
                              'in Kaastra & Bleeker 2016'))
    parser.add_argument("--write_chan", action='store_true', default=False,
                        help=('Write grouping file, then use standard grppha.'
                              'This requires heasoft and ngroups < 100'))

    do_ogrppha(parser.parse_args())


if __name__ == '__main__':
    main()
