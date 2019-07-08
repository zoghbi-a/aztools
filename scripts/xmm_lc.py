#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import subprocess
import argparse
import os
import glob


def run_cmd(cmd):
    """Run cmd command"""
    header = '\n' + '*'*20 + '\n' + cmd + '\n' + '*'*20 + '\n'
    print(header)
    ret = subprocess.call(cmd, shell='True')
    if ret != 0:
       raise SystemExit('\nFailed in the command: ' + header)

if __name__ == '__main__':
    pass
    p   = argparse.ArgumentParser(                                
        description='''
        Extract xmm light curves and correct them using epiclccorr.
        xmm_filter.py is assumed to have been run first to create
        a filtered event file and a region file.
        ''',            
        formatter_class=argparse.ArgumentDefaultsHelpFormatter ) 

    p.add_argument("event_file", metavar="event_file", type=str,
            help="The name of the input event file")
    p.add_argument("region_file", metavar="region_file", type=str,
            help="The name of the region file e.g. from xmm_filter.py")
    p.add_argument('-o', "--out"    , metavar="out", type=str, default='lc',
            help="stem for output fits files.")
    p.add_argument('-e', "--ebins"  , metavar="ebins", type=str, default='0.3 10',
            help="A space separated list of energy limits in keV")
    p.add_argument('-t', "--tbin"  , metavar="tbin", type=float, default=256,
            help="The time bin, negative means 2**tbin")
    p.add_argument("--gti"  , metavar="gti", type=str, default='',
            help="A user gti file to use.")
    p.add_argument("--raw", action='store_true', default=False,
            help="region is in RAWX, RAWY instead of X, Y")
    p.add_argument("--chans", action='store_true', default=False,
            help="ebins are give in pha channels; don't run lccorr")
    args = p.parse_args()

    # ----------- #
    # parse input #
    tbin = args.tbin
    if tbin<0: tbin = 2**tbin
    dumst = [x for x in np.array(args.ebins.split()) if len(x)>0]
    if args.chans:
        ebins = np.array(dumst, np.int)
    else:
        ebins = np.array(dumst, np.double) * 1000
    nbins = len(ebins)-1
    ebins = [[ebins[i], ebins[i+1]] for i in range(nbins)]
    out = '{}_{:03g}__{{}}'.format(args.out, tbin)
    event = args.event_file
    usr_gti = args.gti
    if usr_gti != '':
        usr_gti = '&&gti({},TIME)'.format(usr_gti)
    # ----------- #



    # ------------------ #
    # check system setup #
    sas_ccf = os.environ.get('SAS_CCF', None)
    if sas_ccf is None:
        cwd = os.getcwd() + '/'
        for d in ['.', '..', '../..', '../../odf']:
            if os.path.exists(cwd + d + '/ccf.cif'):
                os.environ['SAS_CCF'] = cwd + d + '/ccf.cif'
                break
    sas_ccf = os.environ.get('SAS_CCF', None) 
    if sas_ccf is None:
        raise RuntimeError('I cannot find ccf.cif, please define SAS_CCF')
    # ------------------ #

    
    # ------------------------ #
    # read regions information #
    selector = '(RAWX,RAWY) IN ' if args.raw else '(X,Y) IN '
    regions = ['', '']
    with open(args.region_file) as fp:
        for line in fp.readlines():
            if '(' in line:
                idx = np.int('back' in line)
                reg = selector + line.split('#')[0]
                regions[idx] += '' if regions[idx] == '' else ' || '
                regions[idx] += selector + line.split('#')[0].rstrip()
    # ------------------------ #


    # --------------------------- #
    # looping through energy bins #
    CMD1 = (
        'evselect table={}:EVENTS withrateset=yes rateset={} '
        'expression="{}" makeratecolumn=yes maketimecolumn=yes timebinsize={}'
    )
    CMD2 = (
        'epiclccorr srctslist={} eventlist={} outset={} '
        'withbkgset=yes bkgtslist={} applyabsolutecorrections=no'
    )

    for ie in range(nbins):

        # names #
        src = out.format('{}__s.fits'.format(ie+1))
        bgd = out.format('{}__b.fits'.format(ie+1))
        smb = out.format('{}.fits'.format(ie+1))

        # source #
        eselect = 'PHA' if args.chans else 'PI'
        expr = '{} IN [{}:{}] && {} {}'.format(
                eselect, ebins[ie][0], ebins[ie][1], regions[0], usr_gti)
        cmd = CMD1.format(event, src, expr, tbin)
        run_cmd(cmd)

        # background #
        expr = '{} IN [{}:{}] && {} {}'.format(
                eselect, ebins[ie][0], ebins[ie][1], regions[1], usr_gti)
        cmd = CMD1.format(event, bgd, expr, tbin)
        run_cmd(cmd)

        # correction #
        if args.chans:
            # get backscale from the spectra in ../../spec
            import astropy.io.fits as pyfits
            spec_dir = '../../spec'
            sfile = glob.glob('%s/spec*pha'%spec_dir)[0]
            bfile = glob.glob('%s/spec*bgd'%spec_dir)[0]
            s_backscale = pyfits.open(sfile)['spectrum'].header['backscal']
            b_backscale = pyfits.open(bfile)['spectrum'].header['backscal']
            cmd = 'lcmath %s %s %s 1.0 %g addsubr=no'%(src, bgd, smb, s_backscale/b_backscale)
            run_cmd(cmd)
        else:
            cmd = CMD2.format(src, event, smb, bgd)
            run_cmd(cmd)



    # --------------------------- #


