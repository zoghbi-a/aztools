#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import subprocess
import argparse
import glob
import os
from astropy.io import fits as pyfits


def run_cmd(cmd, quiet=False):
    """Run cmd command"""
    header = '\n' + '*'*20 + '\n' + cmd + '\n' + '*'*20 + '\n'
    print(header)
    ret = subprocess.call(cmd, shell='True')
    if ret != 0 and not quiet:
       raise SystemExit('\nFailed in the command: ' + header)

if __name__ == '__main__':
    
    p   = argparse.ArgumentParser(                                
        description='''
        Extract nustar light curves and correct them (source and back).
        Assumes heasoft stuff can be run. Event files are in rootdir/event_cl.
        Also assumes current directory has src.reg and bgd.reg files. If not 
        present, try to get them from ../spec, otherwise fail, unless the option
        --create_region is selected
        ''',            
        formatter_class=argparse.ArgumentDefaultsHelpFormatter ) 


    p.add_argument('-o', "--out"    , metavar="out", type=str, default='lc',
            help="stem for output fits files.")
    p.add_argument('-e', "--ebins"  , metavar="ebins", type=str, default='3 79',
            help="A space separated list of energy limits in keV")
    p.add_argument('-t', "--tbin"  , metavar="tbin", type=float, default=256,
            help="The time bin, negative means 2**tbin")
    p.add_argument("--gti"  , metavar="gti", type=str, default='',
            help="A user gti file to use.")
    p.add_argument("--rootdir"  , metavar="rootdir", type=str, default='../',
            help="the root directory that contains event_cl")
    p.add_argument("--bary", action='store_true', default=False,
            help="Apply barycenter corrections")
    p.add_argument("--gti_bary", action='store_true', default=False,
            help="Apply barycenter corrections to user gti")
    p.add_argument("--create_region", action='store_true', default=False,
            help="Create region files if they don't exist")
    args = p.parse_args()

    # ----------- #
    # parse input #
    tbin = args.tbin
    if tbin<0: tbin = 2**tbin
    dumst = [x for x in np.array(args.ebins.split()) if len(x)>0]
    ebins = np.array(dumst, np.double)
    nbins = len(ebins)-1
    out = '{}_{:03g}__{{}}'.format(args.out, tbin)
    usr_gti = args.gti
    if usr_gti != '':
        usr_gti = ' usrgtifile="{}"'.format(usr_gti)
    instr = ['a', 'b']
    INSTR = ['FPMA', 'FPMB']
    # ----------- #


    # ---- #
    # dirs #
    idir = args.rootdir + '/event_cl'
    odir = '.'
    evnt = glob.glob('{}/*A01*evt'.format(idir))
    if len(evnt) == 0:
        raise IOError("I couldn't find the events files. use --rootdir")
    evnt = evnt[0]
    stem = evnt.split('/')[-1][:-10]
    # ---- #

    
    # ----------------------------- #
    # create region files if needed #
    cwd = os.getcwd() + '/'
    for d in ['.', '..', '../spec', '../../spec']:
        if os.path.exists(cwd + d + '/src.reg'):
            if d != '.': run_cmd('cp {}/*reg .'.format(cwd+d))
            print('\nregions found in {}\n'.format(d))
            break
    if not os.path.exists('src.reg') and not args.create_region:
        raise IOError(
            'I cannot find region files, copy them here or use --create_region')
    if args.create_region:
        xsel = (
            'tmp\nread event {} ./\nyes\nextract image\n'
            'save image tmp.img\nexit\nno\n').format(evnt)
        proc = subprocess.Popen("xselect", stdout=subprocess.PIPE, 
                    stdin=subprocess.PIPE)
        proc.communicate(xsel.encode())
        proc.wait()
        run_cmd('ds9 tmp.img -log -zoom 2 -cmap heat')
        run_cmd('rm tmp.* xselect.log &> /dev/null')
    # ----------------------------- #



    # ---------------------------------- #
    # convert energies to channel number #
    conv  = lambda en:int(np.floor((en-1.6)/0.04))
    chans = [ [conv(ebins[i]), conv(ebins[i+1])-1] for i in range(nbins) ]
    enegs = [ [ebins[i],ebins[i+1]] for i in range(nbins) ]
    np.savez('energy_{:03g}.npz'.format(tbin), en=enegs, chans=chans)
    # ---------------------------------- #



    # ------------------------- #
    # do we need barycorrection #
    bary_txt = ''
    gti_bary = 'yes' if args.gti_bary else 'no'
    with pyfits.open(evnt) as fp:
        ra  = fp['EVENTS'].header['RA_OBJ']
        dec = fp['EVENTS'].header['DEC_OBJ']
    if args.bary:
        orbfile = glob.glob('{}/*orb.fits'.format(idir))
        if len(orbfile) == 0:
            raise IOError("I couldn't find the orbit file in rootdir/event_cl")
        orbfile = orbfile[0]
        bary_txt = (
            'barycorr=yes srcra_barycorr={} srcdec_barycorr={} '
            'orbitfile={} usrgtibarycorr={}'
        ).format(ra, dec, orbfile, gti_bary)
    # ------------------------- #



    # --------------------------- #
    # looping through energy bins #
    CMD = (
        'nuproducts bkgextract=yes imagefile=none '
        'phafile={} bkgphafile={} runbackscale=yes correctlc=yes '
        'runmkarf=no runmkrmf=no  srcregionfile=src.reg bkgregionfile=bgd.reg '
        'indir={} outdir={} instrument={} steminputs={} stemout={} '
        'binsize={} pilow={} pihigh={} lcenergy={} {} {}'
    )

    bcorr = []
    for ie in range(nbins):

        # extract spectra and backscale only once
        specfiles = 'DEFAULT' if ie==0 else 'NONE'

        for ii in [0,1]:
            sout = out.format('{}__{}'.format(instr[ii], ie+1))
            cmd = CMD.format(specfiles, specfiles, idir, odir, INSTR[ii],
                    stem, sout, tbin, chans[ie][0], chans[ie][1],
                    (enegs[ie][0] + enegs[ie][1])/2., usr_gti, bary_txt)
            run_cmd(cmd)

            ## backscale correction factor ##
            if ie == 0:
                pha_s = '{}_sr.pha'.format(sout)
                pha_b = '{}_bk.pha'.format(sout)
                with pyfits.open(pha_s) as fs:
                    s_backscale = fs['SPECTRUM'].header['BACKSCAL']
                with pyfits.open(pha_b) as fs:
                    b_backscale = fs['SPECTRUM'].header['BACKSCAL']
                bcorr.append(s_backscale/b_backscale)

            # subtract background from source #
            cmd = 'lcmath {0}_sr.lc {0}_bk.lc {0}.lc 1.0 {1} no'.format(
                        sout, bcorr[ii])
            run_cmd(cmd)

            # clean #
            run_cmd('rm *flc *gif *pha', True)

    # --------------------------- #


