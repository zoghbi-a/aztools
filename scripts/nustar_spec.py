#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import subprocess
import argparse
import glob
import os
from astropy.io import fits as pyfits


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
        Extract nustar spectra.
        Assumes heasoft stuff can be run. Event files are in rootdir/event_cl.
        Also assumes current directory has src.reg and bgd.reg files. If not 
        present, search .., ../lc/1b etc. If not present create them with
        --create_region.
        ''',            
        formatter_class=argparse.ArgumentDefaultsHelpFormatter ) 


    p.add_argument('-o', "--out"    , metavar="out", type=str, default='spec',
            help="stem for output fits files.")
    p.add_argument("--gti"  , metavar="gti", type=str, default='',
            help="A user gti file to use.")
    p.add_argument("--rootdir"  , metavar="rootdir", type=str, default='../',
            help="the root directory that contains event_cl")
    p.add_argument("--instr"  , metavar="instr", type=str, default='both',
            help="what instrument. both|a|b")
    p.add_argument("--create_region", action='store_true', default=False,
            help="Create new region files")
    p.add_argument("--bary", action='store_true', default=False,
            help=("Apply barycenter corrections. Needed when using gti's "
                "where baycor has been applied"))
    p.add_argument("--gti_bary", action='store_true', default=False,
            help="Apply barycenter corrections to user gti")
    args = p.parse_args()

    # ----------- #
    # parse input #
    out = args.out
    usr_gti = args.gti
    if usr_gti != '':
        usr_gti = ' usrgtifile={}'.format(usr_gti)
    if not args.instr in ['both', 'a', 'b']:
        raise ValueError('--instr need to be one of both|a|b')
    instr = ['a', 'b']
    INSTR = ['FPMA', 'FPMB']
    if args.instr in ['a', 'b']:
        idx = 0 if args.instr == 'a' else 1
        instr = [instr[idx]]
        INSTR = [INSTR[idx]]
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
    for d in ['.', '..', '../lc/1b', '../../lc/1b']:
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
        #proc.universal_newlines = True
        proc.communicate(xsel.encode())
        proc.wait()
        run_cmd('ds9 tmp.img -log -zoom 2 -cmap heat')
        run_cmd('rm tmp.* xselect.log &> /dev/null')
    # ----------------------------- #


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
        'nuproducts bkgextract=yes imagefile=none lcfile=none '
        'phafile=DEFAULT bkgphafile=DEFAULT runbackscale=yes runmkarf=yes '
        'runmkrmf=yes  srcregionfile=src.reg bkgregionfile=bgd.reg '
        'indir={} outdir={} instrument={} steminputs={} stemout={} {} {}'
    )

    # --------------- #
    # extract spectra #
    for ii in range(len(instr)):
        sout = '{}_{}'.format(out, instr[ii])
        cmd = CMD.format(idir, odir, INSTR[ii], stem, sout, usr_gti,
                bary_txt)
        run_cmd(cmd)

        # grouping #
        cmd = 'grppha {0}_sr.pha !{0}.grp "group min 20&exit"'
        run_cmd(cmd.format(sout))


        # clean #
        run_cmd('rm *gif')

    # --------------- #


