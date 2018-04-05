#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import subprocess
import argparse
import os
import astropy.io.fits as pyfits


def run_cmd(cmd):
    """Run cmd command"""
    header = '\n' + '*'*20 + '\n' + cmd + '\n' + '*'*20 + '\n'
    print(header)
    ret = subprocess.call(cmd, shell='True')
    if ret != 0:
       raise SystemExit('\nFailed in the command: ' + header)


def check_pileup(event, regions):
    """
    event: unfiltered event file
    regions: a list of src/bgd; we only need src
    """
    # ------------------------------------- #
    # create event file for src region only #
    root = event.split('/')[-1] 
    cmd  = ('evselect table={} filteredset={}.pu.filtered expression="{}"'
            ).format(event, root, regions[0])
    run_cmd(cmd)
    # -------------------------------------- #


    # ------------ #
    # run epatplot #
    cmd = ('epatplot set={0}.pu.filtered plotfile={0}.pu.gif '
           'device=/GIF pileupnumberenergyrange="2000 10000"').format(root)
    run_cmd(cmd)
    os.system('rm pgplot* &> /dev/null')
    # ------------ #


if __name__ == '__main__':
    pass
    p   = argparse.ArgumentParser(                                
        description='''
        Extract xmm spectra and responses.
        xmm_filter.py is assumed to have been run first to create
        a filtered event (with Standard filtering) file and a region file.
        ''',            
        formatter_class=argparse.ArgumentDefaultsHelpFormatter ) 

    p.add_argument("event_file", metavar="event_file", type=str,
            help="The name of the input event file")
    p.add_argument("region_file", metavar="region_file", type=str,
            help="The name of the region file e.g. from xmm_filter.py")
    p.add_argument('-o', "--out"    , metavar="out", type=str, default='spec',
            help="stem for output fits files.")
    p.add_argument("--gti"  , metavar="gti", type=str, default='',
            help="A user gti file to use.")
    p.add_argument("--raw", action='store_true', default=False,
            help="region is in RAWX, RAWY instead of X, Y")
    p.add_argument("--e_expr", metavar="e_expr", type=str, default='',
            help="Extra filtering expression. e.g. &&(PI>4000)")
    p.add_argument("--e_args", metavar="e_args", type=str, default='',
            help="Extra arguments for evselect expression.")
    p.add_argument("--useRsp", metavar="useRsp", type=str, default='',
            help="Don't generate rmf/arf, and use these space separated list")
    p.add_argument("--check_pileup", action='store_true', default=False,
            help=("Check for pileup instead of calculating spectrum."
                "event_file should contain unfiltered file"))
    args = p.parse_args()

    # ----------- #
    # parse input #
    out = args.out
    event = args.event_file
    usr_gti = args.gti
    if usr_gti != '':
        usr_gti = '&&gti({},TIME)'.format(usr_gti)
    if args.useRsp == '':
        rmf_file = '{}.rmf'.format(out)
        arf_file = '{}.arf'.format(out)
    else:
        rmf_file, arf_file = args.useRsp.split()
    # ----------- #


    # ----------------- #
    # what instruments? #
    with pyfits.open(event) as fp:
        instr = fp['EVENTS'].header['INSTRUME']
    maxchan = 20479 if 'PN' in instr else 11999
    # ----------------- #


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

    if args.check_pileup:
        check_pileup(event, regions)
        exit(0)

    # ------------------- #
    # extract src spectra #
    cmd = (
        'evselect table={} spectrumset={}.pha expression="{}{}{}" '
        'energycolumn=PI spectralbinsize=5 withspecranges=yes '
        'specchannelmin=0 specchannelmax={} {}'
    ).format(event, out, regions[0], usr_gti, args.e_expr, maxchan, args.e_args)
    run_cmd(cmd)
    cmd = 'backscale spectrumset={}.pha badpixlocation={}'.format(
                out, event)
    run_cmd(cmd)
    # ------------------- #



    # ------------------- #
    # extract bgd spectra #
    cmd = (
        'evselect table={} spectrumset={}.bgd expression="{}{}{}" '
        'energycolumn=PI spectralbinsize=5 withspecranges=yes '
        'specchannelmin=0 specchannelmax={} {}'
    ).format(event, out, regions[1], usr_gti, args.e_expr, maxchan, args.e_args)
    run_cmd(cmd)
    cmd = 'backscale spectrumset={}.bgd badpixlocation={}'.format(
                out, event)
    run_cmd(cmd)
    # ------------------- #


    if args.useRsp == '':
    # -------- #
        # response #
        cmd = 'rmfgen spectrumset={0}.pha rmfset={0}.rmf'.format(out)
        run_cmd(cmd)
        # -------- #


        # --- #
        # arf #
        cmd = (
            'arfgen spectrumset={0}.pha arfset={0}.arf withrmfset=yes '
            'rmfset={0}.rmf badpixlocation={1} detmaptype=psf'
            ).format(out, event)
        run_cmd(cmd)
        # --- #



    # ----------------- #
    # group the spectra #
    cmd = (
        'specgroup spectrumset={0}.pha rmfset={1} '
        'backgndset={0}.bgd arfset={2} '
        'groupedset={0}.grp minSN=6 oversample=3'
        ).format(out, rmf_file, arf_file)
    run_cmd(cmd)
    # ----------------- #


