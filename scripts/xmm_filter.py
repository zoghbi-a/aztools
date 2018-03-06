#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import subprocess
import argparse
import os


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
        Prepare an event file for light curve extractions.
        This should be called before xmmlc.py
        ''',            
        formatter_class=argparse.ArgumentDefaultsHelpFormatter ) 

    p.add_argument("event_file", metavar="event_file", type=str,
            help="The name of the input event file")
    p.add_argument("instr", metavar="instr", type=str,
            help="What instruments: pn|m1|m2")
    p.add_argument("--region", action='store_true', default=False,
            help="Extract region file")
    p.add_argument("--keep", action='store_true', default=False,
            help="Keep temporary files")
    p.add_argument("--std", action='store_true', default=False,
            help="do standard background flare filtering")
    p.add_argument("--bary", action='store_true', default=False,
            help="Apply barycenter corrections")
    p.add_argument("--e_expr", metavar="e_expr", type=str, default='',
            help="Extra filtering expression. e.g. &&(PI>4000)")
    p.add_argument("--e_args", metavar="e_args", type=str, default='',
            help="Extra arguments for evselect.")
    p.add_argument("--raw", action='store_true', default=False,
            help="region is in RAWX, RAWY instead of X, Y")
    
    args = p.parse_args()
    event = args.event_file
    instr = args.instr
    if instr not in ['pn', 'm1', 'm2']:
        raise ValueError('instr need to be one of pn|m1|m2')


    # ------------------ #
    # check system setup #
    sas_odf = os.environ.get('SAS_ODF', None)
    if sas_odf is None:
        cwd = os.getcwd() + '/'
        for d in ['../odf', '../../odf']:
            if os.path.exists(cwd + d):
                os.environ['SAS_ODF'] = cwd + d
                break
    sas_odf = os.environ.get('SAS_ODF', None) 
    if sas_odf is None:
        raise RuntimeError(
            'I cannot find odf directory, please define SAS_ODF')
    # ------------------ #



    # -------------------------- #
    # background flare filtering #
    filter_options = {
        'pn': ['IN [10000:12000]', '#XMMEA_EP', 
                '(PI IN [200:12000])&&(PATTERN<=4 )&&(FLAG==0)', 0.4],
        'm1': ['> 10000'         , '#XMMEA_EM', 
                '(PI IN [200:12000])&&(PATTERN<=12)&&(FLAG==0)', 0.35]
    }
    filter_options['m2'] = filter_options['m1']

    filter_cmd = (
        "evselect table={}:EVENTS withrateset=yes rateset=tmp.rate " 
        "expression='(PI {})&&(PATTERN==0)&&{}' "
        "makeratecolumn=yes maketimecolumn=yes timebinsize=100"
    )
    
    cmd = filter_cmd.format(event, 
            filter_options[instr][0], filter_options[instr][1])
    run_cmd(cmd)
    # -------------------------- #


    if not args.std:
        # -------------------- #
        # plot the light curve #
        cmd = "dsplot table=tmp.rate:RATE x=TIME y=RATE"
        run_cmd(cmd)
        # -------------------- #



        # -------------------------- #
        # ask for what to do #
        # C: Continue without time filtering: do only pattern and flag
        # S: do standard time filtering with rate<0.5
        # F: do other time filtering expression
        while True:
            choice = input(
                ('Please select one option:\n<C>ontinue, '
                 '<S>tandard, <F>ilter, <R>ate\n->'))
            if choice not in ['C', 'S', 'F', 'R']: continue
            filter_expr = input('->') if choice in ['F', 'R'] else ''
            break
    else:
        choice = 'S'
        filter_expr = ''

    options = {
        'C': '',
        'S': 'gti(tmp.gti,TIME)&&',
        'F': filter_expr + '&&',
        'R': 'gti(tmp.gti,TIME)&&',
    }

    if choice in ['S', 'R']:
        cut = filter_expr if choice=='R' else filter_options[instr][3]
        cmd = (
            'tabgtigen table=tmp.rate gtiset=tmp.gti '
            'expression="RATE<{}"').format(cut)
        run_cmd(cmd)
    
    cmd = (
            "evselect table={}:EVENTS withfilteredset=yes "
            "filteredset={}_filtered.fits expression='{}{}{}' {}"
            ).format(event, instr, options[choice], 
                filter_options[instr][2], args.e_expr, args.e_args)
    run_cmd(cmd)
    # -------------------------- #


    # ----------------------- #
    # barycenter corrections? #
    if args.bary:
        run_cmd('cp {0}_filtered.fits {0}_filtered_nobary.fits'.format(instr))
        cmd = 'barycen table={}_filtered.fits:EVENTS'.format(instr)
        run_cmd(cmd)
    # ----------------------- #


    # ------------------------------------ #
    # extract source and background region #
    selector = 'RAW' if args.raw else ''
    if args.region:
        cmd = (
            "evselect table={0}_filtered.fits:EVENTS withimageset=yes "
            "imageset=tmp.img xcolumn={1}X ycolumn={1}Y"
            ).format(instr, selector)
        run_cmd(cmd)

        print('\nPlease define a standard ds9 source and background regions\n')
        run_cmd('ds9 tmp.img -log -zoom 2 -cmap heat;')
    # ------------------------------------ #


    # ---------------- #
    # remove tmp files #
    if not args.keep:
        run_cmd('rm tmp*')
    # ---------------- #

