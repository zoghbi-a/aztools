#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import subprocess
import argparse
import glob
import time
import os
from astropy.io import fits as pyfits
from IPython import embed


def run_cmd(cmd):
    """Run cmd command"""
    header = '\n' + '*'*20 + '\n' + cmd + '\n' + '*'*20 + '\n'
    print(header)
    ret = subprocess.call(cmd, shell='True')
    if ret != 0:
        raise SystemExit('\nFailed in the command: ' + header)

if __name__ == '__main__':
    p   = argparse.ArgumentParser(                                
        description='''
        Extract suzaku xis spectra.
        Assumes heasoft stuff can be run. Event files are in rootdir/.
        Also assumes current directory has src.reg and bgd.reg files. If not 
        present, search .., ../lc etc. If not present create them with
        --create_region.
        ''',            
        formatter_class=argparse.ArgumentDefaultsHelpFormatter ) 


    p.add_argument('-o', "--out"    , metavar="out", type=str, default='spec',
            help="stem for output fits files.")
    p.add_argument("--rootdir"  , metavar="rootdir", type=str, default='../xis/event_cl',
            help="the root directory that contains event_cl")
    p.add_argument("--create_region", action='store_true', default=False,
            help="Create new region files")
    p.add_argument("--t_expr", metavar='t_expr', type=str, default='',
            help="Time selection expression so in xselect we have; filter time {t_expr}")
    p.add_argument("--noclean", action='store_true', default=False,
            help="Don't clear files; useful when running script multiple times in same dir")
    p.add_argument("--mode", metavar='mode', default='medium', type=str,
            help="slow|medium|fast to be passed to xisresp")
    p.add_argument("--lc", action='store_true', default=False,
            help="extract spec for use in lc; force fast mode, and do not rebin the spectra/response")
    args = p.parse_args()

    # ----------- #
    # parse input #
    out = args.out
    t_expr = args.t_expr
    if t_expr != '':
        t_expr = '\nfilter time %s\n'%t_expr
    noclean = args.noclean
    mode = args.mode
    islc = args.lc
    # ----------- #


    # ---- #
    # dirs #
    idir = args.rootdir
    odir = '.'
    evnt = glob.glob('{}/*xi*cl*evt'.format(idir))
    if len(evnt) == 0:
        raise IOError("I couldn't find the events files. use --rootdir")
    evnt = [e.split('/')[-1] for e in evnt]
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
            'tmp\nread event {} {}\nyes\nextract image\n'
            'save image tmp.img\nexit\nno\n').format(evnt[0], idir)
        proc = subprocess.Popen("xselect", stdout=subprocess.PIPE, 
                    stdin=subprocess.PIPE)
        #proc.universal_newlines = True
        proc.communicate(xsel.encode())
        proc.wait()
        run_cmd('ds9 tmp.img -log -zoom 2 -cmap heat')
        run_cmd('rm tmp.* xselect.log &> /dev/null')
    # ----------------------------- #



    # ------------------------------------------ #
    # Extract the spectra from xi0, xi1, and xi3 #
    irand = np.random.randint(1000, 100000)
    for pat in ['xi0', 'xi1', 'xi3']:
        # add pat output name #
        orig = out.split('_')
        orig.insert(-1, pat)
        suff = '_'.join(orig)

        os.system('rm %s* >& /dev/null'%suff)
        xsel = ('tmp_%s_%d\n'%(pat, irand) + 
                '\n'.join(['read event {} {}'.format(e, idir) for e in evnt if pat in e]) + 
                '\nfilter region src.reg' + 
                t_expr + 
                '\nextract spec\nsave spec %s.src group=no resp=no'%suff + 
                '\nclear region\nfilter region bgd.reg' + 
                '\nextract spec\nsave spec %s.bgd group=no resp=no'%suff + 
                '\nexit\nno')

        # call xselect #
        proc = subprocess.Popen("xselect", stdout=subprocess.PIPE, 
                    stdin=subprocess.PIPE)
        proc.communicate(xsel.encode())
        proc.wait()

        # response #
        # the code in xisresp need to be modified so it doesn't delete chanfile.txt
        # and to use ftrbnpha/ftrbnrmf.
        run_cmd('xisresp {}.src {} src.reg'.format(suff, mode))

        if mode != 'slow':
            run_cmd('ftrbnpha {0}.bgd outfile={0}_tmp.bgd binfile=chanfile.txt properr=yes'.format(suff))
            run_cmd('mv {0}_tmp.bgd {0}.bgd'.format(suff))

        # there a bug in rbnpha; that does not update DETCHANS, so we do it by hnad
        for ext in ['src', 'bgd']:
            with pyfits.open('%s.%s'%(suff, ext)) as fp:
                fp['spectrum'].header['DETCHANS'] = fp['spectrum'].header['naxis2']
                fp.writeto('%s.%s'%(suff, ext), overwrite=True)
        # and the response file too
        with pyfits.open('%s.rsp'%(suff)) as fp:
                fp['matrix'].header['DETCHANS'] = fp['ebounds'].header['DETCHANS']
                fp.writeto('%s.rsp'%(suff), overwrite=True)



        # grouping #
        cmd = 'fthedit {0}.src[SPECTRUM] backfile a {0}.bgd'
        run_cmd(cmd.format(suff))
        cmd = 'fthedit {0}.src[SPECTRUM] respfile a {0}.rsp'
        run_cmd(cmd.format(suff))
        
        #cmd = 'grppha {0}.src {0}.grp "group min 100&exit"'
        #run_cmd(cmd.format(suff))

        os.system('rm %s.grp >& /dev/null'%suff)
        run_cmd('ogrppha.py {0}.src {0}.grp -s 6 -f 3'.format(suff))
        if not noclean:
            os.system('rm xselect.log spec_*orig chanfile.txt energyfile.txt tmp* >& /dev/null')

    # combine xi0, xi3 to get front-illuminated (fi) spectra
    files = glob.glob('*xi[0,3]*grp')
    if len(files) == 2:
        print('--- combining xi0, xi3 ---')
        orig = out.split('_')
        orig.insert(-1, 'fi')
        suff = '_'.join(orig)
        with open('tmp.add', 'w') as fp: fp.write('\n'.join(files))
        run_cmd('addspec tmp.add {} yes yes'.format(suff))
        run_cmd('ogrppha.py {0}.pha {0}.grp -s 6 -f 3'.format(suff))



    # ------------------------------------------ #


