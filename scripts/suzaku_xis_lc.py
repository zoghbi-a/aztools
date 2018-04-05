#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import subprocess
import argparse
import glob
import os
import xspec as xs
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
    pass
    p   = argparse.ArgumentParser(                                
        description='''
        Extract suzaku xis light curve.
        Assumes heasoft stuff can be run. Event files are in rootdir/.
        Also assumes current directory has src.reg and bgd.reg files. If not 
        present, search .., ../spec etc. We also assume the spectra have been
        calculated in ../spec or ../../spec. If not, fail
        ''',            
        formatter_class=argparse.ArgumentDefaultsHelpFormatter ) 


    p.add_argument("--rootdir"  , metavar="rootdir", type=str, default='../../xis/event_cl',
            help="the root directory that contains event_cl")
    p.add_argument('-e', "--ebins"  , metavar="ebins", type=str, default='2 10',
            help="A space separated list of energy limits in keV")
    p.add_argument('-t', "--tbin"  , metavar="tbin", type=float, default=256,
            help="The time bin, negative means 2**tbin")
    args = p.parse_args()

    # ----------- #
    # parse input #
    tbin = args.tbin
    if tbin < 0: tbin = 2**tbin
    ebins = np.array([x for x in args.ebins.split() if len(x)>0], np.double)
    nbins = len(ebins)-1
    out   = 'lc_{:03g}_{{}}__{{}}'.format(tbin) 
    # ----------- #


    # ---- #
    # dirs #
    idir = args.rootdir
    odir = '.'
    evnt = glob.glob('{}/*xi*cl*evt'.format(idir))
    if len(evnt) == 0:
        raise IOError("I couldn't find the events files. use --rootdir")
    evnt = [e.split('/')[-1] for e in evnt]
    
    # spec dir #
    spec_dir = None
    for s in ['../spec', '../../spec']:
        if os.path.exists(s):
            spec_dir = s
            break
    if spec_dir is None:
        raise ValueError('no spec dir found in .. and ../..')
    # ---- #



    
    # ---------------------- #
    # bring the region files #
    cwd = os.getcwd() + '/'
    if not os.path.exists(cwd + spec_dir + '/src.reg'):
        raise ValueError('No src.reg in spec_dir')
    else:
        run_cmd('cp {}/*reg .'.format(spec_dir))
    # ---------------------- #

    # ---------------- #
    # backscale values #
    backscale = []
    for pat in ['xi0', 'xi1', 'xi3']:
        sfile = glob.glob('{}/*{}*src'.format(spec_dir, pat))
        bfile = glob.glob('{}/*{}*bgd'.format(spec_dir, pat))
        if len(sfile)==0 or len(bfile)==0:
            raise ValueError('No spectra found for {} in {}'.format(pat, spec_dir))
        src_bs = pyfits.open(sfile[0])[1].header['backscal']
        bgd_bs = pyfits.open(bfile[0])[1].header['backscal']
        backscale.append(src_bs/bgd_bs)
    # ---------------- #

    # ------------------------------------------------- #
    # use xspec to get the energy to channel conversion #
    chans = []
    for pat in ['xi0', 'xi1', 'xi3']:
        sfile = glob.glob('{}/*{}*src'.format(spec_dir, pat))
        run_cmd('cp {}* .'.format(sfile[0][:-3]))
        xs.AllData.clear()
        spec = xs.Spectrum(sfile[0].split('/')[-1])
        ch = []
        for ie in range(nbins):
            spec.notice('**')
            spec.ignore('0.0-{:3.3f}, {:3.3f}-**'.format(ebins[ie], ebins[ie+1]))
            ch.append([min(spec.noticed), max(spec.noticed)])
        chans.append(ch)
        run_cmd('rm {}*'.format(sfile[0].split('/')[-1][:-3]))
    chans = np.array(chans)
    enegs = [ [ebins[i],ebins[i+1]] for i in range(nbins) ]
    np.savez('energy_{:03g}.npz'.format(tbin), en=enegs, chans=chans)
    # ------------------------------------------------- #



    # ------------------------------------------ #
    # Extract the light curve from xi0, xi1, and xi3 #
    for ie in range(nbins):
        for ipat, pat in enumerate(['xi0', 'xi1', 'xi3']):

            # add pat output name #
            suff = out.format(pat, ie+1)
            ch1, ch2 = chans[ipat, ie]

            os.system('rm %s* >& /dev/null'%suff)
            xsel = ('tmp_%s\n'%pat + 
                    '\n'.join(['read event {} {}'.format(e, idir) for e in evnt if pat in e]) + 
                    '\nfilter PHA_CUTOFF {} {}'.format(ch1, ch2) +
                    '\nfilter region src.reg' + 
                    '\nextract curve bin=%f offset=no\nsave curve %s.src'%(tbin, suff) + 
                    '\nclear region\nfilter region bgd.reg' + 
                    '\nextract curve bin=%f offset=no\nsave curve %s.bgd'%(tbin, suff) + 
                    '\nexit\nno')

            # call xselect #
            proc = subprocess.Popen("xselect", stdout=subprocess.PIPE, 
                        stdin=subprocess.PIPE)
            proc.communicate(xsel.encode())
            proc.wait()

            # subtract background from source #
            cmd = 'lcmath {0}.src {0}.bgd {0}.lc 1.0 {1} no'.format(
                    suff, backscale[ipat])
            run_cmd(cmd)

    # clear
    os.system('rm xselect.log tmp* &> /dev/null')


    # ------------------------------------------ #


