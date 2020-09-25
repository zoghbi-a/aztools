#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import subprocess
import argparse
import glob
import os
from astropy.io import fits as pyfits
from aztools import data_tools


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
    backscale, src_backscale = [], []
    for pat in ['xi0', 'xi1', 'xi3']:
        sfile = glob.glob('{}/*{}*src'.format(spec_dir, pat))
        bfile = glob.glob('{}/*{}*bgd'.format(spec_dir, pat))
        if len(sfile)==0 or len(bfile)==0:
            raise ValueError('No spectra found for {} in {}'.format(pat, spec_dir))
        src_bs = pyfits.open(sfile[0])[1].header['backscal']
        bgd_bs = pyfits.open(bfile[0])[1].header['backscal']
        backscale.append(src_bs/bgd_bs)
        src_backscale.append(src_bs)
    # ---------------- #

    # ------------------------------------------------------------- #
    # use a direct conversion function of energy and channel number #
    conv  = lambda en:int(np.floor((en*1000)/3.65))
    chans = np.array([[conv(ebins[i]),conv(ebins[i+1])-1] for i in range(nbins)])
    enegs = [ [ebins[i],ebins[i+1]] for i in range(nbins) ]
    np.savez('energy_{:03g}.npz'.format(tbin), en=enegs, chans=chans)
    # ------------------------------------------------------------- #


    # ------------------------------------------ #
    # Extract the light curve from xi0, xi1, and xi3 #
    for ie in range(nbins):
        
        ch1, ch2 = chans[ie]
        print('Channels for energy bin %d: %d %d'%(ie+1, ch1, ch2))
            
        for ipat, pat in enumerate(['xi0', 'xi1', 'xi3']):

            # add pat output name #
            suff = out.format(pat, ie+1)

            os.system('rm %s* >& /dev/null'%suff)
            xsel = ('tmp_%s\n'%pat + 
                    '\n'.join(['read event {} {}'.format(e, idir) for e in evnt if pat in e]) + 
                    #'\nfilter PHA_CUTOFF {} {}'.format(ch1, ch2) +
                    '\nfilter COLUMN "PI={}:{}"'.format(ch1, ch2) +
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
            data_tools.fits_lcmath('%s.src'%suff, '%s.bgd'%suff, '%s.lc'%suff, 1.0, -backscale[ipat])
                
            
        # combine xi0 and xi3
        data_tools.fits_lcmath('%s.lc'%out.format('xi0', ie+1), '%s.lc'%out.format('xi3', ie+1), 
               '%s.lc'%out.format('fi', ie+1),  1.0, src_backscale[0]/src_backscale[2])
        
        # combine all xi detectors #
        data_tools.fits_lcmath('%s.lc'%out.format('fi', ie+1), '%s.lc'%out.format('xi1', ie+1), 
               '%s.lc'%out.format('all', ie+1),  1.0, src_backscale[0]/src_backscale[1])

    # clear
    os.system('rm xselect.log tmp* &> /dev/null')


    # ------------------------------------------ #


