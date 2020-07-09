#!/usr/bin/env python

import argparse as ARG
import numpy as np
#import pyfits
from astropy.io import fits as pyfits
from itertools import groupby
import subprocess as subp
import os


def run_cmd(cmd, quiet=False):
    """Run cmd command"""
    header = '\n' + '*'*20 + '\n' + cmd + '\n' + '*'*20 + '\n'
    print(header)
    ret = subp.call(cmd, shell='True')
    if ret != 0 and not quiet:
       raise SystemExit('\nFailed in the command: ' + header)


if __name__ == '__main__':
    
	p   = ARG.ArgumentParser(                                
		description='''
		Rebin a spectrum and its response to match its grouping
		This may require changing rbnrmf.f in heasoft so it can handle
		long lists of bins. change maxcomp to 512
		''',            
		formatter_class=ARG.ArgumentDefaultsHelpFormatter ) 

	p.add_argument("specfile"  , metavar="specfile", type=str,
			help="The name of the input grouped file")
	p.add_argument('-s', "--suffix"  , metavar="suffix", type=str, default='.b',
			help="suffix to add to the file names")
	p.add_argument('-e', "--error"  , metavar="error", type=str, default='properr=no error=poiss-0',
			help="error calculation in ftrbnpha")
	p.add_argument('-o', "--osample"  , metavar="osamples", type=int, default=0,
			help="oversampling factor. Default 0")
	args = p.parse_args()


	## --- read the names of bgd, rmf and arf files --- ##
	specfile = args.specfile
	spec = pyfits.open( specfile )
	try:
		rspfile = spec['SPECTRUM'].header['RESPFILE']
	except:
		rspfile = input('No RESPFILE keyword, please enter name of response file:')

	try:
		bgdfile = spec['SPECTRUM'].header['BACKFILE']
	except:
		bgdfile = input('No BACKFILE keyword, please enter name of response file:')
	try:
		arffile = spec['SPECTRUM'].header['ANCRFILE']
	except:
		print('## WARNING ##: no arf file found, assuming none')
	## ------------------------------------------------ ##


	## -- Read the GROUPING from specfile -- ##
	try: 
		grouping = spec['SPECTRUM'].data['GROUPING']
	except:
		raise KeyError( 
    		'Could not read the GROUPING column in the spectrum file {0}'.
    		format(specfile))
	try: 
		quality = spec['SPECTRUM'].data['QUALITY']
	except:
		quality = np.zeros_like(grouping)
	grouping[quality!=0] = -2
	grouping[0] = 1
	spec.close()
	## -------------------------------------- ##


	## -- Energy boundaries of channels -- ##
	## -- and energy axis               -- ##
	with pyfits.open(rspfile) as fp:
		chan_e = fp['EBOUNDS'].data.field(1)
		e_axis = fp['MATRIX'].data.field(0)
	## ----------------------------------- ##


	## -- The starting channel & -- ##
	## -- energy of every group  -- ##
	chan_grp = [i[0] for i in np.argwhere(grouping==1)]
	chan_e_grp = chan_e[chan_grp]
	## ---------------------------- ##


	## -- corresponding energy of every group -- ##
	## -- in the energy axis of the rmf       -- ##
	en_grp = np.abs(e_axis - chan_e_grp[:,np.newaxis]).argmin(1)+1
	## ----------------------------------------- ##



	## -- produce the binning arrays -- ##
	ch = [[x,y-1,y-x] for x,y in zip(chan_grp[:-1],chan_grp[1:])]
	en = [[x,y-1,y-x] for x,y in zip(en_grp[:-1],en_grp[1:]) if x!=y]
	nch = len(grouping)
	if ch[-1][1] != nch-1: 
		ch.append([ch[-1][1]+1, nch-1,nch-1-ch[-1][1]])
	nen = len(e_axis)
	if en[-1][1] != nen-1: 
		en.append([en[-1][1]+1, nen-1,nen-1-en[-1][1]])
	## -------------------------------- ##


	## -- oversampling factor -- ##
	if args.osample != 0:
		osample = int(args.osample)

		arr = []
		for c in en:
			if c[2] == 1:
				arr.append(c)
				continue
			ii = c[2] // osample
			jj = [ii + (1 if (i<c[2]%osample) else 0) for i in range(osample)]
			ic = c[0]
			for i in jj: 
				arr.append([ic, ic+i-1, i])
				ic += i
		en = [i for i in arr]




	## -- make arrays compact by combining groupings -- ##
	## -- with the same number of channels           -- ##
	ch_compact = []
	for k,g in groupby(ch, lambda x: x[2]):
		#first = last = g.next()
		first = last = next(g)
		for last in g: pass
		ch_compact.append([first[0], last[1], k])
	en_compact = []
	for k,g in groupby(en, lambda x: x[2]):
		#first = last = g.next()
		first = last = next(g)
		for last in g: pass
		en_compact.append([first[0], last[1], k])
	## ------------------------------------------------ ##



	## -- write binning sequences to ch and en files -- ##
	ch_txt  = '\n'.join(['{0} {1} {2}'.format(a,b,c) for a,b,c in ch_compact])
	en_txt  = '\n'.join(['{0} {1} {2}'.format(a,b,c) for a,b,c in en_compact])
	with open('tmp.chan', 'w') as fp: fp.write(ch_txt + '\n')
	with open('tmp.en'  , 'w') as fp: fp.write(en_txt + '\n')
	## ------------------------------------------------ ##


	## -- apply binning to spectrum -- ##
	s = args.suffix
	bspec = '{0}{1}'.format(specfile,s)
	bbgd  = '{0}{1}'.format(bgdfile,s)
	brmf  = '{0}{1}'.format(rspfile,s)
	barf  = '{0}{1}'.format(arffile,s)
	os.system('rm {0} {1} {2} {3} &> /dev/null'.format(bspec, bbgd, brmf, barf))
	
	# --- #
	cmd = 'ftrbnpha {0} binfile=tmp.chan outfile={1} {2}'.format(specfile, bspec, args.error)
	run_cmd(cmd)
	with pyfits.open(bspec) as fp:
		fp['spectrum'].header['DETCHANS'] = fp['spectrum'].header['naxis2']
		fp.writeto(bspec, overwrite=True)
	# --- #
	cmd = 'ftrbnpha {0} binfile=tmp.chan outfile={1} {2}'.format(bgdfile, bbgd, args.error)
	run_cmd(cmd)
	with pyfits.open(bbgd) as fp:
		fp['spectrum'].header['DETCHANS'] = fp['spectrum'].header['naxis2']
		fp.writeto(bbgd, overwrite=True)
	# --- #
	if arffile != 'NONE':
		cmd = 'ftrbnarf {0} binfile=tmp.en outfile={1}'.format(arffile, barf)
		run_cmd(cmd)
	# --- #
	os.system('cp {} tmp.rmf'.format(rspfile))
	os.system('fthedit tmp.rmf["MATRIX"] LO_THRES a "1e-6 /grp2bin"')
	#cmd = 'ftrbnrmf tmp.rmf binfile=tmp.chan cmpmode="binfile" ebinfile=tmp.en outfile=%s'%(brmf)
	cmd = 'ftrbnrmf tmp.rmf binfile=tmp.chan cmpmode=binfile outfile=%s'%(brmf)
	run_cmd(cmd)
	with pyfits.open(brmf) as fp:
		fp['matrix'].header['DETCHANS'] = fp['ebounds'].header['DETCHANS']
		fp.writeto(brmf, overwrite=True)


	# --- #
	os.system("mv {0} tmp.src".format(bspec))
	cmd = ['grppha tmp.src {0} "chkey BACKFILE {1}&'.format(bspec,bbgd),
		   'chkey RESPFILE {0}&chkey ANCRFILE '.format(brmf),
		   '' if arffile=='NONE' else barf,
		   '&exit"']
	cmd = ''.join(cmd)
	run_cmd(cmd)
	## ------------------------------- ##

	## -- clean -- ##
	#os.system("rm tmp.chan tmp.en tmp.rsp tmp.src &> /dev/null")
	## ----------- ##
