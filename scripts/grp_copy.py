#!/usr/bin/env python


import pyfits
import numpy as np
import os
import argparse as ARG


if __name__ == '__main__':
    
	p   = ARG.ArgumentParser(                                
		description='''
		Copy the grouping of one file to another file
		''',            
		formatter_class=ARG.ArgumentDefaultsHelpFormatter ) 

	p.add_argument("grp_from"  , metavar="grp_from", type=str,
			help="The name of the groupped file from which to copy.")
	p.add_argument("grp_to"  , metavar="grp_to", type=str,
			help="The name of the file to which to copy the grouping")
	p.add_argument('-o', '--output', metavar='output', type=str, default='.g',
			help='Name of output file. Default, grp_to.g')


	args = p.parse_args()
	grp1 = args.grp_from
	grp2 = args.grp_to
	out  = args.output
	if out == '.g': out = grp2 + '.g'


	## -- Read the input spec -- ##
	f1 = pyfits.open(grp1)
	try:
		g1 = f1['SPECTRUM'].data.field('GROUPING')
	except:
		print 'I could not read the grouping from {}'.format(grp1)
		exit(0)
	f1.close()
	## -------------------------- ##
	
	## -- write the binning to the output file -- ##
	#from IPython  import embed
	#embed();exit(0)
	f2 = pyfits.open(grp2)
	hdu	= f2['SPECTRUM']
	orig_cols = hdu.columns
	new_cols = pyfits.ColDefs([
		pyfits.Column(name='GROUPING', format='1I', array=g1)])
	try:
		tbl = pyfits.BinTableHDU.from_columns(orig_cols + new_cols)
	except:
		orig_cols.del_col('GROUPING')
		tbl = pyfits.BinTableHDU.from_columns(orig_cols + new_cols)
	hdu.header.update(tbl.header.copy())
	tbl.header = hdu.header.copy()
	grp = pyfits.HDUList([f2[0],tbl])
	os.system('rm {0} &> /dev/null'.format(out))
	grp.writeto(out)
	f2.close()
	## ------------------------------------------ ##
