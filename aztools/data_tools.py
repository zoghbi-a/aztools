import os
import sys
import subprocess as subp
import glob
import numpy as np
from astropy.io import fits as pyfits
import time

from . import lcurve




def process_xmm_obsids(obsids, detector='pn', **kwargs):
    """Process xmm observations. Assume we are in folder containing obsids.
    Also assume that heasoft and xmm-sas commonds can be accessed.
    
    obsids: a list of obsids to process
    detector: pn | mos
    
    Keywords:
    ---------
    nproc_max: maximum number of parallel processess; default: 30
    fresh: fresh data reduction; default: False
    
    """
    
    nproc_max = kwargs.get('nproc_max', 30)
    fresh = kwargs.get('fresh', False)
    
    if not detector in ['pn', 'mos']:
        raise ValueError('detector needs to be pn | mos; %s given'%detector)
    if isinstance(obsids, str):
        obsids = [obsids]
    
    exists = os.path.exists
    os.system('mkdir -p log')
    procs = []
    
    print('*** inital processing ... ***')
    for o in obsids:
        print(o)
        if fresh and exists('%s.tar'%o):
            os.system('rm -rf %s > /dev/null 2>&1'%o)
            os.system('tar -xvf %s.tar'%o)
        if not exists(o):
            raise RuntimeError('%s does not exist!'%o)
        
        os.chdir(o)
        os.system('rm -r ?XMM om_mosaic PPS >/dev/null 2>&1')
        os.system('mv ODF odf >/dev/null 2>&1')
        os.chdir('odf')
        if fresh:
            os.system('rm ccf.cif *SAS >/dev/null 2>&1')
        if not os.path.exists('ccf.cif'):
            os.system('gzip -d *gz >/dev/null 2>&1')
            log_file = '../../log/%s_process.log'%o
            #proc = subp.Popen(['/bin/bash', '-i', '-c', 'xmm_process > %s 2>&1'%log_file])
            proc = subp.Popen(['/bin/bash', '-c', 'xmm_process > %s 2>&1'%log_file])
            procs.append(proc)
            # if we have reached the limit of running processes; wait
            if len(procs) >= nproc_max:
                for p in procs: p.wait()
                procs = []
        # exit obsid/odf
        os.chdir('../..')
    # wait for the tasks to end
    for p in procs: p.wait()
        
    print('*** running processing chain for %s detector ... ***'%detector)
    procs = []
    for o in obsids:
        print(o)
        os.chdir(o)
        os.system('mkdir -p %s'%detector)
        os.chdir(detector)
        if len(glob.glob('*EVL*')) == 0 or fresh:
            log_file = '../../log/%s_process_%s.log'%(o, detector)
            #p = subp.Popen(['/bin/bash', '-i', '-c', 'xmm_process %s > %s 2>&1'%(detector, log_file)])
            p = subp.Popen(['/bin/bash', '-c', 'xmm_process %s > %s 2>&1'%(detector, log_file)])
            procs.append(p)
            if len(procs) >= nproc_max:
                for p in procs: p.wait()
                procs = []
        os.chdir('../..')
    for p in procs: p.wait()

                        
def extract_xmm_spec(obsids, detector='pn', **kwargs):
    """extract xmm spectra. Assume we are in folder containing obsids.
    Also assume that heasoft and xmm-sas commonds can be accessed.
    
    obsids: a list of obsids to process
    detector: pn | mos
    
    Keywords:
    ---------
    nproc_max: maximum number of parallel processess; default: 30
    
    """
    
    nproc_max = kwargs.get('nproc_max', 30)
    
    if not detector in ['pn', 'mos']:
        raise ValueError('detector needs to be pn | mos; %s given'%detector)
    if isinstance(obsids, str):
        obsids = [obsids]
    
    exists = os.path.exists
    os.system('mkdir -p log')
    procs = []
    for iobs,o in enumerate(obsids):
        print(o)
        os.chdir('%s/%s'%(o, detector))
        efiles = ['pn'] if detector=='pn' else ['m1', 'm2']
        if not np.all([exists('%s.fits'%det) for det in efiles]):
            evts = glob.glob('*EVL*')
            if len(evts) == 0:
                raise ValueError('No event files found!')
            if detector == 'pn':
                if len(evts) != 1:
                    raise ValueError('Expected 1 event file for PN not %d'%(len(evts)))
            else:
                if len(evts) != 2:
                    raise ValueError('Expected 2 event files for MOS not %d'%(len(evts)))
            for ef,ev in zip(efiles, evts):
                os.system('rm %s.fits >/dev/null 2>&1; ln -s %s %s.fits'%(ef, ev, ef))
            
        os.system('mkdir -p spec')
        os.chdir('spec')
        for det in efiles:
            if not exists('%s_filtered.fits'%det) or not exists('ds9_%s.reg'%det):
                # check if we have a saved region file, or a temporary region file
                region = ''
                if not exists('ds9_%s'%det):
                    saved_reg = '../../../log/%s_ds9_%s.reg'%(o, det)
                    if exists(saved_reg):
                        os.system('cp %s ds9_%s.reg'%(saved_reg, det))
                    else:
                        region = '--region'
                        
                # run xmm_filter
                print('** filtering the %s data ... **'%(det))
                cmd = 'xmm_filter.py ../%s.fits %s --std %s > xmm_filter.log 2>&1'%(det, det, region)
                #p = subp.call(['/bin/bash', '-i', '-c', cmd])
                p = subp.call(['/bin/bash', '-c', cmd])
                if not exists(saved_reg):
                    os.system('cp ds9_%s.reg %s'%(det, saved_reg))
                
            # now extract the spectra #
            if not exists('spec_%s_%d.grp'%(det, iobs+1)):
                print('** extracting the %s spectra ... **'%det)
                cmd = ('xmm_spec.py %s_filtered.fits ds9_%s.reg -o spec_%s_%d '
                       '> spec_%s.log 2>&1')%(det, det, det, iobs+1, det)
                #p = subp.Popen(['/bin/bash', '-i', '-c', cmd])
                p = subp.Popen(['/bin/bash', '-c', cmd])
                
                procs.append(p)
                if len(procs) >= nproc_max:
                    for p in procs: p.wait()
                    procs = []
        os.chdir('../../..')
    for p in procs: p.wait()

        
def extract_xmm_lc(obsids, lcdir, ebins, dt, detector='pn', **kwargs):
    """extract xmm light curves. Assume we are in folder containing obsids.
    Also assume that heasoft and xmm-sas commonds can be accessed.
    
    obsids: a list of obsids to process
    lcdir: name of folder containing the light curves. e.g. 12b
    ebins: a string of space-separated energy boundaries
    dt: time binning.
    detector: pn | mos

    
    Keywords:
    ---------
    nproc_max: maximum number of parallel processess; default: 30
    stdR: rate cutoff when filtering the data
    extra_opt: extra options to be passed to xmm_lc.py. e.g. --abscorr
    
    """
    
    nproc_max = kwargs.get('nproc_max', 30)
    stdR = kwargs.get('stdR', 0.6)
    extra_opt = kwargs.get('extra_opt', '')
    
    if not detector in ['pn', 'mos']:
        raise ValueError('detector needs to be pn | mos; %s given'%detector)
    if isinstance(obsids, str):
        obsids = [obsids]
    eBins = [' '.join(x) for x in zip(ebins.split()[:-1], ebins.split()[1:])]
    efiles = ['pn'] if detector=='pn' else ['m1', 'm2']
    
    exists = os.path.exists
    os.system('mkdir -p log')
    procs = []
    for iobs,o in enumerate(obsids):
        print(o)
        os.chdir('%s/%s'%(o, detector))
        if not np.all([exists('%s.fits'%det) for det in efiles]):
            evts = glob.glob('*EVL*')
            if len(evts) == 0:
                raise ValueError('No event files found!')
            if detector == 'pn':
                if len(evts) != 1:
                    raise ValueError('Expected 1 event file for PN not %d'%(len(evts)))
            else:
                if len(evts) != 2:
                    raise ValueError('Expected 2 event files for MOS not %d'%(len(evts)))
            for ef,ev in zip(efiles, evts):
                os.system('rm %s.fits >/dev/null 2>&1; ln -s %s %s.fits'%(ef, ev, ef))
            
        os.system('mkdir -p lc')
        os.chdir('lc')
        os.system('ln -s ../../odf/ccf.cif ccf.cif >/dev/null 2>&1')
        for det in efiles:
            region = ''
            if not exists('ds9_%s.reg'%det):
                # check if we have a saved region file, or a temporary region file
                region = ''
                if not exists('ds9_%s'%det):
                    
                    saved_reg = ['../spec/ds9_%s.reg'%(det), 
                                 '../../../log/%s_ds9_%s.reg'%(o, det)]
                    for reg in saved_reg:
                        if exists(reg):
                            os.system('cp %s ds9_%s.reg'%(reg, det))
                            region = ''
                            break
                        else:
                            region = '--region'
            if not exists('%s_filtered.fits'%det) or region == '--region':
                # run xmm_filter
                print('** filtering the %s data ... **'%(det))
                gti = '--e_expr " && gti(nobgd.gti, TIME)"' if exists('nobgd.gti') else ''
                cmd = ('xmm_filter.py ../%s.fits %s --std --stdR %g %s %s'
                       '> xmm_filter.log 2>&1')%(det, det, stdR, region, gti)
                #p = subp.call(['/bin/bash', '-i', '-c', cmd])
                p = subp.call(['/bin/bash', '-c', cmd])
                if region != '':
                    os.system('cp ds9_%s.reg %s'%(det, saved_reg[1]))
                
            # now extract the light curve #
            os.system('mkdir -p %s'%lcdir)
            os.chdir(lcdir)
            if len(glob.glob('lc_{:03g}_*.fits'.format(dt))) != 3*(len(eBins)):
                for ib, eb in enumerate(eBins):
                    wdir = 'tmp_%s_%d'%(det, ib+1)
                    os.system('mkdir -p %s'%wdir); os.chdir(wdir)
                    cmd = ('xmm_lc.py ../../%s_filtered.fits ../../ds9_%s.reg'
                           ' -e "%s" -t %g -o lc_%s %s >lc__1.log 2>&1;'
                            'rename __1 __%d *; mv lc* ..;')%(det, det, eb, dt, det, extra_opt, ib+1)
                    #p = subp.Popen(['/bin/bash', '-i', '-c', cmd])
                    p = subp.Popen(['/bin/bash', '-c', cmd])
                    procs.append(p)
                    os.chdir('..')
                    if len(procs) >= nproc_max:
                        for p in procs: p.wait()
                        procs = []
            os.chdir('..')
        os.chdir('../../..')
    for p in procs: p.wait()
    for o in obsids:
        os.system('rm -rf %s/%s/lc/%s/tmp*'%(o, detector, lcdir))
            
            
def process_nustar_obsids(obsids, **kwargs):
    """Process nustar observations. Assume we are in folder containing obsids.
    Also assume that heasoft commonds can be accessed.
    
    obsids: a list of obsids to process
    
    Keywords:
    ---------
    nproc_max: maximum number of parallel processess; default: 30
    fresh: fresh data reduction; default: False
    
    """
    nproc_max = kwargs.get('nproc_max', 30)
    fresh = kwargs.get('fresh', False)
    
    if isinstance(obsids, str):
        obsids = [obsids]
    
    exists = os.path.exists
    os.system('mkdir -p log')
    procs = []
    for o in obsids:
        print('starting %s ...'%o)
        if fresh and exists('%s.tar'%o):
            os.system('rm -rf %s > /dev/null 2>&1'%o)
            os.system('tar -xvf %s.tar'%o)
        if not exists(o):
            raise RuntimeError('%s does not exist!'%o)
        
        if fresh:
            os.system('rm -r %s_p'%o)
        
        if not exists('%s_p'%o):
            log_file = 'log/%s_process.log'%o
            cmd = ('export HEADASNOQUERY=;export HEADASPROMPT=/dev/null;'
                   'nustar_process %s > %s 2>&1'%(o, log_file))
            #proc = subp.Popen(['/bin/bash', '-i', '-c', cmd])
            proc = subp.Popen(['/bin/bash', '-c', cmd])
            procs.append(proc)
            # if we have reached the limit of running processes; wait
            if len(procs) >= nproc_max:
                for p in procs: p.wait()
                procs = []
        
    # wait for the tasks to end
    for p in procs: p.wait()
        

def extract_nustar_spec(obsids, **kwargs):
    """Extract nustar spectra. Assume we are in folder containing obsids.
    Also assume that heasoft commonds can be accessed. Call nustar_spec.py
    and opens ds9 for region extraction if needed
    
    obsids: a list of obsids to process
    
    Keywords:
    ---------
    nproc_max: maximum number of parallel processess; default: 30
    fresh: fresh data reduction; default: False
    
    """
    nproc_max = kwargs.get('nproc_max', 30)
    fresh = kwargs.get('fresh', False)
    
    if isinstance(obsids, str):
        obsids = [obsids]
    obsids = np.sort(obsids)
    
    exists = os.path.exists
    os.system('mkdir -p log')
    procs = []
    for iobs,o in enumerate(obsids):
        print('starting %s ...'%o)
        os.chdir('%s_p/'%o)
        os.system('mkdir -p spec')
        os.chdir('spec')
        if not fresh and len(glob.glob('spec*grp')) != 2:
            # check if we have a saved region file, or a temporary region file
            saved_reg = '../../log/%s_src.reg'%o
            if exists('src.reg') and exists('bgd.reg'):
                region = ''
            else:
                if exists(saved_reg):
                    os.system('cp %s src.reg'%saved_reg)
                    os.system('cp %s bgd.reg'%(saved_reg.replace('_src.', '_bgd.')))
                    region = ''
                else:
                    region = '--create_region'

            cmd = ('export HEADASNOQUERY=;export HEADASPROMPT=/dev/null;'
                  'nustar_spec.py -o spec_%d %s > ../../log/spec_%s.log 2>&1'%(iobs+1, region, o))
            #p = subp.Popen(['/bin/bash', '-i', '-c', cmd])
            p = subp.Popen(['/bin/bash', '-c', cmd])
            procs.append(p)
            # if we have reached the limit of running processes; wait
            if len(procs) >= nproc_max:
                for p in procs: p.wait()
                procs = []
            if not exists(saved_reg):
                os.system('cp src.reg %s'%saved_reg) 
                os.system('cp bgd.reg %s'%(saved_reg.replace('_src.', '_bgd.')))
        os.chdir('../..')
    # wait for the tasks to end
    for p in procs: p.wait() 
    
    # regreoup the spectra
    for iobs,o in enumerate(obsids):
        os.chdir('%s_p/spec'%o)
        cmd = ('rm *grp; ogrppha.py spec_{0}_a_sr.pha spec_{0}_a.grp -f 3 -s 6;'
               'ogrppha.py spec_{0}_b_sr.pha spec_{0}_b.grp -f 3 -s 6').format(iobs+1)
        #subp.call(['/bin/bash', '-i', '-c', cmd])
        subp.call(['/bin/bash', '-c', cmd])
        os.chdir('../..')


def spec_summary(obsids, sfile):
    """Summary of nustar spec.
    Assume it is called from a folder containing obsids
    
    obsids: a list of them
    sfile: a string for the location of the spectra. e.g.
        '%s_p/spec/spec_%d_a.grp' where it will be formated against
        (obsid, ispec)
    """
    obsids = np.sort(obsids)
    # summary of data
    print('{:5} | {:12} | {:10.8} | {:10.8} | {:10.3} | {:10.3}'.format(
            'num', 'obsid', 'mjd_s', 'mjd_e', 'rate', 'exposure'))
    spec_data = []
    for iobs,o in enumerate(obsids):
        with pyfits.open(sfile%(o, iobs+1)) as fp:
            exposure = fp[1].header['exposure']
            counts = fp[1].data.field('counts').sum()
            tmid = np.array([fp[0].header['tstart'], fp[0].header['tstop']])
            mref = fp[0].header['mjdrefi'] + fp[0].header['mjdreff']
            mjd = tmid / (24*3600) + mref
            spec_data.append([mjd[0], mjd[1], counts/exposure, exposure/1e3])
            text = '{:5} | {:12} | {:10.8} | {:10.8} | {:10.3} | {:10.5}'.format(
                    iobs+1, o, mjd[0], mjd[1], counts/exposure, exposure/1e3)
            print(text)
    spec_data = np.array(spec_data)
    return spec_data


def extract_nustar_lc(obsids, lcdir, ebins, dt, **kwargs):
    """Extract nustar light curves. 
    Assume it is called from a folder containing obsids.
    Use subdirectories to run things in parallel
    
    
    obsids: a list of obsids
    lcdir: name of folder containing lc so that: obsids/lc/{lcdir}
    ebins: a string of space separated energy boudaries. e.g.: '3 10 79'
    dt: time bin
    
    
    Keywords:
    ---------
    nproc_max: maximum number of parallel processess; default: 30
    
    """
    
    nproc_max = kwargs.get('nproc_max', 30)
    
    if isinstance(obsids, str):
        obsids = [obsids]
    
    eBins = [' '.join(x) for x in zip(ebins.split()[:-1], ebins.split()[1:])]
    procs = []
    for o in obsids:
        os.system('mkdir -p %s_p/lc/%s'%(o, lcdir))
        os.chdir('%s_p/lc/%s'%(o, lcdir))
        os.system('cp ../../spec/*reg .')
        for ib, eb in enumerate(eBins):
            if len(glob.glob('lc_{:03g}__?__{}.lc'.format(dt, ib+1))) == 2:
                continue
            wdir = 'tmp_%d'%(ib+1)
            os.system('mkdir -p %s'%wdir); os.chdir(wdir)
            os.system('cp ../*reg .')
            cmd = ('export HEADASNOQUERY=;export HEADASPROMPT=/dev/null;'
                   'nustar_lc.py -e "%s" -t %g --rootdir ../../.. >lc__1.log 2>&1;'
                   'rename __1 __%d lc*; mv lc* ..')%(eb, dt, ib+1)
            #p = subp.Popen(['/bin/bash', '-i', '-c', cmd])
            p = subp.Popen(['/bin/bash', '-c' , cmd])
            procs.append(p)
            if len(procs) >= 20:
                for p in procs: p.wait()
                procs = []
            os.chdir('..')
        os.chdir('../../../')
    # wait for the tasks to end
    for p in procs: p.wait()
    for o in obsids:
        os.system('rm -rf %s_p/lc/%s/tmp*'%(o, lcdir))

        
def process_suzaku_obsids(obsids, **kwargs):
    """Process suzaku observations. Assume we are in folder containing obsids.
    Also assume that heasoft commonds can be accessed.
    
    obsids: a list of obsids to process
    
    Keywords:
    ---------
    nproc_max: maximum number of parallel processess; default: 30
    fresh: fresh data reduction; default: False
    
    """
    nproc_max = kwargs.get('nproc_max', 30)
    fresh = kwargs.get('fresh', False)
    
    if isinstance(obsids, str):
        obsids = [obsids]
    
    exists = os.path.exists
    os.system('mkdir -p log')
    procs = []
    for o in obsids:
        print('starting %s ...'%o)
        if fresh and exists('%s.tar'%o):
            os.system('rm -rf %s > /dev/null 2>&1'%o)
            os.system('tar -xvf %s.tar'%o)
        if not exists(o):
            raise RuntimeError('%s does not exist!'%o)
        
        if fresh:
            os.system('rm -r %s_p'%o)
        
        if not exists('%s_p'%o):
            log_file = 'log/%s_process.log'%o
            cmd = ('export HEADASNOQUERY=;export HEADASPROMPT=/dev/null;'
               'suzaku_process %s xis > %s 2>&1'%(o, log_file))
            #proc = subp.Popen(['/bin/bash', '-i', '-c', cmd])
            proc = subp.Popen(['/bin/bash', '-c' , cmd])
            procs.append(proc)
            # if we have reached the limit of running processes; wait
            if len(procs) >= nproc_max:
                for p in procs: p.wait()
                procs = []
        
    # wait for the tasks to end
    for p in procs: p.wait()

        
        
def extract_suzaku_spec(obsids, **kwargs):
    """Extract suzaku spectra. Assume we are in folder containing obsids.
    Also assume that heasoft commonds can be accessed. Call nustar_spec.py
    and opens ds9 for region extraction if needed
    
    obsids: a list of obsids to process
    
    Keywords:
    ---------
    nproc_max: maximum number of parallel processess; default: 30
    fresh: fresh data reduction; default: False
    extra_opt: extra options for suzaku_xis_spec.py. e.g. "--mode fast"
    
    """
    nproc_max = kwargs.get('nproc_max', 30)
    fresh = kwargs.get('fresh', False)
    extra_opt = kwargs.get('extra_opt', '')
    
    if isinstance(obsids, str):
        obsids = [obsids]
    obsids = np.sort(obsids)
    
    exists = os.path.exists
    os.system('mkdir -p log')
    procs = []
    for iobs,o in enumerate(obsids):
        print('starting %s ...'%o)
        os.chdir('%s_p/'%o)
        os.system('mkdir -p spec')
        os.chdir('spec')
        if not fresh and len(glob.glob('spec*grp')) != 4:
            # check if we have a saved region file, or a temporary region file
            #os.system('rm spec*')
            saved_reg = '../../log/%s_src.reg'%o
            if exists('src.reg') and exists('bgd.reg'):
                region = ''
            else:
                if exists(saved_reg):
                    os.system('cp %s src.reg'%saved_reg)
                    os.system('cp %s bgd.reg'%(saved_reg.replace('_src.', '_bgd.')))
                    region = ''
                else:
                    region = '--create_region'

            cmd = ('export HEADASNOQUERY=;export HEADASPROMPT=/dev/null;'
                  'suzaku_xis_spec.py -o spec_%d %s %s > ../../log/spec_%s.log 2>&1'%(iobs+1, region, extra_opt, o))
            #p = subp.Popen(['/bin/bash', '-i', '-c', cmd])
            p = subp.Popen(['/bin/bash', '-c' , cmd])
            procs.append(p)
            # if we have reached the limit of running processes; wait
            if len(procs) >= nproc_max:
                for p in procs: p.wait()
                procs = []
            if not exists(saved_reg):
                os.system('cp src.reg %s'%saved_reg) 
                os.system('cp bgd.reg %s'%(saved_reg.replace('_src.', '_bgd.')))
        os.chdir('../..')
    # wait for the tasks to end
    for p in procs: p.wait() 
    
    # regreoup the spectra
    for iobs,o in enumerate(obsids):
        os.chdir('%s_p/spec'%o)
        os.system('rm *grp >/dev/null 2>&1')
        for s in ['fi', 'xi0', 'xi1', 'xi3']:
            cmd = ('ogrppha.py spec_{1}_{0}.src spec_{1}_{0}.grp -f 3 -s 6').format(iobs+1, s)
            #subp.call(['/bin/bash', '-i', '-c', cmd])
            subp.call(['/bin/bash', '-c' , cmd])
        os.chdir('../..')

        
def extract_suzaku_lc(obsids, lcdir, ebins, dt, **kwargs):
    """Extract suzaku xis light curves. 
    Assume it is called from a folder containing obsids.
    Use subdirectories to run things in parallel
    
    
    obsids: a list of obsids
    lcdir: name of folder containing lc so that: obsids/lc/{lcdir}
    ebins: a string of space separated energy boudaries. e.g.: '3 10 79'
    dt: time bin
    
    
    Keywords:
    ---------
    nproc_max: maximum number of parallel processess; default: 30
    
    """
    
    nproc_max = kwargs.get('nproc_max', 30)
    
    if isinstance(obsids, str):
        obsids = [obsids]
    
    eBins = [' '.join(x) for x in zip(ebins.split()[:-1], ebins.split()[1:])]
    procs = []
    for o in obsids:
        os.system('mkdir -p %s_p/lc/%s'%(o, lcdir))
        os.chdir('%s_p/lc/%s'%(o, lcdir))
        os.system('cp ../../spec/*reg .')
        os.system('ln -s ../../spec')
        for ib, eb in enumerate(eBins):
            if len(glob.glob('lc_{:03g}_*__{}.lc'.format(dt, ib+1))) == 5:
                continue
            wdir = 'tmp_%d'%(ib+1)
            os.system('mkdir -p %s'%wdir); os.chdir(wdir)
            os.system('cp ../*reg .')
            cmd = ('export HEADASNOQUERY=;export HEADASPROMPT=/dev/null;'
                   'suzaku_xis_lc.py -e "%s" -t %g --rootdir ../../../xis/event_cl '
                   '>lc__1.log 2>&1; rename __1 __%d *; mv lc* ..')%(eb, dt, ib+1)
            #p = subp.Popen(['/bin/bash', '-i', '-c', cmd])
            p = subp.Popen(['/bin/bash', '-c' , cmd])
            procs.append(p)
            if len(procs) >= 20:
                for p in procs: p.wait()
                procs = []
            os.chdir('..')
        os.chdir('../../../')
    # wait for the tasks to end
    for p in procs: p.wait()
    for o in obsids:
        os.system('rm -rf %s_p/lc/%s/tmp*'%(o, lcdir))
        


def fits_lcmath(lc1, lc2, lcout, f1=1.0, f2=1.0):
    """synch the two fits files lc1, lc2 and calculate f1*lc1 + f2*lc2
    """
    
    cols = ['time', 'rate', 'error', 'fracexp']
    with pyfits.open(lc2) as fp:
        d2 = [fp[1].data.field(i) for i in [0,1,2,3]]
    with pyfits.open(lc1) as fp:
        d1 = [fp[1].data.field(i) for i in [0,1,2,3]]
        d_sync = np.array(lcurve.LCurve.sync([d1, d2]))
        d1 = np.c_[d_sync[0,0], 
                   f1*d_sync[0,1] + f2*d_sync[1,1], 
                   ((f1*d_sync[0,2])**2 + (f2*d_sync[1,2])**2)**0.5,
                   d_sync[0,3]]

        orig_cols = fp[1].columns
        for i in range(4):
            orig_cols[cols[i]].array = np.array(d1[:,i], np.float32)
        cols = pyfits.ColDefs(orig_cols)
        tbl = pyfits.BinTableHDU.from_columns(cols)
        fp[1].header.update(tbl.header.copy())
        tbl.header = fp[1].header.copy()
        new_fp = pyfits.HDUList([fp[0],tbl] + fp[2:])
        new_fp.writeto(lcout, overwrite=True)
