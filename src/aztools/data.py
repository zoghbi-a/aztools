"""Data Tools"""

import functools
import glob
import os
from multiprocessing import Pool

import heasoftpy as hsp

from . import misc

__all__ = ['process_nicer_obsid', 'process_nicer_obsids']


def _make_parallel(func, nproc=4):
    """A wrapper to make a function run in parallel
    
    Parameters
    ----------
    func: method
        The method to parallelize. It should expects on obsid
    nproc: int
        Number of processes to run

    Return
    ------
    return a method that takes a list of obsids and calls func
    on each of them in parallel
    
    """

    @functools.wraps(func)
    def parallelize(obsids, **kwargs):

        if isinstance(obsids, str):
            obsids = [obsids]

        with Pool(min(nproc, len(obsids))) as pool:
            results = pool.map(functools.partial(func, **kwargs), obsids)

        return results

    return parallelize


def process_nicer_obsid(obsid: str, **kwargs):
    """Process NICER obsid with nicerl2
    
    Parameters
    ----------
    obsid: str
        Obsid to be processed
    
    Keywords
    --------
    Any parameters to be passed to the reduction pipeline
    
    Return
    ------
    0 if succesful, and a heasoft error code otherwise
    
    """

    # defaults
    in_pars = {
        'geomag_path': '/local/data/reverb/azoghbi/soft/caldb/data/gen/bcf/geomag',
        'filtcolumns': 'NICERV4,3C50',
        'detlist'    : 'launch,-14,-34',
        'min_fpm'    : 50,

        'clobber'    : True,
        'noprompt'   : True
    }
    # update input with given parameter keywords
    in_pars.update(**kwargs)
    in_pars['indir'] = obsid

    # run task
    with hsp.utils.local_pfiles_context(): # pylint: disable=no-member
        out = hsp.nicerl2(**in_pars) # pylint: disable=no-member

    if out.returncode == 0:
        print(f'{obsid} processed sucessfully!')
    else:
        logfile = f'process_nicer_{obsid}.log'
        print(f'ERROR processing {obsid}; Writing log to {logfile}')
        with open(logfile, 'w', encoding='utf8') as filep:
            filep.write(out)
    return out.returncode


# parallel version of process_nicer_obsid
process_nicer_obsids = _make_parallel(process_nicer_obsid)


def process_nustar_obsid(obsid: str, **kwargs):
    """Process NuSTAR obsid with nupipeline
    
    Parameters
    ----------
    obsid: str
        Obsid to be processed
    
    Keywords
    --------
    Any parameters to be passed to the reduction pipeline
    
    Return
    ------
    0 if succesful, and a heasoft error code otherwise
    
    """

    # defaults
    in_pars = {
        'outdir'     : f'{obsid}_p/event_cl',
        'steminputs' : f'nu{obsid}',
        'entrystage' : 1,
        'exitstage'  : 2,
        'pntra'      : 'OBJECT',
        'pntdec'     : 'OBJECT',

        'clobber'    : True,
        'noprompt'   : True,
    }


    # update input with given parameter keywords
    in_pars.update(**kwargs)
    in_pars['indir'] = obsid

    # run task
    with hsp.utils.local_pfiles_context(): # pylint: disable=no-member
        out = hsp.nupipeline(**in_pars) # pylint: disable=no-member

    if out.returncode == 0:
        print(f'{obsid} processed sucessfully!')
    else:
        logfile = f'process_nustar_{obsid}.log'
        print(f'ERROR processing {obsid}; Writing log to {logfile}')
        with open(logfile, 'w', encoding='utf8') as filep:
            filep.write(out)
    return out.returncode

# parallel version of process_nicer_obsid
process_nustar_obsids = _make_parallel(process_nustar_obsid)


def process_suzaku_obsid(obsid: str, **kwargs):
    """Process SUZAKU XIS obsid with aepipeline
    
    Parameters
    ----------
    obsid: str
        Obsid to be processed
    
    Keywords
    --------
    Any parameters to be passed to the reduction pipeline
    
    Return
    ------
    0 if succesful, and a heasoft error code otherwise
    
    """

    # defaults
    instr = 'xis'

    in_pars = {
        'outdir'     : f'{obsid}_p/{instr}/event_cl',
        'steminputs' : f'ae{obsid}',
        'instrument' : instr,
        'entrystage' : 1,
        'exitstage'  : 2,

        'clobber'    : True,
        'noprompt'   : True,
    }


    # update input with given parameter keywords
    in_pars.update(**kwargs)
    in_pars['indir'] = obsid

    # run task
    with hsp.utils.local_pfiles_context(): # pylint: disable=no-member
        out = hsp.aepipeline(**in_pars) # pylint: disable=no-member

    if out.returncode == 0:
        print(f'{obsid} processed sucessfully!')
    else:
        logfile = f'process_suzaku_{obsid}.log'
        print(f'ERROR processing {obsid}; Writing log to {logfile}')
        with open(logfile, 'w', encoding='utf8') as filep:
            filep.write(out)
    return out.returncode

# parallel version of process_suzaku_obsid
process_suzaku_obsids = _make_parallel(process_suzaku_obsid)


def process_xmm_obsid(obsid: str, **kwargs):
    """Process XMM obsid with xmm sas
    
    Parameters
    ----------
    obsid: str
        Obsid to be processed
    
    Keywords
    --------
    instr: str
        Instrument or instrument mode: pn|mos|rgs|om
    Any parameters to be passed to the reduction pipeline
    
    Return
    ------
    0 if succesful, and a heasoft error code otherwise
    
    """

    # defaults
    instr = kwargs.pop('instr', 'pn')
    cwd = os.getcwd()

    try:
        # preparation
        os.chdir(obsid)
        if os.path.exists('ODF'):
            os.system('mv ODF odf')
        os.chdir('odf')
        env = {'SAS_ODF': os.getcwd()}

        if not os.path.exists('ccf.cif'):
            if len(glob.glob('*gz')) > 0:
                os.system('gzip -d *gz')
            cmd = 'cifbuild withccfpath=no analysisdate=now category=XMMCCF fullpath=yes'
            misc.run_cmd_line_tool(cmd, env, logfile='processing_xmm_cifbuild.log')

        env['SAS_CCF'] = f'{os.getcwd()}/ccf.cif'

        if len(glob.glob('*.SAS')) > 0:
            os.system('rm *.SAS')
        cmd = f'odfingest odfdir={os.getcwd()} outdir={os.getcwd()}'
        misc.run_cmd_line_tool(cmd, env, logfile='processing_xmm_odfingest.log')


        # prepare the command
        if instr == 'pn':
            cmd = 'epchain'
        elif instr == 'mos':
            cmd = 'emchain'
        elif instr == 'rgs':
            kwargs.setdefault('orders', '"1 2"')
            kwargs.setdefault('bkgcorrect', 'no')
            kwargs.setdefault('withmlambdacolumn', 'yes')
            cmd = 'rgsproc'
        elif instr == 'om':
            cmd = 'omfchain'
        else:
            raise ValueError('instr needs to be pn|om|rgs|om')

        # the following with raise RuntimeError if the task fails
        cmd = f'{cmd} {" ".join([f"{par}={val}" for par,val in kwargs.items()])}'
        misc.run_cmd_line_tool(cmd, env, logfile=f'processing_xmm_{instr}.log')

        # post run extra tasks
        if instr == 'pn':
            evt = glob.glob('*EVL*')
            if len(evt) != 1:
                raise ValueError('Found >1 event files for pn')
            os.system(f'mv {evt[0]} {instr}.fits')
            print(f'{instr}.fits created successfully')

        if instr == 'mos':
            for subi in [1, 2]:
                evt = glob.glob(f'*M{subi}*MIEVL*')
                if len(evt) != 1:
                    raise ValueError(f'Found >1 event files for mos-{subi}')
                os.system(f'mv {evt[0]} {instr}{subi}.fits')
                print(f'{instr}{subi}.fits created successfully')

        if instr == 'rgs':
            os.system('cp *R1*SRSPEC1* spec_r1.src')
            os.system('cp *R2*SRSPEC1* spec_r2.src')
            os.system('cp *R1*BGSPEC1* spec_r1.bgd')
            os.system('cp *R2*BGSPEC1* spec_r2.bgd')
            os.system('cp *R1*RSPMAT1* spec_r1.rsp')
            os.system('cp *R2*RSPMAT1* spec_r2.rsp')

            cmd = ('rgscombine pha="spec_r1.src spec_r2.src" bkg="spec_r1.bgd spec_r2.bgd" '
                   'rmf="spec_r1.rsp spec_r2.rsp" filepha="spec_rgs.src" filebkg="spec_rgs.bgd" '
                   'filermf="spec_rgs.rsp"')
            misc.run_cmd_line_tool(cmd, env, logfile='processing_xmm_rgscombine.log')
            print('rgs spectra created successfully')

        os.chdir(cwd)

    except Exception as exception: # pylint: disable=broad-exception-caught
        os.chdir(cwd)
        raise exception
