"""Data Tools"""

import functools
from multiprocessing import Pool

import heasoftpy as hsp

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
    # indir=$indir outdir=$outdir steminputs=$stem \
# entry_stage=1 exit_stage=2 clobber=yes instrument=$instr

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
