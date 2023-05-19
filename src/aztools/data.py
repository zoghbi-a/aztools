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
    
    Run from top level containting obsid folder
    
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
    
    Run from top level containting obsid folder
    
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
    
    Run from top level containting obsid folder
    
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
    
    Run from top level containting obsid folder
    
    Parameters
    ----------
    obsid: str
        Obsid to be processed
    
    Keywords
    --------
    instr: str
        Instrument or instrument mode: pn|mos|rgs|om
    Any parameters to be passed to the reduction pipeline
    
    
    """

    # defaults
    instr = kwargs.pop('instr', 'pn')
    if instr not in ['pn', 'mos', 'rgs', 'om']:
        raise ValueError('instr need to be one of pn|mos|rgs|om')
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


def filter_xmm_obsid(obsid: str, **kwargs):
    """Filter XMM pn or mos obsid with xmm sas
    
    Run from top level containting obsid folder.
    By default, do standard filtering; R<0.4 in pn or R<0.35 for mos
    
    Parameters
    ----------
    obsid: str
        Obsid to be processed
    
    Keywords
    --------
    instr: str
        Instrument or instrument mode: pn|mos1|mos2
    gtiexpr: str
        GTI selection expression if selection based on rate is not
        desired. It should be of the form: TIME < VAL or TIME IN [LOW:HI]
    extra_expr: str
        Extra filtering expression for evselect. e.g '&&(PI>4000)'.
        Note the &&.
    barycorr: bool
        Apply barycenter correction to the filtered event file
    region: bool
        If True, create an image from the filtered file and launch
        ds9 to create a region file
    
    """
    instr = kwargs.pop('instr', 'pn')
    gtiexpr = kwargs.pop('gtiexpr', None)
    barycorr = kwargs.pop('barycorr', False)
    extra_expr = kwargs.pop('extra_expr', '')
    region = kwargs.pop('region', False)
    if extra_expr != '' and '&&' not in extra_expr and '||' not in extra_expr:
        raise ValueError(('extra_expr has to contrain && or || '
                          'to connect to other expression'))


    if instr not in ['pn', 'mos1', 'mos2']:
        raise ValueError('instr need to be one of pn|mo1|mos2')
    cwd = os.getcwd()

    try:
        os.system(f'mkdir -p {obsid}/{instr}')
        os.chdir(f'{obsid}/{instr}')
        evt = f'../odf/{instr}.fits'
        filtered_evt = f'{instr}_filtered.fits'
        if not os.path.exists(evt):
            raise ValueError(f'No {instr}.fits found under odf; run process_xmm_obsid')

        # background flare filtering #
        foptions = {
            'pn' : ['IN [10000:12000]', 'EP', 4,  0.4],
            'mos': ['> 10000'         , 'EM', 12, 0.35]
        }

        cmd = (
            f"evselect table={evt}:EVENTS withrateset=yes rateset=tmp.rate " 
            f"expression='(PI {foptions[instr[:3]][0]})&&(PATTERN==0)&&"
            f"#XMMEA_{foptions[instr[:3]][1]}' "
            "makeratecolumn=yes maketimecolumn=yes timebinsize=100"
        )
        misc.run_cmd_line_tool(cmd, logfile='filter_xmm_evselect.log')

        # generate GTI
        expr = gtiexpr or f'RATE < {foptions[instr[:3]][3]}'
        cmd = ('tabgtigen table=tmp.rate gtiset=tmp.gti '
               f'expression="RATE<{foptions[instr[:3]][3]}"')
        misc.run_cmd_line_tool(cmd, logfile='filter_xmm_tabgtigen.log')

        # apply GTI
        expr = ('gti(tmp.gti,TIME)&&(PI IN [200:12000])&&(FLAG==0)&&'
                f'(PATTERN<={foptions[instr[:3]][2]}){extra_expr}')
        cmd = (f"evselect table={evt}:EVENTS withfilteredset=yes "
               f"filteredset={filtered_evt} expression='{expr}'")
        misc.run_cmd_line_tool(cmd, logfile='filter_xmm_evselect2.log')

        # barycenter corrections? #
        if barycorr:
            os.system(f'cp {filtered_evt} {instr}_filtered_nobary.fits'.format(instr))
            cmd = f'barycen table={filtered_evt}:EVENTS'
            misc.run_cmd_line_tool(cmd, logfile='filter_xmm_barycen.log')

        # region?
        if region:
            # create an image
            cmd = (f"evselect table={filtered_evt}:EVENTS withimageset=yes "
                   "imageset=tmp.img xcolumn=X ycolumn=Y")
            misc.run_cmd_line_tool(cmd, logfile='filter_xmm_image.log')

            # call ds9
            print('launching ds9')
            ret = os.system('ds9 tmp.img -log -zoom 2 -cmap heat')
            if ret != 0:
                raise RuntimeError('Failed launching ds9')

        os.chdir(cwd)
        print(f'{filtered_evt} created successfully')

    except Exception as exception: # pylint: disable=broad-exception-caught
        os.chdir(cwd)
        raise exception


def extract_xmm_spec(obsid: str, **kwargs):
    """Extract XMM pn or mos spectrum with xmm sas
    
    Run from top level containting obsid folder
    
    Parameters
    ----------
    obsid: str
        Obsid to be processed
    
    Keywords
    --------
    instr: str
        Instrument or instrument mode: pn|mos1|mos2
    regfile: str
        name of region file. It should under: obsid/{instr}/; where instr
        is pn|mos1|mos2
    extra_expr: str
        Extra filtering expression for evselect. e.g '&&gti("gtifile.gti",TIME)'.
        Note the &&.
    genrsp: bool
        Generate rmf and arf files.
    irun: int
        Run number if in parallel. Can be used to name outputs as spec_{irun}.*
    
    """
    # get keywords
    instr = kwargs.pop('instr', 'pn')
    regfile = kwargs.pop('regfile', 'ds9.reg')
    extra_expr = kwargs.pop('extra_expr', '')
    genrsp = kwargs.pop('genrsp', True)
    irun = kwargs.pop('irin', None)

    # check keywords
    if instr not in ['pn', 'mos1', 'mos2']:
        raise ValueError('instr need to be one of pn|mo1|mos2')

    if extra_expr != '' and '&&' not in extra_expr and '||' not in extra_expr:
        raise ValueError(('extra_expr has to contrain && or || '
                          'to connect to other expression'))

    out = 'spec'
    if irun is not None:
        out += f'_{irun}'
    cwd = os.getcwd()

    try:
        os.system(f'mkdir -p {obsid}/{instr}/spec')
        os.chdir(f'{obsid}/{instr}/spec')
        filtered_evt = f'../{instr}_filtered.fits'
        if not os.path.exists(filtered_evt):
            raise FileNotFoundError(f'{obsid}/{instr}/{instr}_filtered.fits not found')

        # read region file; expecting obsid/instr/ds9.reg
        regfile = f'../{regfile}'
        if not os.path.exists(regfile):
            raise FileNotFoundError(f'{regfile} not found.')
        selector = '(X,Y) IN '
        regions = ['', '']
        with open(regfile, encoding='utf8') as filep:
            for line in filep.readlines():
                if '(' in line:
                    idx = 1 if 'back' in line else 0
                    reg = f'{selector} {line.split("#")[0].rstrip()}'
                    regions[idx] += '' if regions[idx] == '' else ' || '
                    regions[idx] += f'{selector} {line.split("#")[0].rstrip()}'

        env = {
            'SAS_ODF': f'{cwd}/{obsid}/odf',
            'SAS_CCF': f'{cwd}/{obsid}/odf/ccf.cif',
        }


        # extract src and bgd spectra #
        maxchan = 20479 if instr == 'pn' else 11999
        labels = ['pha', 'bgd']
        for lab,reg in zip(labels, regions):
            cmd = (f'evselect table={filtered_evt} spectrumset={out}.{lab} '
                   f'expression="{reg}{extra_expr}" '
                   'energycolumn=PI spectralbinsize=5 withspecranges=yes '
                   f'specchannelmin=0 specchannelmax={maxchan}'
            )
            misc.run_cmd_line_tool(cmd, env, logfile=f'extract_xmm_spec_{lab}.log')
            # backscale
            cmd = f'backscale spectrumset={out}.{lab} badpixlocation={filtered_evt}'
            misc.run_cmd_line_tool(cmd, env, logfile=f'extract_xmm_spec_{lab}_backscale.log')

        if genrsp:
            # response
            cmd = f'rmfgen spectrumset={out}.pha rmfset={out}.rmf'
            misc.run_cmd_line_tool(cmd, env, logfile='extract_xmm_spec_rmf.log')

            # arf
            cmd = (f'arfgen spectrumset={out}.pha arfset={out}.arf withrmfset=yes '
                   f'rmfset={out}.rmf badpixlocation={filtered_evt} detmaptype=psf')
            misc.run_cmd_line_tool(cmd, env, logfile='extract_xmm_spec_arf.log')

            # group spectra
            cmd = (f'specgroup spectrumset={out}.pha rmfset={out}.rmf '
                   f'backgndset={out}.bgd arfset={out}.arf groupedset={out}.grp '
                   'minSN=6 oversample=3')
            misc.run_cmd_line_tool(cmd, env, logfile='extract_xmm_spec_grp.log')

        os.chdir(cwd)
        print(f'spectra {out}* created successfully')

    except Exception as exception: # pylint: disable=broad-exception-caught
        os.chdir(cwd)
        raise exception
