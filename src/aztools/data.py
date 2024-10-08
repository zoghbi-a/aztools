"""Data Tools"""

import glob
import os

import numpy as np
from astropy.io import fits

from .misc import hsp, parallelize, run_cmd_line_tool

__all__ = [
    'process_nicer_obsid', 'process_nicer_obsids',
    'process_nustar_obsid', 'process_nustar_obsids',
    'process_suzaku_obsid', 'process_suzaku_obsids',
    'process_xmm_obsid', 'process_xmm_obsids',
    'filter_xmm_obsid', 'filter_xmm_obsids',
    'extract_xmm_spec', 'extract_xmm_specs',
    'extract_xmm_lc', 'extract_xmm_lcs',
    'extract_nustar_spec', 'extract_nustar_specs',
    'extract_nustar_lc', 'extract_nustar_lcs',
    'extract_nicer_spec', 'extract_nicer_specs'
]


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
    if hsp is None:
        raise ImportError('process_nicer_obsid depends on heasoftpy. Install it first')

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
process_nicer_obsids = parallelize(process_nicer_obsid, use_irun=False)


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
    if hsp is None:
        raise ImportError('process_nustar_obsid depends on heasoftpy. Install it first')

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
process_nustar_obsids = parallelize(process_nustar_obsid, use_irun=False)


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
    if hsp is None:
        raise ImportError('process_suzaku_obsid depends on heasoftpy. Install it first')

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
process_suzaku_obsids = parallelize(process_suzaku_obsid, use_irun=False)


def _run_sas_cmd(cmd, *args, **kwargs):
    """prefix run_cmd_line_tool with sas initialization"""

    if 'SAS_DIR' not in os.environ:
        raise KeyError('SAS_DIR is not defined')

    pre_cmd = 'source $SAS_DIR/sas-setup.sh; '
    return run_cmd_line_tool(pre_cmd + cmd, *args, **kwargs)


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
        Instrument or instrument mode: pn|mos|rgs|omi|omf|omg
    odfingest: bool
        run odfingest? This should be True for the first run
        and False subsequently
    Any parameters to be passed to the reduction pipeline
    
    
    """
    odfingest = kwargs.pop('odfingest', True)

    # defaults
    instr = kwargs.pop('instr', 'pn')
    if instr not in ['pn', 'mos', 'rgs', 'omi', 'omf', 'omg']:
        raise ValueError('instr need to be one of pn|mos|rgs|omi|omf|omg')
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
            _run_sas_cmd(cmd, env, logfile='processing_xmm_cifbuild.log')

        env['SAS_CCF'] = f'{os.getcwd()}/ccf.cif'

        if len(glob.glob('*.SAS')) > 0 and odfingest:
            os.system('rm *.SAS')
        if odfingest:
            cmd = f'odfingest odfdir={os.getcwd()} outdir={os.getcwd()}'
            _run_sas_cmd(cmd, env, logfile='processing_xmm_odfingest.log')


        # prepare the command
        if instr == 'pn':
            cmd = 'epproc'
        elif instr == 'mos':
            cmd = 'emproc'
        elif instr == 'rgs':
            kwargs.setdefault('orders', '"1 2"')
            kwargs.setdefault('bkgcorrect', 'no')
            kwargs.setdefault('withmlambdacolumn', 'yes')
            cmd = 'rgsproc'
        elif instr == 'omi':
            cmd = 'omiproc'
        elif instr == 'omf':
            cmd = 'omiproc'
        elif instr == 'omg':
            cmd = 'omgproc'
        else:
            raise ValueError('instr needs to be pn|om|rgs|om')

        # the following will raise RuntimeError if the task fails
        cmd = f'{cmd} {" ".join([f"{par}={val}" for par,val in kwargs.items()])}'
        _run_sas_cmd(cmd, env, logfile=f'processing_xmm_{instr}.log')

        # post run extra tasks
        if instr == 'pn':
            evt = glob.glob('*EPN_S*Evts*')
            if len(evt) != 1:
                raise ValueError('Found >1 event files for pn')
            os.system(f'mv {evt[0]} {instr}.fits')
            print(f'{obsid}:{instr}.fits created successfully')

        if instr == 'mos':
            for subi in [1, 2]:
                evt = glob.glob(f'*EMOS{subi}_S*Evts*')
                if len(evt) != 1:
                    raise ValueError(f'Found >1 event files for mos-{subi}')
                os.system(f'mv {evt[0]} {instr}{subi}.fits')
                print(f'{obsid}:{instr}{subi}.fits created successfully')

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
            _run_sas_cmd(cmd, env, logfile='processing_xmm_rgscombine.log')
            print(f'{obsid}:rgs spectra created successfully')

        os.chdir(cwd)

    except Exception as exception: # pylint: disable=broad-exception-caught
        os.chdir(cwd)
        raise exception

# parallel version of process_xmm_obsid
process_xmm_obsids = parallelize(process_xmm_obsid, use_irun=False)


def filter_xmm_obsid(obsid: str, **kwargs):
    """Filter XMM pn or mos obsid with xmm sas
    
    Run from top level containting obsid folder.
    This assumes process_xmm_obsid was called first.
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
    use_raw: bool
        If True, use RAWX, RAWY instead of X, Y
    
    """
    instr = kwargs.pop('instr', 'pn')
    gtiexpr = kwargs.pop('gtiexpr', None)
    barycorr = kwargs.pop('barycorr', False)
    extra_expr = kwargs.pop('extra_expr', '')
    region = kwargs.pop('region', False)
    use_raw = kwargs.pop('use_raw', False)
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
        _run_sas_cmd(cmd, logfile='filter_xmm_evselect.log')

        # generate GTI
        expr = gtiexpr or f'RATE < {foptions[instr[:3]][3]}'
        cmd = ('tabgtigen table=tmp.rate gtiset=tmp.gti '
               f'expression="RATE<{foptions[instr[:3]][3]}"')
        _run_sas_cmd(cmd, logfile='filter_xmm_tabgtigen.log')

        # apply GTI
        expr = ('gti(tmp.gti,TIME)&&(PI IN [200:12000])&&(FLAG==0)&&'
                f'(PATTERN<={foptions[instr[:3]][2]}){extra_expr}')
        cmd = (f"evselect table={evt}:EVENTS withfilteredset=yes "
               f"filteredset={filtered_evt} expression='{expr}'")
        _run_sas_cmd(cmd, logfile='filter_xmm_evselect2.log')

        # barycenter corrections? #
        if barycorr:
            os.system(f'cp {filtered_evt} {instr}_filtered_nobary.fits'.format(instr))
            cmd = f'barycen table={filtered_evt}:EVENTS'
            _run_sas_cmd(cmd, logfile='filter_xmm_barycen.log')

        # region?
        if region:
            # create an image
            pref = 'RAW' if use_raw else ''
            cmd = (f"evselect table={filtered_evt}:EVENTS withimageset=yes "
                   f"imageset=tmp.img xcolumn={pref}X ycolumn={pref}Y")
            _run_sas_cmd(cmd, logfile='filter_xmm_image.log')

            # call ds9
            print(f'{obsid}: launching ds9')
            ret = os.system('ds9 tmp.img -log -zoom 2 -cmap heat')
            if ret != 0:
                raise RuntimeError('Failed launching ds9')

        os.chdir(cwd)
        print(f'{obsid}:{filtered_evt} created successfully')

    except Exception as exception: # pylint: disable=broad-exception-caught
        os.chdir(cwd)
        raise exception

# parallel version of filter_xmm_obsid
filter_xmm_obsids = parallelize(filter_xmm_obsid, use_irun=False)


def extract_xmm_spec(obsid: str, **kwargs):
    """Extract XMM pn or mos spectrum with xmm sas
    
    Run from top level containting obsid folder
    This assumes filter_xmm_obsid was called first.
    
    Parameters
    ----------
    obsid: str
        Obsid to be processed; or {obsid}:{label} where output
        is spec_{label}*
    
    Keywords
    --------
    instr: str
        Instrument or instrument mode: pn|mos1|mos2
    regfile: str
        name of region file. It should under: obsid/{instr}/; where instr
        is pn|mos1|mos2. Default is ds9.reg.
    use_raw: bool
        if True, region files uses RAWX, RAWY instead of X, Y; used in timing mode
    extra_expr: str
        Extra filtering expression for evselect. e.g '&&gti("gtifile.gti",TIME)'.
        Note the &&.
    genrsp: bool
        Generate rmf and arf files.
    irun: int
        name suffix, so the output is spec_{irun}*
    
    """
    # get keywords
    instr = kwargs.pop('instr', 'pn')
    regfile = kwargs.pop('regfile', 'ds9.reg')
    use_raw = kwargs.pop('use_raw', False)
    extra_expr = kwargs.pop('extra_expr', '')
    genrsp = kwargs.pop('genrsp', True)

    # check keywords
    if instr not in ['pn', 'mos1', 'mos2']:
        raise ValueError('instr need to be one of pn|mo1|mos2')

    if extra_expr != '' and '&&' not in extra_expr and '||' not in extra_expr:
        raise ValueError(('extra_expr has to contrain && or || '
                          'to connect to other expression'))

    prefix = 'spec'
    irun = kwargs.get('irun', None)
    if irun is not None:
        prefix += f'_{irun}'
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
        selector = '(RAWX,RAWY) IN ' if use_raw else '(X,Y) IN '
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
            cmd = (f'evselect table={filtered_evt} spectrumset={prefix}.{lab} '
                   f'expression="{reg}{extra_expr}" '
                   'energycolumn=PI spectralbinsize=5 withspecranges=yes '
                   f'specchannelmin=0 specchannelmax={maxchan}'
            )
            _run_sas_cmd(cmd, env, logfile=f'extract_xmm_spec_{lab}.log')
            # backscale
            cmd = f'backscale spectrumset={prefix}.{lab} badpixlocation={filtered_evt}'
            _run_sas_cmd(cmd, env, logfile=f'extract_xmm_spec_{lab}_backscale.log')

        if genrsp:
            # response
            cmd = f'rmfgen spectrumset={prefix}.pha rmfset={prefix}.rmf'
            _run_sas_cmd(cmd, env, logfile='extract_xmm_spec_rmf.log')

            # arf
            cmd = (f'arfgen spectrumset={prefix}.pha arfset={prefix}.arf withrmfset=yes '
                   f'rmfset={prefix}.rmf badpixlocation={filtered_evt} detmaptype=psf')
            _run_sas_cmd(cmd, env, logfile='extract_xmm_spec_arf.log')

            # group spectra
            cmd = (f'specgroup spectrumset={prefix}.pha rmfset={prefix}.rmf '
                   f'backgndset={prefix}.bgd arfset={prefix}.arf groupedset={prefix}.grp '
                   'minSN=6 oversample=3')
            _run_sas_cmd(cmd, env, logfile='extract_xmm_spec_grp.log')

        os.chdir(cwd)
        print(f'spectra {obsid}:{prefix}* created successfully')

    except Exception as exception: # pylint: disable=broad-exception-caught
        os.chdir(cwd)
        raise exception

# parallel version of extract_xmm_spec
extract_xmm_specs = parallelize(extract_xmm_spec, use_irun=True)


def extract_xmm_lc(obsid: str, **kwargs):
    """Extract XMM pn or mos light curves with xmm sas
    
    Run from top level containting obsid folder
    This assumes @filter_xmm_obsid was called first.
    
    Parameters
    ----------
    obsid: str
        Obsid to be processed; or {obsid}:{label} where output
        is lc_{label}*
    
    Keywords
    --------
    instr: str
        Instrument or instrument mode: pn|mos1|mos2
    ebins: str
        Space-separated string of the energy bin boundaries in keV. Default is '0.3 10'
    tbin: float
        The time bin, negative means 2**tbin
    regfile: str
        name of region file. It should under: obsid/{instr}/; where instr
        is pn|mos1|mos2. Default is ds9.reg.
    use_raw: bool
        if True, use RAWX, RAWY instead of X,Y (used in timing mode)
    extra_expr: str
        Extra filtering expression for evselect. e.g '&&gti("gtifile.gti",TIME)'.
        Note the &&.
    lccorr: bool
        Run epiclccorr. Default is True.
    outdir: str
        output folder name under {obsid}/{instr}/. Default is lc.
    irun: int
        name suffix so the output file is lc_{irun}*
    
    """
    # get keywords
    instr = kwargs.pop('instr', 'pn')
    ebins = kwargs.pop('ebins', '0.3 10')
    tbin = kwargs.pop('tbin', 1.0)
    regfile = kwargs.pop('regfile', 'ds9.reg')
    use_raw = kwargs.pop('use_raw', False)
    extra_expr = kwargs.pop('extra_expr', '')
    lccorr = kwargs.pop('lccorr', True)
    outdir = kwargs.pop('outdir', 'lc')
    irun = kwargs.get('irun', None)

    # check keywords
    if instr not in ['pn', 'mos1', 'mos2']:
        raise ValueError('instr need to be one of pn|mo1|mos2')

    ebins = np.array(ebins.split(), np.double) * 1000
    ebins = [list(ebin) for ebin in zip(ebins[:-1], ebins[1:])]

    if tbin < 0:
        tbin = 2**tbin

    if extra_expr != '' and '&&' not in extra_expr and '||' not in extra_expr:
        raise ValueError(('extra_expr has to contrain && or || '
                          'to connect to other expression'))

    prefix = f'lc_{tbin:03g}'
    if irun is not None:
        prefix += f'_{irun}'
    cwd = os.getcwd()

    try:
        os.system(f'mkdir -p {obsid}/{instr}/{outdir}')
        os.chdir(f'{obsid}/{instr}/{outdir}')
        filtered_evt = f'../{instr}_filtered.fits'
        if not os.path.exists(filtered_evt):
            raise FileNotFoundError(f'{obsid}/{instr}/{instr}_filtered.fits not found')

        # read region file; expecting obsid/instr/ds9.reg
        regfile = f'../{regfile}'
        if not os.path.exists(regfile):
            raise FileNotFoundError(f'{regfile} not found.')
        selector = '(RAWX,RAWY) IN ' if use_raw else '(X,Y) IN '
        regions = ['', '']
        with open(regfile, encoding='utf8') as filep:
            for line in filep.readlines():
                if '(' in line:
                    idx = 1 if 'back' in line else 0
                    regions[idx] += '' if regions[idx] == '' else ' || '
                    regions[idx] += f'{selector} {line.split("#")[0].rstrip()}'

        env = {
            'SAS_ODF': f'{cwd}/{obsid}/odf',
            'SAS_CCF': f'{cwd}/{obsid}/odf/ccf.cif',
        }


        # extract src and bgd spectra #
        for ibin,ebin in enumerate(ebins):

            # names #
            src = f'{prefix}_e{ibin+1}__s.fits'
            bgd = f'{prefix}_e{ibin+1}__b.fits'
            smb = f'{prefix}_e{ibin+1}.fits'

            # extract src lc
            expr = f'PI IN [{ebin[0]}:{ebin[1]}) && {regions[0]} {extra_expr}'
            cmd = (f'evselect table={filtered_evt}:EVENTS withrateset=yes '
                   f'rateset={src} expression="{expr}" makeratecolumn=yes '
                   f'maketimecolumn=yes timebinsize={tbin}')
            _run_sas_cmd(cmd, env, logfile='extract_xmm_lc_src.log')


            # extract bgd lc
            expr = f'PI IN [{ebin[0]}:{ebin[1]}) && {regions[1]} {extra_expr}'
            cmd = (f'evselect table={filtered_evt}:EVENTS withrateset=yes '
                   f'rateset={bgd} expression="{expr}" makeratecolumn=yes '
                   f'maketimecolumn=yes timebinsize={tbin}')
            _run_sas_cmd(cmd, env, logfile='extract_xmm_lc_bgd.log')


            # correct lc
            if lccorr:
                cmd = (f'epiclccorr srctslist={src} eventlist={filtered_evt} outset={smb} '
                       f'withbkgset=yes bkgtslist={bgd} applyabsolutecorrections=no')
                _run_sas_cmd(cmd, env, logfile='extract_xmm_lc_corr.log')

        os.chdir(cwd)
        print(f'Light cruves {obsid}:{prefix}* created successfully')

    except Exception as exception: # pylint: disable=broad-exception-caught
        os.chdir(cwd)
        raise exception

# parallel version of extract_xmm_lc
extract_xmm_lcs = parallelize(extract_xmm_lc, use_irun=True)


def extract_nustar_spec(obsid: str, **kwargs):
    """Extract NuSTAR spectra for obsid with nuproducts
    
    Run from top level containting obsid folder
    
    Parameters
    ----------
    obsid: str
        Obsid to be processed; or {obsid}:{label} where output
        is spec_{label}*
    
    Keywords
    --------
    processed_obsid: str
        The name of the processed obsid folder. Default: {obsid}_p
    irun: int
        name suffix so the output is spec_{irun}*
    
    Any parameters to be passed to the reduction pipeline
    
    Return
    ------
    0 if succesful, and a heasoft error code otherwise
    
    """
    if hsp is None:
        raise ImportError('extract_nustar_spec depends on heasoftpy. Install it first')

    processed_obsid = kwargs.pop('processed_obsid', None)
    irun = kwargs.pop('irun', None)

    prefix = 'spec'
    if irun is not None:
        prefix += f'_{irun}'
    if processed_obsid is None:
        processed_obsid = f'{obsid}_p'


    outdir = f'{processed_obsid}/spec'
    os.system(f'mkdir -p {outdir}')

    # get ra and dec of the object
    evtfile = f'{processed_obsid}/event_cl/nu{obsid}A01_cl.evt'
    if not os.path.exists(evtfile):
        raise ValueError(f'No event file {evtfile} found.')
    with fits.open(evtfile) as filep:
        obj_ra  = filep['events'].header['ra_obj'] # pylint: disable=no-member
        obj_dec = filep['events'].header['dec_obj'] # pylint: disable=no-member


    # defaults
    in_pars = {
        'indir'        : f'{processed_obsid}/event_cl',
        'steminputs'   : f'nu{obsid}',
        'outdir'       : f'{outdir}',
        'srcregionfile': 'DEFAULT',
        'bkgregionfile': 'DEFAULT',
        'srcra'        : obj_ra,
        'srcdec'       : obj_dec,
        'srcradius'    : 150 / 2.46,
        'bkgextract'   : 'yes',
        'bkgra'        : obj_ra,
        'bkgdec'       : obj_dec,
        'bkgradius1'   : 180 / 2.46,
        'bkgradius2'   : 320 / 2.46,
        'lcfile'       : 'none',
        'phafile'      : 'DEFAULT',
        'bkgphafile'   : 'DEFAULT',
        'xcolf'        : 'X',
        'ycolf'        : 'Y',
        'runbackscale' : 'yes',
        'runmkarf'     : 'yes',
        'runmkrmf'     : 'yes',
        'rungrppha'    : 'yes',
        'grpmincounts' : 20,
        'clobber'      : 'yes',
        'noprompt'   : True,
    }
    # update input with given parameter keywords
    in_pars.update(**kwargs)


    # get the spectra for the two instruments
    for instr in ['A', 'B']:
        sfiles = glob.glob(f'{outdir}/{prefix}_{instr.lower()}_sr.???')
        if len(sfiles) == 3:
            # nothing to do, spectra alreay exist
            continue
        lpars = {
            'instrument': f'FPM{instr}',
            'stemout'   : f'{prefix}_{instr.lower()}',
            'imagefile' : f'{processed_obsid}/spec/image_{instr.lower()}.fits',
        }
        in_pars.update(**lpars)
        with hsp.utils.local_pfiles_context():
            out = hsp.nuproducts(**in_pars) # pylint: disable=no-member

        if out.returncode == 0:
            print(f'spectra sucessfully extracted for {obsid}:{instr.lower()}!')
        else:
            logfile = f'extract_nustar_spec_{obsid}-{instr.lower()}.log'
            print(f'ERROR processing {obsid}; Writing log to {logfile}')
            with open(logfile, 'w', encoding='utf8') as filep:
                filep.write(str(out))
            return out.returncode

    return 0

# parallel version of extract_nustar_spec
extract_nustar_specs = parallelize(extract_nustar_spec, use_irun=True)


def extract_nustar_lc(obsid: str, **kwargs):
    """Extract NuSTAR light curve for obsid with nuproducts
    
    Run from top level containting obsid folder
    
    Parameters
    ----------
    obsid: str
        Obsid to be processed; or {obsid}:{label} where output
        is lc_{label}*
    
    Keywords
    --------
    processed_obsid: str
        The name of the processed obsid folder. Default: {obsid}_p
    ebins: str
        Space-separated string of the energy bin boundaries in keV. Default is '3 79'
    tbin: float
        The time bin, negative means 2**tbin
    lccorr: bool
        Correct light curves. Default is True.
    barycorr: bool
        Barycenter the light curve (assume gti if used is bary-centered). Default is False.
    outdir: str
        output folder name under {processed_obsid}/. Default is lc.
    irun: int or str
        name suffix so that the output is lc_{irun}*
    
    
    Any parameters to be passed to the reduction nuproduct. e.g usrgtifile="somefile.gti"
    
    Return
    ------
    0 if succesful, and a heasoft error code otherwise
    
    """
    if hsp is None:
        raise ImportError('extract_nustar_lc depends on heasoftpy. Install it first')

    processed_obsid = kwargs.pop('processed_obsid', None)
    ebins = kwargs.pop('ebins', '3 79')
    tbin = kwargs.pop('tbin', 1.0)
    lccorr = kwargs.pop('lccorr', True)
    barycorr = kwargs.pop('barycorr', False)
    outdir = kwargs.pop('outdir', 'lc')
    irun = kwargs.get('irun', None)


    prefix = 'lc'
    if irun is not None:
        prefix += f'_{irun}'
    if processed_obsid is None:
        processed_obsid = f'{obsid}_p'


    outdir = f'{processed_obsid}/{outdir}'
    os.system(f'mkdir -p {outdir}')


    if tbin < 0:
        tbin = 2**tbin

    ebins = np.array(ebins.split(), np.double)
    nbins = len(ebins) - 1
    # convert energies to channel number #
    def conv(ene):
        return int(np.floor((ene-1.6)/0.04))
    chans = [ [conv(ebins[idx]), conv(ebins[idx+1])-1] for idx in range(nbins) ]
    enegs = [ [ebins[idx],ebins[idx+1]] for idx in range(nbins) ]
    np.savez(f'{outdir}/energy_{tbin:03g}.npz', en=enegs, chans=chans)


    # get ra and dec of the object
    evtfile = f'{processed_obsid}/event_cl/nu{obsid}A01_cl.evt'
    if not os.path.exists(evtfile):
        raise ValueError(f'No event file {evtfile} found.')
    with fits.open(evtfile) as filep:
        obj_ra  = filep['events'].header['ra_obj'] # pylint: disable=no-member
        obj_dec = filep['events'].header['dec_obj'] # pylint: disable=no-member


    # do we need barycorr?
    if barycorr:
        bary = {
            'barycorr' : 'yes',
            'srcra_barycorr'  : obj_ra,
            'srcdec_barycorr' : obj_dec
        }
    else:
        bary = {'barycorr': 'no'}

    # defaults
    in_pars = {
        'indir'        : f'{processed_obsid}/event_cl',
        'steminputs'   : f'nu{obsid}',
        'outdir'       : outdir,
        'srcregionfile': 'DEFAULT',
        'bkgregionfile': 'DEFAULT',
        'srcra'        : obj_ra,
        'srcdec'       : obj_dec,
        'srcradius'    : 150 / 2.46,
        'bkgextract'   : 'yes',
        'bkgra'        : obj_ra,
        'bkgdec'       : obj_dec,
        'bkgradius1'   : 180 / 2.46,
        'bkgradius2'   : 320 / 2.46,
        'imagefile'    : 'none',
        'correctlc'    : lccorr,
        'runbackscale' : 'yes',
        'binsize'      : tbin,
        'runmkarf'     : 'no',
        'runmkrmf'     : 'no',

        'clobber'      : 'yes',
        'noprompt'   : True,
    }
    # update input with given parameter keywords
    in_pars.update(**bary)
    in_pars.update(**kwargs)


    # loop through the enegies
    for ien in range(nbins):

        # get the spectra for the two instruments
        for instr in ['A', 'B']:
            stemout = f'{prefix}_{instr.lower()}_e{ien+1}'
            lpars = {
                'instrument': f'FPM{instr}',
                'stemout'   : stemout,
                'pilow'     : chans[ien][0],
                'pihigh'    : chans[ien][1],
                'lcenergy'  : np.max([(enegs[ien][0] + enegs[ien][1])/2., 3.1]),
            }
            if barycorr:
                lpars['orbitfile'] = glob.glob(f'{processed_obsid}/event_cl/*{instr}*orb*')[0]
            in_pars.update(**lpars)
            try:
                with hsp.utils.local_pfiles_context():
                    out = hsp.nuproducts(**in_pars) # pylint: disable=no-member
                    if out.returncode != 0:
                        raise RuntimeError(str(out))

                    # get backscale
                    with fits.open(f'{outdir}/{stemout}_sr.pha') as filep:
                        s_backscale = filep['SPECTRUM'].header['BACKSCAL'] # pylint: disable=no-member
                    with fits.open(f'{outdir}/{stemout}_bk.pha') as filep:
                        b_backscale = filep['SPECTRUM'].header['BACKSCAL'] # pylint: disable=no-member
                    backscale = s_backscale/b_backscale
                    # background subtraction
                    out = hsp.lcmath(infile=f'{outdir}/{stemout}_sr.lc', # pylint: disable=no-member
                                     bgfile=f'{outdir}/{stemout}_bk.lc',
                                     outfile=f'{outdir}/{stemout}.lc',
                                     multi=1.0, multb=backscale, addsubr='no')
                    if out.returncode != 0:
                        raise RuntimeError(str(out))

                print(f'light curve sucessfully extracted for {obsid}:{instr}-e{ien+1}!')
            except RuntimeError as exception:
                logfile = f'extract_nustar_lc_{obsid}-{instr}-e{ien+1}.log'
                print(f'ERROR processing {obsid}; Writing log to {logfile}')
                with open(logfile, 'w', encoding='utf8') as filep:
                    filep.write(str(exception))
                return -1

    return 0

# parallel version of extract_nustar_lc
extract_nustar_lcs = parallelize(extract_nustar_lc, use_irun=True)


def extract_nicer_spec(obsid: str, **kwargs):
    """Extract NICER spectra for obsid with nicerl3-spect
    
    Run from top level containting obsid folder
    
    Parameters
    ----------
    obsid: str
        Obsid to be processed; or {obsid}:{label} where output
        is spec_{label}*
    
    Keywords
    --------
    irun: int or str
        name suffix so the output is spec_{irun}*
    Any parameters to be passed to the reduction pipeline
    
    Return
    ------
    0 if succesful, and a heasoft error code otherwise
    
    """
    if hsp is None:
        raise ImportError('extract_nicer_spec depends on heasoftpy. Install it first')

    irun = kwargs.get('irun', None)
    prefix = 'spec'
    if irun is not None:
        prefix += f'_{irun}'
    processed_obsid = obsid


    outdir = f'{processed_obsid}/spec'
    os.system(f'mkdir -p {outdir}')


    # defaults
    in_pars = {
        'indir'        : obsid,
        'phafile'      : f'{outdir}/{prefix}.pha',
        'rmffile'      : f'{outdir}/{prefix}.rmf',
        'arffile'      : f'{outdir}/{prefix}.arf',
        'grouptype'    : 'NONE',
        'loadfile'     : f'{outdir}/{prefix}.xcm',
        'bkgformat'    : 'file',
        'clobber'      : 'yes',
        'noprompt'   : True,
    }
    bg_models = {
        'scorpeon': 'sc',
        '3c50'    : '3c',
        'sw'      : 'sw'
    }

    # update input with given parameter keywords
    in_pars.update(**kwargs)

    if len(glob.glob(f'{outdir}/{prefix}*')) == 6:
        print('Files already exist. Nothing to do!')
        return 0

    incr = 'no'
    for bname, blabel in bg_models.items():

        bgfile = f'{prefix}_{blabel}.bgd'
        bpars = {
            'bkgmodeltype': bname,
            'bkgfile': f'{outdir}/{bgfile}',
            'incremental': incr,
        }
        in_pars.update(**bpars)
        out = hsp.nicerl3_spect(**in_pars) # pylint: disable=no-member
        if out.returncode == 0:
            print(f'{obsid}:{blabel} spectra extracted sucessfully!')
            incr = 'yes'

            with fits.open(f'{outdir}/{prefix}.pha') as filep:
                filep['spectrum'].header['backfile'] = bgfile
                filep['spectrum'].header['respfile'] = f'{prefix}.rmf'
                filep['spectrum'].header['arffile']  = f'{prefix}.arf'
                filep.writeto(f'{outdir}/{prefix}_{blabel}.pha', overwrite=True)

        else:
            logfile = f'extract_nicer_spec_{obsid}_{blabel}.log'
            print(f'ERROR provessing {obsid}; Writing log to {logfile}')
            with open(logfile, 'w', encoding='utf8') as filep:
                filep.write(str(out))
            return -1

    return 0

# parallel version of extract_nustar_lc
extract_nicer_specs = parallelize(extract_nicer_spec, use_irun=True)
