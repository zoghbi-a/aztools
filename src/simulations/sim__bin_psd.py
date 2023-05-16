#!/usr/bin/env python

"""Check LCurve.bin_psd"""

import argparse as argp
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.console import ProgressBar

import aztools as az

az.misc.set_fancy_plot(plt)


def bin_psd__1():
    """Simple powerlaw psd, no poisson noise, BINNING, no noise leak
    

    TEST: linear vs log binning
    SUMMARY:
        1- logavg does not do better than linear. The applied bias from Papadakis
            over-corrects the values when averaging a small number of frequencies. 
            It helps with lowest frequency when averaging >10 bins
    """
    npoints = 2**12
    deltat = 1.0
    mean = 100.0
    nsim = 200
    bins = [2, 5, 10, 20, 50]

    sim = az.SimLC(seed=37751)
    sim.add_model('powerlaw', [8e-3, -2])

    psd = []
    with ProgressBar(nsim) as pbar:
        for _ in range(nsim):
            sim.simulate(npoints, deltat, mean, norm='var')
            rpsd = az.LCurve.calculate_psd(sim.lcurve[1], deltat, 'var')

            # loop though bin sizes
            bpsd = []
            for _bin in bins:
                psd1 = az.LCurve.bin_psd(rpsd[0], rpsd[1],
                                         fqbin={'by_n':[_bin,1]}, logavg=False)[:3]
                psd2 = az.LCurve.bin_psd(rpsd[0], rpsd[1],
                                         fqbin={'by_n':[_bin,1]}, logavg=True)[:3]
                bpsd.append([psd1,psd2])

            psd.append(bpsd)
            pbar.update()

    nbins = len(bins)
    _,axs = plt.subplots(1, nbins, figsize=(15,4), sharex=True, sharey=True)

    for ibin in range(nbins):
        bpsd = np.array([_[ibin] for _ in psd])
        freq = bpsd[0,0,0]
        psdm = np.mean(bpsd[:,:,1], 0)

        axs[ibin].loglog(freq, psdm[0], color='C0', label='LIN')
        axs[ibin].loglog(freq, psdm[1], color='C1', label='LOG')
        axs[ibin].plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:],
                       color='C2', alpha=0.5, label='INP')
        if ibin==0:
            axs[ibin].legend()
        axs[ibin].set_title(f'bin:{bins[ibin]}')
    plt.tight_layout()
    os.system('mkdir -p png')
    plt.savefig('png/bin_psd__1.png')


def bin_psd__2():
    """Simple powerlaw psd, no poisson noise, BINNING, LEAK
    

    TEST: linear vs log binning, with leak/tapering
    SUMMARY:
        1- Similar to bin_psd__1. logavg is overall slightly better than linear at 
            the lowerst frequencies, but it is biased when num. of averaged freq is <~+10
        2- taper is always better as we saw in calculate_psd simulations.
    """
    npoints = 2**12
    deltat = 1.0
    mean = 100.0
    nsim = 200
    bins = [2, 10, 50]

    sim = az.SimLC(seed=3051)
    sim.add_model('powerlaw', [8e-4, -2.5])

    psd = []
    with ProgressBar(nsim) as pbar:
        for _ in range(nsim):
            sim.simulate(3*npoints, deltat, mean, norm='var')
            rpsd1 = az.LCurve.calculate_psd(sim.lcurve[1][:npoints], deltat, 'var')
            rpsd2 = az.LCurve.calculate_psd(sim.lcurve[1][:npoints], deltat, 'var', taper=True)

            # loop though bin sizes
            bpsd = []
            for _bin in bins:
                psd1 = az.LCurve.bin_psd(rpsd1[0], rpsd1[1],
                                         fqbin={'by_n':[_bin,1]}, logavg=False)[:3]
                psd2 = az.LCurve.bin_psd(rpsd1[0], rpsd1[1],
                                         fqbin={'by_n':[_bin,1]}, logavg=True)[:3]
                psd3 = az.LCurve.bin_psd(rpsd2[0], rpsd2[1],
                                         fqbin={'by_n':[_bin,1]}, logavg=False)[:3]
                psd4 = az.LCurve.bin_psd(rpsd2[0], rpsd2[1],
                                         fqbin={'by_n':[_bin,1]}, logavg=True)[:3]
                bpsd.append([psd1, psd2, psd3, psd4])

            psd.append(bpsd)
            pbar.update()

    nbins = len(bins)
    _,axs = plt.subplots(1, nbins, figsize=(15,4), sharex=True, sharey=True)

    for ibin in range(nbins):
        bpsd = np.array([_[ibin] for _ in psd])
        freq = bpsd[0,0,0]
        psdm = np.mean(bpsd[:,:,1], 0)

        axs[ibin].loglog(freq, psdm[0], color='C0', label='Lin-No-Taper')
        axs[ibin].loglog(freq, psdm[1], color='C1', label='Log-No-Taper')
        axs[ibin].loglog(freq, psdm[2], color='C2', label='Lin-Taper')
        axs[ibin].loglog(freq, psdm[3], color='C3', label='Log-Taper')
        axs[ibin].plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:],
                       color='C4', alpha=0.5, label='INP')
        if ibin==0:
            axs[ibin].legend()
        axs[ibin].set_title(f'bin:{bins[ibin]}')
    plt.tight_layout()
    os.system('mkdir -p png')
    plt.savefig('png/bin_psd__2.png')


def bin_psd__3():
    """Simple powerlaw psd, POISSON, BINNING, LEAK
    

    TEST: linear vs log binning, with leak/tapering, with noise
    SUMMARY:
        1- logavg and taper do the best
        2- logavg on noise doesn't matter a lot,
    """
    npoints = 2**12
    deltat = 1.0
    mean = 1000.0
    nsim = 200
    bins = [2, 10, 50]

    sim = az.SimLC(seed=12001)
    sim.add_model('powerlaw', [8e-4, -2.5])

    psd = []
    with ProgressBar(nsim) as pbar:
        for _ in range(nsim):
            sim.simulate(3*npoints, deltat, mean, norm='var')
            # use different sampling dt, so we reduce the noise a little bit
            xarr = np.random.poisson(sim.lcurve[1][:npoints]*500)/500
            xerr = (xarr/500)**0.5
            rpsd1 = az.LCurve.calculate_psd(xarr, deltat, 'var', rerr=xerr)
            rpsd2 = az.LCurve.calculate_psd(xarr, deltat, 'var', taper=True, rerr=xerr)

            # loop though bin sizes
            bpsd = []
            for _bin in bins:
                psd1 = az.LCurve.bin_psd(rpsd1[0], rpsd1[1], fqbin={'by_n':[_bin,1]},
                                         logavg=False, noise=rpsd1[2])
                psd2 = az.LCurve.bin_psd(rpsd1[0], rpsd1[1], fqbin={'by_n':[_bin,1]},
                                         logavg=True, noise=rpsd1[2])
                psd3 = az.LCurve.bin_psd(rpsd2[0], rpsd2[1], fqbin={'by_n':[_bin,1]},
                                         logavg=False, noise=rpsd2[2])
                psd4 = az.LCurve.bin_psd(rpsd2[0], rpsd2[1], fqbin={'by_n':[_bin,1]},
                                         logavg=True, noise=rpsd2[2])

                psd1 = np.vstack([psd1[:3], psd1[3]['noise']])
                psd2 = np.vstack([psd2[:3], psd2[3]['noise']])
                psd3 = np.vstack([psd3[:3], psd3[3]['noise']])
                psd4 = np.vstack([psd4[:3], psd4[3]['noise']])

                bpsd.append([psd1, psd2, psd3, psd4])

            psd.append(bpsd)
            pbar.update()

    nbins = len(bins)
    _,axs = plt.subplots(1, nbins, figsize=(11,4), sharex=True, sharey=True)

    for ibin in range(nbins):
        bpsd = np.array([_[ibin] for _ in psd])
        freq = bpsd[0,0,0]
        psdm = np.mean(bpsd[:,:,1], 0)
        psdn = np.mean(bpsd[:,:,3], 0)

        axs[ibin].loglog(freq, psdm[0], color='C0', label='Lin-No-Taper')
        axs[ibin].loglog(freq, psdm[1], color='C1', label='Log-No-Taper')
        axs[ibin].loglog(freq, psdm[2], color='C2', label='Lin-Taper')
        axs[ibin].loglog(freq, psdm[3], color='C3', label='Log-Taper')
        axs[ibin].loglog(freq, psdn[0], '-.', color='C0')
        axs[ibin].loglog(freq, psdn[1], '-.', color='C1')
        axs[ibin].loglog(freq, psdn[2], '-.', color='C2')
        axs[ibin].loglog(freq, psdn[3], '-.', color='C3')
        axs[ibin].plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:],
                       color='C4', alpha=0.5, label='INP')

        if ibin==0:
            axs[ibin].legend()
        axs[ibin].set_title(f'bin:{bins[ibin]}')
    plt.tight_layout()
    os.system('mkdir -p png')
    plt.savefig('png/bin_psd__3.png')


def bin_psd__4():
    """Simple powerlaw psd, POISSON, BINNING, LEAK
    

    TEST: formula error
    SUMMARY:
        1. Formula works. It just need to be compared to the percentiles of the
            distribution rather than the std-dev
        2. The linear binning has a large bias at the lowest frequencies. logavg does better,
            but not entirely
    """
    npoints = 2**12
    deltat = 1.0
    mean = 1000.0
    nsim = 200
    bins = [2, 10, 50]

    sim = az.SimLC(seed=199051)
    sim.add_model('powerlaw', [1.2e-3, -2.5])

    psd = []
    with ProgressBar(nsim) as pbar:
        for _ in range(nsim):
            sim.simulate(3*npoints, deltat, mean, norm='var')
            # use different sampling dt, so we reduce the noise a little bit
            xarr = np.random.poisson(sim.lcurve[1][:npoints]*500)/500
            xerr = (xarr/500)**0.5
            rpsd = az.LCurve.calculate_psd(xarr, deltat, 'var', taper=True, rerr=xerr)

            # loop though bin sizes
            bpsd = []
            for _bin in bins:
                psd1 = az.LCurve.bin_psd(rpsd[0], rpsd[1], fqbin={'by_n':[_bin,1]},
                                         logavg=False, noise=rpsd[2])
                psd2 = az.LCurve.bin_psd(rpsd[0], rpsd[1], fqbin={'by_n':[_bin,1]},
                                         logavg=True, noise=rpsd[2])

                psd1 = np.vstack([psd1[:3], psd1[3]['noise']])
                psd2 = np.vstack([psd2[:3], psd2[3]['noise']])

                bpsd.append([psd1, psd2])

            psd.append(bpsd)
            pbar.update()

    nbins = len(bins)
    _,axs = plt.subplots(2, nbins, figsize=(11,8), sharex=True, sharey=True)

    for ibin in range(nbins):
        bpsd = np.array([_[ibin] for _ in psd])
        freq = bpsd[0,0,0]
        psdm = np.mean(bpsd[:,:,1], 0)
        psde = np.mean(bpsd[:,:,2], 0)
        psds = np.std(bpsd[:,:,1], 0)
        psdp = np.percentile(bpsd[:,:,1], [16,100-16], 0)

        axs[0,ibin].loglog(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:],
                       color='C4', alpha=0.5, label='INP')
        axs[0,ibin].errorbar(freq, psdm[0], psds[0], color='C0', lw=0.4)
        axs[0,ibin].fill_between(freq, psdm[0]-psde[0], psdm[0]+psde[0], color='C1', alpha=0.5)
        axs[0,ibin].fill_between(freq, psdp[0,0], psdp[1,0], color='C2', alpha=0.6)
        axs[0,ibin].set_title(f'bin:{bins[ibin]}')

        axs[1,ibin].loglog(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:],
                       color='C4', alpha=0.5, label='INP')
        axs[1,ibin].errorbar(freq, psdm[1], psds[1], color='C0', lw=0.4)
        axs[1,ibin].fill_between(freq, psdm[1]-psde[1], psdm[1]+psde[1], color='C1', alpha=0.5)
        axs[1,ibin].fill_between(freq, psdp[0,1], psdp[1,1], color='C2', alpha=0.6)


    plt.tight_layout()
    os.system('mkdir -p png')
    plt.savefig('png/bin_psd__4.png')


if __name__ == '__main__':

    pars = argp.ArgumentParser(
        description="""
        Run simulations for the LCurve functionality
        """,
        formatter_class=argp.ArgumentDefaultsHelpFormatter
    )



    pars.add_argument('--sim_1', action='store_true', default=False,
                      help="psd binning. Test log/linear averaging.")
    pars.add_argument('--sim_2', action='store_true', default=False,
                      help="psd binning. Test log/linear averaging, with leak/taper.")
    pars.add_argument('--sim_3', action='store_true', default=False,
                      help="psd binning. Test log/linear averaging, with leak/taper, and noise.")
    pars.add_argument('--sim_4', action='store_true', default=False,
                      help="psd binning. test formula errors")

    # process arguments #
    args = pars.parse_args()


    # log/linear averaging
    if args.sim_1:
        bin_psd__1()

    # log/linear averaging, with leak/taper.
    if args.sim_2:
        bin_psd__2()

    # log/linear averaging, with leak/taper and noise
    if args.sim_3:
        bin_psd__3()

    # formula errors
    if args.sim_4:
        bin_psd__4()
