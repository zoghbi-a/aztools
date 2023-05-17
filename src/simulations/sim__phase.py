#!/usr/bin/env python

"""Check LCurve.bin_psd"""

import argparse as argp
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.console import ProgressBar

import aztools as az

az.misc.set_fancy_plot(plt)


def phase__1():
    """Constant phase lag, no binning, no noise, RED NOISE LEAK

    TEST: phases from segments without noise, i.e with leak
    SUMMARY:
        1. taper helps alot! both in the scatter and bias
    """
    npoints = 2**12
    deltat = 1.0
    mean = 100.0
    lag  = 0.5
    nsim = 200

    sim = az.SimLC(seed=463284)
    sim.add_model('powerlaw', [1e-2, -2])
    sim.add_model('constant', [lag], lag=True)

    lag = []
    with ProgressBar(nsim) as pbar:
        for _ in range(nsim):
            sim.simulate(npoints*4, deltat, mean, norm='var')
            sim.apply_lag(phase=True)

            xarr = sim.lcurve[1][:npoints]
            yarr = sim.lcurve[2][:npoints]
            lag1 = az.LCurve.calculate_lag(yarr, xarr, deltat,
                                           phase=True, taper=False)[:2]
            lag2 = az.LCurve.calculate_lag(yarr, xarr, deltat,
                                           phase=True, taper=True)[:2]

            lag.append([lag1, lag2])
            pbar.update()

    lag = np.array(lag)
    freq = lag[0,0,0]
    lagm = np.mean(lag[:,:,1], 0)
    lagp = np.percentile(lag[:,:,1], [50, 16, 100-16], 0)

    _,axs = plt.subplots(1, 2, figsize=(8,4), sharex=True, sharey=True)

    for idx in [0,1]:
        axs[idx].semilogx(freq, lagm[idx], color='C0', label='Mean')
        axs[idx].plot(freq, lagp[0,idx], color='C1', label='Median')
        axs[idx].fill_between(freq, lagp[1,idx], lagp[2,idx], alpha=0.5, color='C1')
        axs[idx].plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:],
                      color='C2', alpha=0.5, label='Inp')
        axs[idx].set_title(f'{"No" if idx==0 else "With"} Taper')

    plt.tight_layout()
    os.system('mkdir -p png')
    plt.savefig('png/phase__1.png')


def phase__2():
    """Constant phase lag, no binning, NOISE, RED NOISE LEAK

    TEST: phases from segments (i.e. leak) with noise,
    SUMMARY:
        1. again, taper helps alot! The scatter is very small 
            in intermediate frequencies not affected by noise, 
            unlike in no-tpaer case.
    """
    npoints = 2**12
    deltat = 1.0
    mean = 100.0
    lag  = 0.5
    nsim = 200

    sim = az.SimLC(seed=463)
    sim.add_model('powerlaw', [1e-2, -2])
    sim.add_model('constant', [lag], lag=True)

    lag = []
    with ProgressBar(nsim) as pbar:
        for _ in range(nsim):
            sim.simulate(npoints*4, deltat, mean, norm='var')
            sim.apply_lag(phase=True)

            scale_fac = 100
            xarr = np.random.poisson(sim.lcurve[1][:npoints]*scale_fac)/scale_fac
            yarr = np.random.poisson(sim.lcurve[2][:npoints]*scale_fac)/scale_fac
            lag1 = az.LCurve.calculate_lag(yarr, xarr, deltat,
                                           phase=True, taper=False)[:2]
            lag2 = az.LCurve.calculate_lag(yarr, xarr, deltat,
                                           phase=True, taper=True)[:2]

            lag.append([lag1, lag2])
            pbar.update()

    lag = np.array(lag)
    freq = lag[0,0,0]
    lagm = np.mean(lag[:,:,1], 0)
    lagp = np.percentile(lag[:,:,1], [50, 16, 100-16], 0)

    _,axs = plt.subplots(1, 2, figsize=(8,4), sharex=True, sharey=True)

    for idx in [0,1]:
        axs[idx].semilogx(freq, lagm[idx], color='C0', label='Mean')
        axs[idx].plot(freq, lagp[0,idx], color='C1', label='Median')
        axs[idx].fill_between(freq, lagp[1,idx], lagp[2,idx], alpha=0.5, color='C1')
        axs[idx].plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:],
                      color='C2', alpha=0.5, label='Inp')
        axs[idx].set_title(f'{"No" if idx==0 else "With"} Taper')

    plt.tight_layout()
    os.system('mkdir -p png')
    plt.savefig('png/phase__2.png')


def phase__3():
    """Constant phase lag, BINNING, NOISE, RED NOISE LEAK

    TEST: leak with binning
    SUMMARY:
        1. no-taper is biased.
        2. taper has less scatter in intermediate frequencies and is not biased.
    """
    npoints = 2**12
    deltat = 1.0
    mean = 100.0
    lag  = 0.5
    nsim = 200
    bins = [2, 5, 10, 20, 50]

    sim = az.SimLC(seed=463)
    sim.add_model('powerlaw', [1e-2, -2])
    sim.add_model('constant', [lag], lag=True)

    lag = []
    with ProgressBar(nsim) as pbar:
        for _ in range(nsim):
            sim.simulate(npoints*4, deltat, mean, norm='var')
            sim.apply_lag(phase=True)

            scale_fac = 100
            xarr = np.random.poisson(sim.lcurve[1][:npoints]*scale_fac)/scale_fac
            yarr = np.random.poisson(sim.lcurve[2][:npoints]*scale_fac)/scale_fac
            xerr = (xarr/scale_fac)**0.5
            yerr = (yarr/scale_fac)**0.5

            blag = []
            for _bin in bins:

                lag1 = az.LCurve.calculate_lag(
                    yarr, xarr, deltat, phase=True, taper=False,
                    fqbin={'by_n': [_bin,1]}, xerr=xerr, yerr=yerr
                )[:3]
                lag2 = az.LCurve.calculate_lag(
                    yarr, xarr, deltat, phase=True, taper=True,
                    fqbin={'by_n': [_bin,1]}, xerr=xerr, yerr=yerr
                )[:3]
                blag.append([lag1, lag2])

            lag.append(blag)
            pbar.update()

    nbins = len(bins)
    _,axs = plt.subplots(2, nbins, figsize=(11,6), sharex=True, sharey=True)

    for ibin in range(nbins):
        blag = np.array([_[ibin] for _ in lag])
        freq = blag[0,0,0]
        lagm = np.mean(blag[:,:,1], 0)
        lagp = np.percentile(blag[:,:,1], [50, 16, 100-16], 0)

        for idx in [0,1]:
            axs[idx,ibin].semilogx(freq, lagm[idx], color='C0', label='Mean')
            axs[idx,ibin].plot(freq, lagp[0,idx], color='C1', label='Median', alpha=0.6)
            axs[idx,ibin].fill_between(freq, lagp[1,idx], lagp[2,idx], alpha=0.2, color='C1')
            axs[idx,ibin].plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:],
                          color='C2', alpha=0.5, label='Inp')

        axs[idx,ibin].set_title(f'bin: {bins[ibin]}')

    plt.tight_layout()
    os.system('mkdir -p png')
    plt.savefig('png/phase__3.png')


def phase__4():
    """Constant phase lag, BINNING, NOISE, RED NOISE LEAK

    TEST: error formular, and the psd/noise inside
    SUMMARY:
        1. Formula works with taper, and simple ftt (no norms, logavg) using
            psd and cxd without noise subtraction as in Nowak+99 and not Uttley+14
        2. Without tapering, the error formula gives large error. Even with tapering,
            the formula gives a slightly larger errors.
    """
    npoints = 2**12
    deltat = 1.0
    mean = 100.0
    lag  = 0.5
    nsim = 20
    bins = [2, 5, 10, 20, 50]

    sim = az.SimLC(seed=46377)
    sim.add_model('powerlaw', [1e-2, -2])
    sim.add_model('constant', [lag], lag=True)

    lag = []
    with ProgressBar(nsim) as pbar:
        for _ in range(nsim):
            sim.simulate(npoints*4, deltat, mean, norm='var')
            sim.apply_lag(phase=True)

            scale_fac = 100
            xarr = np.random.poisson(sim.lcurve[1][:npoints]*scale_fac)/scale_fac
            yarr = np.random.poisson(sim.lcurve[2][:npoints]*scale_fac)/scale_fac
            xerr = (xarr/scale_fac)**0.5
            yerr = (yarr/scale_fac)**0.5

            blag = []
            for _bin in bins:

                lag1 = az.LCurve.calculate_lag(
                    yarr, xarr, deltat, phase=True, taper=False,
                    fqbin={'by_n': [_bin,1]}, xerr=xerr, yerr=yerr
                )
                lag2 = az.LCurve.calculate_lag(
                    yarr, xarr, deltat, phase=True, taper=True,
                    fqbin={'by_n': [_bin,1]}, xerr=xerr, yerr=yerr
                )
                lag1 = np.vstack([lag1[:3], lag1[3]['coh']])
                lag2 = np.vstack([lag2[:3], lag2[3]['coh']])
                blag.append([lag1, lag2])

            lag.append(blag)
            pbar.update()

    nbins = len(bins)
    _,axs = plt.subplots(4, nbins, figsize=(11,12), sharex=True, sharey='row')

    for ibin in range(nbins):
        blag = np.array([_[ibin] for _ in lag])
        freq = blag[0,0,0]
        lage = np.mean(blag[:,:,2], 0)
        cohe = np.mean(blag[:,:,4], 0)
        cohp = np.percentile(blag[:,:,3], [50, 16, 100-16], 0)
        lagp = np.percentile(blag[:,:,1], [50, 16, 100-16], 0)

        for idx in [0,1]:
            axs[idx,ibin].semilogx(freq, lagp[0,idx], color='C0',
                                   label='Median', alpha=0.6)
            axs[idx,ibin].fill_between(freq, lagp[1,idx], lagp[2,idx],
                                       alpha=0.2, color='C0')
            axs[idx,ibin].fill_between(freq, lagp[0,idx]-lage[idx],
                                       lagp[0,idx]+lage[idx], alpha=0.5, color='C1')
            axs[idx,ibin].plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:],
                          color='C2', alpha=0.5, label='Inp')
            idx2 = idx+2
            axs[idx2,ibin].semilogx(freq, cohp[0,idx], color='C0',
                                    label='Median', alpha=0.6)
            axs[idx2,ibin].fill_between(freq, cohp[1,idx], cohp[2,idx],
                                        alpha=0.2, color='C0')
            axs[idx2,ibin].fill_between(freq, cohp[0,idx]-cohe[idx],
                                        cohp[0,idx]+cohe[idx], alpha=0.5, color='C1')

        axs[idx,ibin].set_title(f'bin: {bins[ibin]}')

    plt.tight_layout()
    os.system('mkdir -p png')
    plt.savefig('png/phase__4.png')



if __name__ == '__main__':

    pars = argp.ArgumentParser(
        description="""
        Run simulations for the LCurve functionality
        """,
        formatter_class=argp.ArgumentDefaultsHelpFormatter
    )



    pars.add_argument('--sim_1', action='store_true', default=False,
                      help="constant phase. leak.")
    pars.add_argument('--sim_2', action='store_true', default=False,
                      help="constant phase. leak & noise")
    pars.add_argument('--sim_3', action='store_true', default=False,
                      help="constant phase. leak, noise & binning.")
    pars.add_argument('--sim_4', action='store_true', default=False,
                      help="constant phase. error formula")

    # process arguments #
    args = pars.parse_args()


    # constant phase; leak
    if args.sim_1:
        phase__1()

    # constant phase; leak & noise
    if args.sim_2:
        phase__2()

    # constant phase; leak, noise and binning
    if args.sim_3:
        phase__3()

    # constant phase: formula errors
    if args.sim_4:
        phase__4()
