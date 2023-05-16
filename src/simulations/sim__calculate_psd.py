#!/usr/bin/env python

"""Check LCurve.calcuate_psd"""

import argparse as argp
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.console import ProgressBar

import aztools as az

az.misc.set_fancy_plot(plt)

def calculate_psd__1():
    """Simple powerlaw psd, no poisson noise, no binning, no bias correction

    TEST: calls to fft functions
    """
    npoints = 512
    deltat = 1.0
    mean = 100.0
    nsim = 200

    sim = az.SimLC(seed=344551)
    sim.add_model('powerlaw', [1e-2, -2])

    psd = []
    with ProgressBar(nsim) as pbar:
        for _ in range(nsim):
            sim.simulate(npoints, deltat, mean, norm='var')
            psd.append(az.LCurve.calculate_psd(sim.lcurve[1], deltat, 'var'))
            pbar.update()

    psd = np.array(psd)
    freq = psd[0,0]
    psdm = psd[:,1].mean(0)
    psds = psd[:,1].std(0)

    _ = plt.figure(figsize=(7,7))

    plt.errorbar(freq, psdm, psds, fmt='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:])

    os.system('mkdir -p png')
    plt.savefig('png/calculate_psd__1.png')


def calculate_psd__2():
    """Simple powerlaw psd, POISSON/GAUSS noise, no binning, no bias correction
    TEST: noise level estimates
    """
    npoints = 512
    deltat = 1.0
    mean = 100.0
    nsim = 200

    sim = az.SimLC(seed=34451)
    sim.add_model('powerlaw', [1e-2, -2])

    psd = []
    with ProgressBar(nsim) as pbar:
        for _ in range(nsim):

            sim.simulate(npoints, deltat, mean, norm='var')
            xarr = np.random.poisson(sim.lcurve[1])
            psd1 = az.LCurve.calculate_psd(xarr, deltat, 'var')

            xarr = np.random.normal(sim.lcurve[1], mean*0.01)
            psd2 = az.LCurve.calculate_psd(xarr, deltat, 'var', rerr=xarr*0+mean*0.01)

            psd.append([psd1,psd2])
            pbar.update()

    psd = np.array(psd)
    freq = psd[0,0,0]
    psdm = psd[:,:,1].mean(0)
    psds = psd[:,:,1].std(0)
    psdn = psd[:,:,2].mean(0)

    _ = plt.figure(figsize=(7,7))

    plt.errorbar(freq, psdm[0], psds[0], fmt='o', color='C0', elinewidth=.2)
    plt.errorbar(freq, psdm[1], psds[1], fmt='s', color='C1', elinewidth=.2)
    plt.plot(freq, psdn[0], color='C0', label='POISS')
    plt.plot(freq, psdn[1], color='C1', label='GAUSS')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:],
             label='Input', color='C2', lw=2, alpha=0.5)
    plt.legend()

    os.system('mkdir -p png')
    plt.savefig('png/calculate_psd__2.png')


def calculate_psd__3():
    """Simple powerlaw psd, no poisson noise, no binning, RED NOISE LEAK

    TEST: tapering
    SUMMARY:
        1- Where there is red noise leak, tapering clearly helps.
        2- The median of the simulations is slightly biased, while the mean is not
        3- The effect is strongest when gamma >~2
    """
    npoints = 512
    deltat = 1.0
    mean = 100.0
    nsim = 500

    sim = az.SimLC(seed=6451)
    sim.add_model('powerlaw', [1e-3, -2.5])

    psd = []
    with ProgressBar(nsim) as pbar:
        for _ in range(nsim):

            sim.simulate(4*npoints, deltat, mean, norm='var')
            psd1 = az.LCurve.calculate_psd(sim.lcurve[1][:npoints], deltat, 'var')
            psd2 = az.LCurve.calculate_psd(sim.lcurve[1][:npoints], deltat, 'var', taper=True)

            psd.append([psd1,psd2])
            pbar.update()

    psd = np.array(psd)
    freq = psd[0,0,0]
    psdm = psd[:,:,1].mean(0)
    psdd = np.median(psd[:,:,1], 0)

    _ = plt.figure(figsize=(7,7))

    plt.plot(freq, psdm[0], color='C0', label='No-Taper-Mean')
    plt.plot(freq, psdd[0], '-.', color='C0', label='No-Taper-Median')
    plt.plot(freq, psdm[1], color='C1', label='Taper-Mean')
    plt.plot(freq, psdd[1], '-.', color='C1', label='Taper-Median')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:],
             color='C2', lw=2, alpha=0.5, label='Input')
    plt.legend()

    os.system('mkdir -p png')
    plt.savefig('png/calculate_psd__3.png')


if __name__ == '__main__':

    pars = argp.ArgumentParser(
        description="""
        Run simulations for the LCurve functionality
        """,
        formatter_class=argp.ArgumentDefaultsHelpFormatter
    )



    pars.add_argument('--sim_1', action='store_true', default=False,
                      help="Simple psd simulation. Test fft calls.")
    pars.add_argument('--sim_2', action='store_true', default=False,
                      help="Simple psd simulation. Test noise level.")
    pars.add_argument('--sim_3', action='store_true', default=False,
                      help="Simple psd simulation. Test tapering ")

    # process arguments #
    args = pars.parse_args()


    # simple psd: fft calls
    if args.sim_1:
        calculate_psd__1()

    # simple psd: noise level #
    if args.sim_2:
        calculate_psd__2()

    # simple psd; tapering #
    if args.sim_3:
        calculate_psd__3()
