#!/usr/bin/env python


import numpy as np
import sys
import os
import argparse as ARG
from IPython import embed
import pylab as plt


sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import aztools as az



def psd_1():
    """Simple powerlaw psd, no noise, no binning, no bias correction"""
    n    = 512
    dt   = 1.0
    mu   = 100.0
    nsim = 200

    sim = az.SimLC(seed=467384)
    sim.add_model('powerlaw', [1e-2, -2])

    psd = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n, dt, mu, norm='var')
        psd.append(az.LCurve.calculate_psd(sim.x, dt, 'var'))
    psd = np.array(psd)
    fq  = psd[0,0]
    p   = psd[:,1].mean(0)
    pe  = psd[:,1].std(0) 


    plt.rcParams['figure.figsize'] = [4, 6]
    plt.rcParams['font.size'] = 7

    os.system('mkdir -p png')
    plt.errorbar(fq, p, pe, fmt='o')
    plt.xscale('log'); plt.yscale('log')
    plt.plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:])
    plt.savefig('png/psd_1.png')


def psd_2():
    """Simple powerlaw psd, no noise, with BINNING, no bias correction"""
    n    = 512
    dt   = 1.0
    mu   = 100.0
    nsim = 200

    sim = az.SimLC(seed=467384)
    sim.add_model('powerlaw', [1e-2, -2])

    psd = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n, dt, mu, norm='var')
        p = az.LCurve.calculate_psd(sim.x, dt, 'var')

        # bins: 2#
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[2,1]}, p[2], logavg=False)[:3])
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[2,1]}, p[2], logavg=True)[:3])

        # bins: 5#
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[5,1]}, p[2], logavg=False)[:3])
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[5,1]}, p[2], logavg=True)[:3])

        # bins: 10#
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[10,1]}, p[2], logavg=False)[:3])
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[10,1]}, p[2], logavg=True)[:3])
        
        # bins: 20#
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[20,1]}, p[2], logavg=False)[:3])
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[20,1]}, p[2], logavg=True)[:3])

        # bins: 50#
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[50,1]}, p[2], logavg=False)[:3])
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[50,1]}, p[2], logavg=True)[:3])
    psd = np.array(psd).reshape((nsim, 10, 3))


    # prepare for plotting #
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 7


    # plot & save #
    for iplt in range(5):
        p  = np.array([[p for p in ps[(2*iplt):(2*iplt+2)]] for ps in psd])
        ax = plt.subplot(2,3,iplt+1)
        ax.set_xscale('log'); ax.set_yscale('log')
        plt.errorbar(p[0,0,0], p[:,0,1].mean(), p[:,0,1].std(), 
            fmt='o', ms=3, alpha=0.5) # logavg=0
        plt.errorbar(p[0,1,0], p[:,1,1].mean(), p[:,1,1].std(), 
            fmt='o', ms=3, alpha=0.5) # logavg=1
        plt.plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:])
    plt.tight_layout(pad=0)
    plt.savefig('png/psd_2.png')


def psd_3():
    """Simple powerlaw psd, no noise, with BINNING, RED NOISE LEAK"""
    n    = 512
    dt   = 1.0
    mu   = 100.0
    nsim = 200

    sim = az.SimLC(seed=467384)
    sim.add_model('powerlaw', [1e-2, -2])

    psd = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n*4, dt, mu, norm='var')
        p  = az.LCurve.calculate_psd(sim.x[n:(2*n)], dt, 'var')
        pt = az.LCurve.calculate_psd(sim.x[n:(2*n)], dt, 'var', taper=True)

        # bins: 10 #
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[10,1]}, p[2], logavg=False)[:3])
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[10,1]}, p[2], logavg=True)[:3])
        
        # bins: 10; taper #
        psd.append(az.LCurve.bin_psd(pt[0], pt[1], {'by_n':[10,1]}, pt[2], logavg=False)[:3])
        psd.append(az.LCurve.bin_psd(pt[0], pt[1], {'by_n':[10,1]}, pt[2], logavg=True)[:3])


        # bins: 40#
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[40,1]}, p[2], logavg=False)[:3])
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[40,1]}, p[2], logavg=True)[:3])

        # bins: 40; taper #
        psd.append(az.LCurve.bin_psd(pt[0], pt[1], {'by_n':[40,1]}, pt[2], logavg=False)[:3])
        psd.append(az.LCurve.bin_psd(pt[0], pt[1], {'by_n':[40,1]}, pt[2], logavg=True)[:3])
    
    psd = np.array(psd).reshape((nsim, 8, 3))


    # prepare for plotting #
    plt.rcParams['figure.figsize'] = [8, 5]
    plt.rcParams['font.size'] = 7
    labels = ['10', '10-taper', '40', '40-taper']


    # plot & save #
    for iplt in range(4):
        p  = np.array([[p for p in ps[(2*iplt):(2*iplt+2)]] for ps in psd])
        ax = plt.subplot(2,2,iplt+1)
        ax.set_xscale('log'); ax.set_yscale('log')
        plt.errorbar(p[0,0,0], p[:,0,1].mean(), p[:,0,1].std(), 
            fmt='o', ms=3, alpha=0.5) # logavg=0
        plt.errorbar(p[0,1,0], p[:,1,1].mean(), p[:,1,1].std(), 
            fmt='o', ms=3, alpha=0.5) # logavg=1
        plt.plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:])
        plt.title(labels[iplt])
    plt.tight_layout(pad=0)
    plt.savefig('png/psd_3.png')


def psd_4():
    """Simple powerlaw psd, no noise, with BINNING, RED NOISE LEAK.
    Different slopes
    """
    n    = 512
    dt   = 1.0
    mu   = 100.0
    nsim = 200
    slopes = [-1,-1.5,-2,-2.5,-3,-3.5]



    sim = [az.SimLC(seed=467384) for s in slopes]
    for i,s in enumerate(sim):
        s.add_model('powerlaw', [1e-2, slopes[i]])

    psd = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        pp = []
        for s in sim:
            s.simulate(n*4, dt, mu, norm='var')
            p  = az.LCurve.calculate_psd(s.x[n:(2*n)], dt, 'var', taper=True)
            pp.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[10,1]}, p[2], logavg=True)[:3])
        psd.append(pp)

    psd = np.array(psd)

    # prepare for plotting #
    plt.rcParams['figure.figsize'] = [6, 6]
    plt.rcParams['font.size'] = 7

    # plot & save #
    for iplt in range(6):
        p  = psd[:, iplt]
        plt.xscale('log'); plt.yscale('log')
        plt.errorbar(p[0,0], p[:,1].mean(0), p[:,1].std(0), 
            fmt='o', ms=3, alpha=0.5) 
        plt.plot(sim[iplt].normalized_psd[0,1:], sim[iplt].normalized_psd[1,1:], lw=0.5)
    plt.savefig('png/psd_4.png')


def psd_5():
    """Simple powerlaw psd, no noise.
    Test using several light curves with different lengths
    """
    n    = 512
    dt   = 1.0
    mu   = 100.0
    nsim = 200


    sim = az.SimLC(seed=4673)
    sim.add_model('powerlaw', [1e-2, -2])

    psd = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)
        sim.simulate(n*4, dt, mu, norm='var')

        # 3 equal lengths #
        x  = [sim.x[:n], sim.x[n:(2*n)], sim.x[(2*n):(3*n)]]
        p  = az.LCurve.calculate_psd(x, dt, 'var', taper=True)
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[10,1]}, p[2], logavg=True)[:3])

        # 3 un-equal lengths #
        nn = np.int(0.5*n)
        x  = [sim.x[:nn], sim.x[nn:(nn+n)], sim.x[(nn+n):(nn+n+n//2)], sim.x[(nn+n+n//2):(nn+2*n)]]
        p  = az.LCurve.calculate_psd(x, dt, 'var', taper=True)
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[10,1]}, p[2], logavg=True)[:3])

    psd = np.array(psd).reshape((nsim, 2, 3, -1))

    # prepare for plotting #
    plt.rcParams['figure.figsize'] = [7, 5]
    plt.rcParams['font.size'] = 7

    # plot & save #
    for iplt in range(2):
        p  = psd[:, iplt]
        pp = np.percentile(p[:,1],[50,16,100-16],0)
        ax = plt.subplot(1,2,iplt+1)
        ax.set_xscale('log'); ax.set_yscale('log')
        plt.errorbar(p[0,0], p[:,1].mean(0), p[:,1].std(0), 
            fmt='o', ms=3, alpha=0.5)
        plt.fill_between(p[0,0], p[:,1].mean(0)-p[:,2].mean(0), p[:,1].mean(0)+p[:,2].mean(0),
                alpha=0.5)
        plt.plot(p[0,0], pp[0], '-.', lw=0.8)
        plt.plot(p[0,0], pp[1], ':', lw=0.8)
        plt.plot(p[0,0], pp[2], ':', lw=0.8)
        plt.plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:], lw=0.5)
    #plt.show();exit(0)
    plt.savefig('png/psd_5.png')


def psd_6():
    """Simple powerlaw psd, no noise, with BINNING, RED NOISE LEAK.
    Compare expontiation to no expontiation


    #1 The variance in the exponential light curves is higher than the simple case.
    #2 Red-noise leak is severe is the exponential light curves 
    """
    n    = 512
    dt   = 1.0
    mu   = 100
    nsim = 200



    sim = [az.SimLC(seed=467) for i in [0,1]]
    for s in sim:
        #s.add_model('powerlaw', [1e-2, -2])
        s.add_model('broken_powerlaw', [1e-2, -1, -2, 1e-1])

    psd = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim[0].simulate(n*8, dt, mu, norm='var')
        p  = az.LCurve.calculate_psd(sim[0].x[:n], dt, 'var', taper=True)
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[10,1]}, p[2], logavg=True)[:3])

        sim[1].simulate_pdf(n*8, dt, mu, norm='var', pdf='lognorm(s=0.5)')
        p  = az.LCurve.calculate_psd(sim[1].x[:n], dt, 'var', taper=True)
        psd.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[10,1]}, p[2], logavg=True)[:3])

    psd = np.array(psd).reshape((nsim, 2, 3, -1))

    # prepare for plotting #
    plt.rcParams['figure.figsize'] = [9, 4]
    plt.rcParams['font.size'] = 7

    # plot & save #
    ax = plt.subplot(131)
    ax.set_xscale('log'); ax.set_yscale('log')
    iplt = 0
    p  = psd[:, iplt]
    pp = np.percentile(p[:,1],[50,16,100-16],0)
    plt.plot(p[0,0], pp[0], '-', alpha=0.5, color='C%d'%(iplt)) 
    plt.fill_between(p[0,0], pp[1], pp[2], facecolor='C%d'%(iplt), alpha=0.2)
    plt.errorbar(p[0,0], p[:,1].mean(0), p[:,2].mean(0), alpha=0.4, fmt='o', ms=4)
    plt.plot(sim[iplt].normalized_psd[0,1:], sim[iplt].normalized_psd[1,1:], color='C2')

    ax = plt.subplot(132)
    ax.set_xscale('log'); ax.set_yscale('log')
    iplt = 1
    p  = psd[:, iplt]
    pp = np.percentile(p[:,1],[50,16,100-16],0)
    plt.plot(p[0,0], pp[0], '-', alpha=0.5, color='C%d'%(iplt)) 
    plt.fill_between(p[0,0], pp[1], pp[2], facecolor='C%d'%(iplt), alpha=0.2)
    plt.errorbar(p[0,0], p[:,1].mean(0), p[:,2].mean(0), alpha=0.4, fmt='o', ms=4)
    plt.plot(sim[iplt].normalized_psd[0,1:], sim[iplt].normalized_psd[1,1:], color='C2')

    ax = plt.subplot(133)
    plt.plot(sim[0].t[:n], sim[0].x[:n])
    plt.plot(sim[1].t[:n], sim[1].x[:n])
    plt.tight_layout(pad=0)
    plt.savefig('png/psd_6.png')


def psd_7():
    """Simple powerlaw psd, with BINNING, RED NOISE LEAK.
    Different noise levels (means)
    """
    n    = 2048
    dt   = 1.0
    mu   = [100, 10, 100, 500, 2000]
    nsim = 200



    sim = az.SimLC(seed=67384)
    #sim.add_model('powerlaw', [1e-5, -2])
    sim.add_model('broken_powerlaw', [1e-5, -1, -2, 1e-3])

    psd = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        pp = []
        for m in mu:
            #sim.simulate_pdf(n*4, dt, m, norm='rms', pdf='lognorm(s=0.3)')
            sim.simulate(n*4, dt, m, norm='rms')
            x = sim.x[:n]
            if len(pp)!=0:
               x = sim.add_noise(x)
            p  = az.LCurve.calculate_psd(x, dt, 'rms', taper=True)
            pp.append(az.LCurve.bin_psd(p[0], p[1], {'by_n':[10,1]}, p[2], logavg=True)[:3])
        psd.append(pp)

    psd = np.array(psd)

    # prepare for plotting #
    plt.rcParams['figure.figsize'] = [6, 6]
    plt.rcParams['font.size'] = 7

    # plot & save #
    for iplt in range(5):
        p  = psd[:, iplt]
        plt.xscale('log'); plt.yscale('log')
        plt.fill_between(p[0,0], p[:,1].mean(0)-p[:,1].std(0), p[:,1].mean(0)+p[:,1].std(0), 
            alpha=0.5) 
    plt.plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:], lw=0.5)
    plt.savefig('png/psd_7.png')

###################

def lag_1():
    """Simple powerlaw psd, no noise, no binning, no bias correction
    constant time lag
    """
    n    = 512
    dt   = 1.0
    mu   = 100.0
    lag  = 30
    nsim = 200

    sim = az.SimLC(seed=4632184)
    sim.add_model('broken_powerlaw', [1e-5, -1, -2, 1e-3])
    sim.add_model('constant', lag, lag=True)

    lag = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n, dt, mu, norm='rms')
        sim.apply_lag()
        lag.append(az.LCurve.calculate_lag(sim.y, sim.x, dt))
    
    lag = np.array(lag)
    fq  = lag[0,0]
    l   = lag[:,1].mean(0)
    le  = lag[:,1].std(0) 


    plt.rcParams['figure.figsize'] = [4, 6]
    plt.rcParams['font.size'] = 7

    plt.errorbar(fq, l, le, fmt='o')
    plt.xscale('log')
    plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:])
    plt.savefig('png/lag_1.png')


def lag_2():
    """Simple powerlaw psd, no noise, no binning, with RED NOISE LEAK.
    constant time lag; 
    """
    n     = 512
    dt    = 1.0
    mu    = 100.0
    phase = False
    lag   = 1 if phase else 30
    nsim  = 200

    sim = az.SimLC(seed=4632184)
    sim.add_model('broken_powerlaw', [1e-5, -1, -2, 1e-3])
    sim.add_model('constant', lag, lag=True)

    lag = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n*4, dt, mu, norm='rms')
        sim.apply_lag(phase)
        l = [az.LCurve.calculate_lag(sim.y[:n], sim.x[:n], dt, phase=phase)]
        l.append(az.LCurve.calculate_lag(sim.y[:n], sim.x[:n], dt, taper=True, phase=phase))
        lag.append(l)
    
    lag = np.array(lag)
    fq  = lag[0,0,0]
    l1  = lag[:,0,1].mean(0)
    l1s = lag[:,0,1].std(0) 
    l2  = lag[:,1,1].mean(0)
    l2s = lag[:,1,1].std(0) 

    plt.rcParams['figure.figsize'] = [4, 6]
    plt.rcParams['font.size'] = 7

    plt.fill_between(fq, l1-l1s, l1+l1s, alpha=0.3, color='C0')
    plt.fill_between(fq, l2-l2s, l2+l2s, alpha=0.3, color='C1')
    plt.xscale('log')
    plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:], color='C2')
    plt.savefig('png/lag_2.png')


def lag_3():
    """Simple powerlaw psd, no noise, with BINNING, RED NOISE LEAK.
    constant time lag/or phase lag; 

    Without tapering: the errors from the lag formula are always 
        smaller than the simulations.
    With tapering, for binning of 5 frequencies per bin and higher,
        the errors from the formula are larger than simulations.
    The lowest bin is always small when using a constant lag. It is
        ok when fitting for phases. It seems to be related to how
        we define the central frequncy of the bin.
    """
    n     = 2048
    dt    = 1.0
    mu    = 100.0
    phase = True
    lag   = 1 if phase else 5
    nsim  = 100
    bins  = [1, 2, 5, 10, 20, 40]

    sim = az.SimLC(seed=42184)
    sim.add_model('broken_powerlaw', [1e-5, -1, -2, 1e-3])
    sim.add_model('constant', lag, lag=True)

    Lag = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n*4, dt, mu, norm='rms')
        sim.apply_lag(phase)
        x,y = sim.x[:n], sim.y[:n]

        l = []
        for b in bins:
            l.append(az.LCurve.calculate_lag(y, x, dt, fqbin={'by_n':[b,1]}, 
                    phase=phase, rerr=[np.zeros(n)], Rerr=[np.zeros(n)], norm='none'))
            l.append(az.LCurve.calculate_lag(y, x, dt, fqbin={'by_n':[b,1]}, 
                    taper=True, phase=phase, rerr=[np.zeros(n)], Rerr=[np.zeros(n)]))

        Lag.append(l)
    
    Lag = np.array(Lag)

    plt.rcParams['figure.figsize'] = [12, 5]
    plt.rcParams['font.size'] = 7


    for iplt in range(6):
    
        l0  = np.array([list(l[iplt*2][:3]) for l in Lag])
        l0t = np.array([list(l[iplt*2+1][:3]) for l in Lag])

        ax = plt.subplot(2,6,2*iplt+1)
        ax.set_xscale('log')
        if phase: ax.set_ylim([0, 2])
        l,le,ll = l0[:,1].mean(0), l0[:,1].std(0), l0[:,2].mean(0)
        plt.errorbar(l0[0,0], l, le, fmt='o', ms=3, alpha=0.3, color='C1')
        plt.fill_between(l0[0,0], l-ll, l+ll, alpha=0.3, facecolor='C0')
        plt.title('b{}::{:3.3g}'.format(bins[iplt], (ll/le)[1:-1].mean() ))
        plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[0,1:]*0+lag)

        ax = plt.subplot(2,6,2*iplt+2)
        ax.set_xscale('log')
        if phase: ax.set_ylim([0, 2])     
        l,le,ll = l0t[:,1].mean(0), l0t[:,1].std(0), l0t[:,2].mean(0)
        plt.errorbar(l0[0,0], l, le, fmt='o', ms=3, alpha=0.3, color='C3')
        plt.fill_between(l0[0,0], l-ll, l+ll, alpha=0.3, color='C2')
        plt.title('b{}:taper:{:3.3g}'.format(bins[iplt], (ll/le)[1:-1].mean() ))

        plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[0,1:]*0+lag)
    plt.tight_layout(pad=0)
    plt.savefig('png/lag_3.png')


def lag_4():
    """Simple powerlaw psd, with BINNING, RED NOISE LEAK. POISSON NOISE
    constant time lag/or phase lag; 

    The errors from Nowak99 seem to be around 2/3 of the errors from the
        distributions. I don't understand this. This seems to depend on the
        psd slope, the steeper the psd, the higher the discrepency.

    """
    n     = 2048
    dt    = 1.0
    mu    = 10000.0
    phase = True
    lag   = 1 if phase else 5
    nsim  = 200
    bins  = [1, 2, 5, 10, 20, 40]

    sim = az.SimLC(seed=42184)
    sim.add_model('broken_powerlaw', [1e-5, -1, -2, 1e-3])
    sim.add_model('constant', lag, lag=True)
    embed();exit(0)

    Lag = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n*4, dt, mu, norm='rms')
        sim.apply_lag(phase)
        x,y = sim.x[:n], sim.y[:n]
        x  = np.random.poisson(x)
        y  = np.random.poisson(y)

        l = []
        for b in bins:
            l.append(az.LCurve.calculate_lag(y, x, dt, fqbin={'by_n':[b,1.1]}, 
                    phase=phase))
            l.append(az.LCurve.calculate_lag(y, x, dt, fqbin={'by_n':[b,1.1]}, 
                    taper=True, phase=phase))

        Lag.append(l)
    
    Lag = np.array(Lag)

    plt.rcParams['figure.figsize'] = [12, 5]
    plt.rcParams['font.size'] = 7


    for iplt in range(6):
    
        l0  = np.array([list(l[iplt*2][:3]) for l in Lag])
        l0t = np.array([list(l[iplt*2+1][:3]) for l in Lag])

        ax = plt.subplot(4,6,4*iplt+1)
        ax.set_xscale('log')
        if phase: ax.set_ylim([0, 2])
        l,le,ll = l0[:,1].mean(0), l0[:,1].std(0), np.median(l0[:,2], 0)
        plt.errorbar(l0[0,0], l, le, fmt='o', ms=3, alpha=0.3, color='C1')
        plt.fill_between(l0[0,0], l-ll, l+ll, alpha=0.3, facecolor='C0')
        pp = np.percentile(l0[:,1],[50,16,100-16],0)
        plt.plot(l0[0,0], pp[1], ':', color='C0', lw=0.5)
        plt.plot(l0[0,0], pp[2], ':', color='C0', lw=0.5)
        plt.title('b{}::{:3.3g}'.format(bins[iplt], (ll/le)[1:-1].mean() ))
        plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[0,1:]*0+lag)
        
        ax = plt.subplot(4,6,4*iplt+2)
        ax.set_ylim([0.1, 2])
        plt.semilogx(l0[0,0], ll/le)
        plt.plot(l0[0,0], l*0+1, lw=0.5)
        plt.title('error formula/simulation')
        

        ax = plt.subplot(4,6,4*iplt+3)
        ax.set_xscale('log')
        if phase: ax.set_ylim([0, 2])     
        l,le,ll = l0t[:,1].mean(0), l0t[:,1].std(0), np.median(l0t[:,2], 0)
        plt.errorbar(l0[0,0], l, le, fmt='o', ms=3, alpha=0.3, color='C3')
        plt.fill_between(l0[0,0], l-ll, l+ll, alpha=0.3, color='C2')
        pp = np.percentile(l0t[:,1],[50,16,100-16],0)
        plt.plot(l0[0,0], pp[1], ':', color='C2', lw=0.5)
        plt.plot(l0[0,0], pp[2], ':', color='C2', lw=0.5)
        plt.title('b{}:taper:{:3.3g}'.format(bins[iplt], (ll/le)[1:-1].mean() ))
        plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[0,1:]*0+lag)
        
        ax = plt.subplot(4,6,4*iplt+4)
        ax.set_ylim([0.1, 2])
        plt.semilogx(l0[0,0], ll/le)
        plt.plot(l0[0,0], l*0+1, lw=0.5)
        plt.title('error formula/simulation')

    plt.tight_layout(pad=0)
    plt.savefig('png/lag_4.png')


if __name__ == '__main__':
    
    p   = ARG.ArgumentParser(                                
        description='''
        Run simulations for the LCurve functionality
        ''',            
        formatter_class=ARG.ArgumentDefaultsHelpFormatter ) 


    p.add_argument('--psd_1', action='store_true', default=False,
            help="Simple psd simulation. ")
    p.add_argument('--psd_2', action='store_true', default=False,
            help="Simple psd simulation with BINNING")
    p.add_argument('--psd_3', action='store_true', default=False,
            help="Simple psd simulation with BINNING & RED LEAK")
    p.add_argument('--psd_4', action='store_true', default=False,
            help="Simple psd simulation with BINNING & RED LEAK & different slopes")
    p.add_argument('--psd_5', action='store_true', default=False,
            help="Simple psd simulation with BINNING & RED LEAK & different seglen")
    p.add_argument('--psd_6', action='store_true', default=False,
            help="BINNING & RED LEAK & Expontiation")
    p.add_argument('--psd_7', action='store_true', default=False,
            help="BINNING & RED LEAK & Differnet noise levels (means)")


    p.add_argument('--lag_1', action='store_true', default=False,
            help="Simple lag simulation. ")
    p.add_argument('--lag_2', action='store_true', default=False,
            help="Simple lag simulation. RED NOISE LEAK")
    p.add_argument('--lag_3', action='store_true', default=False,
            help="Simple lag simulation. RED NOISE LEAK, BINNING")
    p.add_argument('--lag_4', action='store_true', default=False,
            help="Simple lag simulation. RED NOISE LEAK, BINNING, POISSON NOISE")



    # process arguments #
    args = p.parse_args()


    # simple psd #
    if args.psd_1:
        psd_1()

    # Binning #
    elif args.psd_2:
        psd_2()

    # Red noise leak #
    elif args.psd_3:
        psd_3()

    # Red noise leak with different slopes #
    elif args.psd_4:
        psd_4()

    # Red noise leak segments with different length #
    elif args.psd_5:
        psd_5()

    # Red noise leak; Expontiation#
    elif args.psd_6:
        psd_6()

    # Red noise leak; different noise levels (means)#
    elif args.psd_7:
        psd_7()

    ################

    # simple lag #
    if args.lag_1:
        lag_1()

    # simple lag; red noise leak #
    if args.lag_2:
        lag_2()

    # simple lag; red noise leak & binning #
    if args.lag_3:
        lag_3()

    # simple lag; red noise leak & binning & poisson noise #
    if args.lag_4:
        lag_4()