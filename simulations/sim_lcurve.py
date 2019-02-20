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


#####################

def calculate_psd__1():
    """"""
    """Simple powerlaw psd, no poisson noise, no binning, no bias correction

    TEST: calls to fft functions
    """
    n    = 512
    dt   = 1.0
    mu   = 100.0
    nsim = 200

    sim = az.SimLC(seed=344551)
    sim.add_model('powerlaw', [1e-2, -2])

    psd = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n, dt, mu, norm='var')
        psd.append(az.LCurve.calculate_psd(sim.x, dt, 'var'))
    psd = np.array(psd)
    fq  = psd[0,0]
    p   = psd[:,1].mean(0)
    ps  = psd[:,1].std(0) 


    plt.rcParams['figure.figsize'] = [4, 6]
    plt.rcParams['font.size'] = 7

    os.system('mkdir -p png')
    plt.errorbar(fq, p, ps, fmt='o')
    plt.xscale('log'); plt.yscale('log')
    plt.plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:])
    plt.savefig('png/calculate_psd__1.png')


def calculate_psd__2():
    """"""
    """Simple powerlaw psd, POISSON/GAUSS noise, no binning, no bias correction
    TEST: noise level estimates

    """
    n    = 512
    dt   = 1.0
    mu   = 100.0
    nsim = 200

    sim = az.SimLC(seed=34451)
    sim.add_model('powerlaw', [1e-2, -2])

    psd = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n, dt, mu, norm='var')
        x  = np.random.poisson(sim.x)
        p1 = az.LCurve.calculate_psd(x, dt, 'var')

        x  = np.random.normal(sim.x, mu*0.01)
        p2 = az.LCurve.calculate_psd(x, dt, 'var', rerr=x*0+mu*0.01)

        psd.append([p1,p2])
    psd = np.array(psd)
    fq  = psd[0,0,0]
    p   = psd[:,:,1].mean(0)
    ps  = psd[:,:,1].std(0)
    pn  = psd[:,:,2].mean(0)


    plt.rcParams['figure.figsize'] = [5, 5]
    plt.rcParams['font.size'] = 7

    os.system('mkdir -p png')
    plt.loglog(fq, p[0], 'o', color='C0', alpha=0.3)
    plt.plot(fq, pn[0], color='C0')
    plt.loglog(fq, p[1], 'o', color='C1', alpha=0.3)
    plt.plot(fq, pn[1], color='C1')
    plt.plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:], color='C2')
    plt.savefig('png/calculate_psd__2.png')


def calculate_psd__3():
    """Simple powerlaw psd, no poisson noise, no binning, RED NOISE LEAK
    Use psd index of -3 for extreme effect

    TEST: tapering
    SUMMARY:
        1- Where there is red noise leak, tapering clearly helps.
        2- The median of the simulations is slightly biased, while the mean is not
        3- The effect is strongest when gamma >~2
    """
    n    = 512
    dt   = 1.0
    mu   = 100.0
    nsim = 200

    sim = az.SimLC(seed=349851)
    sim.add_model('powerlaw', [1e-2, -2.5])

    psd = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n*4, dt, mu, norm='var')
        p1 = az.LCurve.calculate_psd(sim.x[:n], dt, 'var')
        p2 = az.LCurve.calculate_psd(sim.x[:n], dt, 'var', taper=True)
        psd.append([p1,p2])

    psd = np.array(psd)
    fq  = psd[0,0,0]
    p   = np.mean(psd[:,:,1], 0)
    pmd = np.median(psd[:,:,1], 0)
    ps  = np.std(psd[:,:,1], 0)


    plt.rcParams['figure.figsize'] = [5, 5]
    plt.rcParams['font.size'] = 7

    os.system('mkdir -p png')
    plt.loglog(fq, p[0], 'o', color='C0', alpha=0.3, label='no-taper, mean')
    plt.loglog(fq, pmd[0], 's', color='C3', alpha=0.3, label='no-taper, median')
    plt.loglog(fq, p[1], 'o', color='C1', alpha=0.3, label='with-taper')
    plt.loglog(fq, pmd[1], 's', color='C4', alpha=0.3, label='with-taper, median')
    plt.plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:], color='C2')
    plt.legend()
    plt.savefig('png/calculate_psd__3.png')


def bin_psd__1():
    """Simple powerlaw psd, no poisson noise, BINNING, no noise leak
    

    TEST: linear vs log binning
    SUMMARY:
        1- logavg does not do better than linear. The applied bias from Papadakis
            over-corrects the values when averaging a small number of frequencies. 
            It only helps with lowest frequency when averaging >10 bins
    """
    n    = 2**12
    dt   = 1.0
    mu   = 100.0
    nsim = 200
    bins = [2, 5, 10, 20, 50]

    sim = az.SimLC(seed=349851)
    sim.add_model('powerlaw', [1e-2, -2.])

    psd = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n, dt, mu, norm='var')
        p = az.LCurve.calculate_psd(sim.x[:n], dt, 'var')
        pb = []
        for b in bins:
            p1 = az.LCurve.bin_psd(p[0], p[1], fqbin={'by_n':[b,1]}, logavg=False)[:3]
            p2 = az.LCurve.bin_psd(p[0], p[1], fqbin={'by_n':[b,1]}, logavg=True)[:3]
            pb.append([p1,p2])
        psd.append(pb)


    plt.rcParams['figure.figsize'] = [14, 4]
    plt.rcParams['font.size'] = 7

    for ib in range(len(bins)):

        pb = np.array([x[ib] for x in psd])
        fq = pb[0,0,0]
        p   = np.mean(pb[:,:,1], 0)
        ps  = np.std(pb[:,:,1], 0)

        ax = plt.subplot(1, len(bins), ib+1)
        plt.loglog(fq, p[0], '-', color='C0', label='lin, mean')
        plt.loglog(fq, p[1], '-.', color='C1', label='log, mean')
        plt.plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:], color='C2')
        plt.legend()
        plt.title('bin:%d'%bins[ib])
    plt.savefig('png/bin_psd__1.png')


def bin_psd__2():
    """Simple powerlaw psd, no poisson noise, BINNING, LEAK
    

    TEST: linear vs log binning, with leak/tapering
    SUMMARY:
        1- Similar to bin_psd__1. logavg is overall slightly better than linear at 
            the lowerst frequencies, but it is biased when num. of averaged freq is <~+10
        2- taper is always better as we saw in calculate_psd simulations.
    """
    n    = 2**12
    dt   = 1.0
    mu   = 100.0
    nsim = 200
    bins = [2, 10, 50]

    sim = az.SimLC(seed=351)
    sim.add_model('powerlaw', [1e-2, -2.])

    psd = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n*4, dt, mu, norm='var')
        p = az.LCurve.calculate_psd(sim.x[:n], dt, 'var')
        psd.append([])
        pb = []
        for b in bins:
            p1 = az.LCurve.bin_psd(p[0], p[1], fqbin={'by_n':[b,1]}, logavg=False)[:3]
            p2 = az.LCurve.bin_psd(p[0], p[1], fqbin={'by_n':[b,1]}, logavg=True)[:3]
            pb.append([p1,p2])
        psd[-1].append(pb)

        pb = []
        p = az.LCurve.calculate_psd(sim.x[:n], dt, 'var', taper=True)
        for b in bins:
            p1 = az.LCurve.bin_psd(p[0], p[1], fqbin={'by_n':[b,1]}, logavg=False)[:3]
            p2 = az.LCurve.bin_psd(p[0], p[1], fqbin={'by_n':[b,1]}, logavg=True)[:3]
            pb.append([p1,p2])
        psd[-1].append(pb)


    plt.rcParams['figure.figsize'] = [14, 4]
    plt.rcParams['font.size'] = 7

    for ib in range(len(bins)):

        pb = np.array([[y[ib] for y in x] for x in psd])
        fq = pb[0,0,0,0]
        p   = np.mean(pb[:,:,:,1], 0)
        ps  = np.std(pb[:,:,:,1], 0)

        ax = plt.subplot(1, len(bins), ib+1)
        plt.loglog(fq, p[0,0], '-', color='C0', label='lin, no-taper')
        plt.loglog(fq, p[0,1], '-.', color='C1', label='log, no-taper')
        plt.loglog(fq, p[1,0], '-', color='C2', label='lin, taper')
        plt.loglog(fq, p[1,1], '-.', color='C3', label='log, taper')
        plt.plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:], color='C4')
        plt.legend()
        plt.title('bin:%d'%bins[ib])
    plt.savefig('png/bin_psd__2.png')


def bin_psd__3():
    """Simple powerlaw psd, POISSON, BINNING, LEAK
    

    TEST: linear vs log binning, with leak/tapering, with noise
    SUMMARY:
        1- logavg and taper do the best
        2- logavg on noise doesn't matter a lot,
        3- The bias in the first frequency is high. it is not noise-leak, somehow
        
    """
    n    = 2**12
    dt   = 1.0
    mu   = 1000.0
    nsim = 200
    bins = [2, 10, 50]

    sim = az.SimLC(seed=123351)
    sim.add_model('powerlaw', [1e-2, -2.])


    psd = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n*4, dt, mu, norm='var')
        x = np.random.poisson(sim.x[:n]*20)/20
        # use different sampling dt, so we reduce the noise a little bit
        p = az.LCurve.calculate_psd(x, dt, 'var', rerr=(x/20)**0.5)
        psd.append([])
        pb = []
        for b in bins:
            p1 = az.LCurve.bin_psd(p[0], p[1], fqbin={'by_n':[b,1]}, noise=p[2], logavg=False)
            p2 = az.LCurve.bin_psd(p[0], p[1], fqbin={'by_n':[b,1]}, noise=p[2], logavg=True)
            p1 = np.vstack([p1[:3], p1[3]['noise']])
            p2 = np.vstack([p2[:3], p2[3]['noise']])
            pb.append([p1,p2])
        psd[-1].append(pb)

        pb = []
        p = az.LCurve.calculate_psd(x, dt, 'var', taper=True, rerr=(x/20)**0.5)
        for b in bins:
            p1 = az.LCurve.bin_psd(p[0], p[1], fqbin={'by_n':[b,1]}, noise=p[2], logavg=False)
            p2 = az.LCurve.bin_psd(p[0], p[1], fqbin={'by_n':[b,1]}, noise=p[2], logavg=True)
            p1 = np.vstack([p1[:3], p1[3]['noise']])
            p2 = np.vstack([p2[:3], p2[3]['noise']])
            pb.append([p1,p2])
        psd[-1].append(pb)

    plt.rcParams['figure.figsize'] = [14, 8]
    plt.rcParams['font.size'] = 7

    for ib in range(len(bins)):

        pb = np.array([[y[ib] for y in x] for x in psd])
        fq = pb[0,0,0,0]
        p   = np.mean(pb[:,:,:,1], 0)
        ps  = np.std(pb[:,:,:,1], 0)
        pe  = np.mean(pb[:,:,:,2], 0)
        pn  = np.mean(pb[:,:,:,3], 0)

        ax = plt.subplot(4, len(bins), ib+1)
        plt.loglog(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:])
        plt.errorbar(fq, p[0,0], ps[0,0], label='lin, no-taper', lw=0.5)
        plt.fill_between(fq, p[0,0]-pe[0,0], p[0,0]+pe[0,0], alpha=0.6)
        plt.plot(fq, pn[0,0])
        plt.legend(); plt.title('bin %d'%bins[ib]); ax.set_xlim([2e-4,0.5]); plt.ylim([50,5e5])

        ax = plt.subplot(4, len(bins), len(bins)+ib+1)
        plt.loglog(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:])
        plt.errorbar(fq, p[0,1], ps[0,1], label='log, no-taper', lw=0.5)
        plt.fill_between(fq, p[0,1]-pe[0,1], p[0,1]+pe[0,1], alpha=0.6)
        plt.plot(fq, pn[0,1])
        plt.legend(); ax.set_xlim([2e-4,0.5]); plt.ylim([50,5e5])

        ax = plt.subplot(4, len(bins), 2*len(bins)+ib+1)
        plt.loglog(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:])
        plt.errorbar(fq, p[1,0], ps[1,0], label='lin, taper', lw=0.5)
        plt.fill_between(fq, p[1,0]-pe[1,0], p[1,0]+pe[1,0], alpha=0.6)
        plt.plot(fq, pn[1,0])
        plt.legend(); ax.set_xlim([2e-4,0.5]); plt.ylim([50,5e5])

        ax = plt.subplot(4, len(bins), 3*len(bins)+ib+1)
        plt.loglog(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:])
        plt.errorbar(fq, p[1,1], ps[1,1], label='log, taper', lw=0.5)
        plt.fill_between(fq, p[1,1]-pe[1,1], p[1,1]+pe[1,1], alpha=0.6)
        plt.plot(fq, pn[1,1])
        plt.legend(); ax.set_xlim([2e-4,0.5]); plt.ylim([50,5e5])
        

        plt.title('bin:%d'%bins[ib])
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
    n    = 2**12
    dt   = 1.0
    mu   = 1000.0
    nsim = 200
    bins = [2, 10, 50]

    sim = az.SimLC(seed=12651)
    sim.add_model('powerlaw', [1e-2, -2.])


    psd = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n*4, dt, mu, norm='var')
        # use different sampling dt, so we reduce the noise a little bit
        x = np.random.poisson(sim.x[:n]*200)/200
        p = az.LCurve.calculate_psd(x, dt, 'var', taper=True, rerr=(x/200)**0.5)
        pb = []
        for b in bins:
            p1 = az.LCurve.bin_psd(p[0], p[1], fqbin={'by_n':[b,1]}, noise=p[2], logavg=False)
            p2 = az.LCurve.bin_psd(p[0], p[1], fqbin={'by_n':[b,1]}, noise=p[2], logavg=True)
            p1 = np.vstack([p1[:3], p1[3]['noise']])
            p2 = np.vstack([p2[:3], p2[3]['noise']])
            pb.append([p1,p2])
        psd.append(pb)

    plt.rcParams['figure.figsize'] = [14, 8]
    plt.rcParams['font.size'] = 7

    for ib in range(len(bins)):

        pb = np.array([x[ib] for x in psd])
        fq = pb[0,0,0]
        p   = np.mean(pb[:,:,1], 0)
        ps  = np.std(pb[:,:,1], 0)
        pp = np.percentile(pb[:,:,1],[16,100-16], 0)
        pe  = np.mean(pb[:,:,2], 0)
        pn  = np.mean(pb[:,:,3], 0)

        ax = plt.subplot(2, len(bins), ib+1)
        plt.loglog(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:])
        plt.errorbar(fq, p[0], ps[0], lw=0.5, label='lin')
        plt.fill_between(fq, p[0]-pe[0], p[0]+pe[0], alpha=0.6)
        plt.fill_between(fq, pp[0,0], pp[1,0], alpha=0.6, color='C2')
        plt.plot(fq, pn[0])
        plt.legend()
        plt.title('bin %d'%bins[ib]); ax.set_xlim([2e-4,0.5]); plt.ylim([5,5e5])

        ax = plt.subplot(2, len(bins), len(bins)+ib+1)
        plt.loglog(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:])
        plt.errorbar(fq, p[1], ps[1], lw=0.5, label='log')
        plt.fill_between(fq, p[1]-pe[1], p[1]+pe[0], alpha=0.6)
        plt.fill_between(fq, pp[0,1], pp[1,1], alpha=0.6, color='C2')
        plt.plot(fq, pn[1])
        plt.legend(); ax.set_xlim([2e-4,0.5]); plt.ylim([5,5e5])
        

        plt.title('bin:%d'%bins[ib])
    plt.savefig('png/bin_psd__4.png')


#####################


def phase__1():
    """Constant phase lag, no binning, no noise, RED NOISE LEAK

    TEST: phases from segments without noise, i.e with leak
    SUMMARY:
        1. taper helps alot! both in the scatter and bias
    """
    n    = 2**12
    dt   = 1.0
    mu   = 100.0
    lag  = 0.5
    nsim = 200

    sim = az.SimLC(seed=463284)
    sim.add_model('powerlaw', [1e-2, -2.])
    sim.add_model('constant', lag, lag=True)

    lag = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n*4, dt, mu, norm='var')
        sim.apply_lag(phase=True)
        l1 = az.LCurve.calculate_lag(sim.y[:n], sim.x[:n], dt, phase=True)
        l2 = az.LCurve.calculate_lag(sim.y[:n], sim.x[:n], dt, phase=True, taper=True)
        lag.append([l1,l2])

    lag = np.array(lag)
    fq  = lag[0,0,0]
    l   = lag[:,:,1].mean(0)
    lp  = np.percentile(lag[:,:,1],[50,16,100-16], 0)


    plt.rcParams['figure.figsize'] = [8, 4]
    plt.rcParams['font.size'] = 7

    ax = plt.subplot(121)
    plt.plot(fq, lp[0,0], color='C0'); plt.title('no-taper')
    plt.fill_between(fq, lp[1,0], lp[2,0], alpha=0.5, color='C1')
    plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:], color='C2')
    ax.set_xscale('log')

    ax = plt.subplot(122)
    plt.plot(fq, lp[0,1], color='C0'); plt.title('taper')
    plt.fill_between(fq, lp[1,1], lp[2,1], alpha=0.5, color='C1')
    plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:], color='C2')
    ax.set_xscale('log')

    plt.savefig('png/phase__1.png')


def phase__2():
    """Constant phase lag, no binning, NOISE, RED NOISE LEAK

    TEST: phases from segments (i.e. leak) with noise,
    SUMMARY:
        1. again, taper helps alot! The scatter is very small in intermediate frequencies
            not affected by noise, unlike in no-tpaer case.
    """
    n    = 2**12
    dt   = 1.0
    mu   = 100.0
    lag  = 0.5
    nsim = 200

    sim = az.SimLC(seed=4634)
    sim.add_model('powerlaw', [1e-2, -2.])
    sim.add_model('constant', lag, lag=True)

    lag = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n*4, dt, mu, norm='var')
        sim.apply_lag(phase=True)
        scale_fac = 100
        x  = np.random.poisson(sim.x[:n]*scale_fac)/scale_fac
        y  = np.random.poisson(sim.y[:n]*scale_fac)/scale_fac
        l1 = az.LCurve.calculate_lag(y, x, dt, phase=True)
        l2 = az.LCurve.calculate_lag(y, x, dt, phase=True, taper=True)
        lag.append([l1,l2])

    lag = np.array(lag)
    fq  = lag[0,0,0]
    l   = lag[:,:,1].mean(0)
    lp  = np.percentile(lag[:,:,1],[50,16,100-16], 0)


    plt.rcParams['figure.figsize'] = [8, 4]
    plt.rcParams['font.size'] = 7

    ax = plt.subplot(121)
    plt.plot(fq, lp[0,0], color='C0'); plt.title('no-taper')
    plt.fill_between(fq, lp[1,0], lp[2,0], alpha=0.5, color='C1')
    plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:], color='C2')
    ax.set_xscale('log')

    ax = plt.subplot(122)
    plt.plot(fq, lp[0,1], color='C0'); plt.title('taper')
    plt.fill_between(fq, lp[1,1], lp[2,1], alpha=0.5, color='C1')
    plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:], color='C2')
    ax.set_xscale('log')

    plt.savefig('png/phase__2.png')


def phase__3():
    """Constant phase lag, BINNING, NOISE, RED NOISE LEAK

    TEST: leak with binning
    SUMMARY:
        1. no-taper is biased.
        2. taper has less scatter in intermediate frequencies and is not biased.
    """
    n    = 2**12
    dt   = 1.0
    mu   = 100.0
    lag  = 0.5
    nsim = 200
    bins = [2, 5, 10, 20, 50]

    sim = az.SimLC(seed=463444)
    sim.add_model('powerlaw', [1e-2, -2.])
    sim.add_model('constant', lag, lag=True)

    lag = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n*4, dt, mu, norm='var')
        sim.apply_lag(phase=True)
        scale_fac = 100
        x  = np.random.poisson(sim.x[:n]*scale_fac)/scale_fac
        y  = np.random.poisson(sim.y[:n]*scale_fac)/scale_fac
        lb = []
        for b in bins:
            l1 = az.LCurve.calculate_lag(y, x, dt, phase=True, fqbin={'by_n':[b, 1]},
                        rerr=(y/scale_fac)**0.5, Rerr=(x/scale_fac)**0.5)
            l2 = az.LCurve.calculate_lag(y, x, dt, phase=True, taper=True,
                fqbin={'by_n':[b, 1]}, rerr=(y/scale_fac)**0.5, Rerr=(x/scale_fac)**0.5)
            lb.append([l1,l2])
        lag.append(lb)


    plt.rcParams['figure.figsize'] = [14, 10]
    plt.rcParams['font.size'] = 7


    for ib in range(len(bins)):
        lb = np.array([[m[:3] for m in l[ib]] for l in lag])
        fq = lb[0,0,0]
        lp = np.percentile(lb[:,:,1],[50,16,100-16], 0)


        ax = plt.subplot(2, len(bins), ib+1)
        plt.plot(fq, lp[0,0], color='C0', lw=0.5)
        plt.fill_between(fq, lp[1,0], lp[2,0], alpha=0.5, color='C1')
        plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:], color='C2')
        ax.set_xscale('log'); ax.set_ylim([-1,1])
        plt.title('bin: %d, no-taper'%bins[ib])

    
        ax = plt.subplot(2, len(bins), len(bins)+ib+1)
        plt.plot(fq, lp[0,1], color='C0', lw=0.5); plt.title('taper')
        plt.fill_between(fq, lp[1,1], lp[2,1], alpha=0.5, color='C1')
        plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:], color='C2')
        ax.set_xscale('log'); ax.set_ylim([-1,1])

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
    n    = 2**12
    dt   = 1.0
    mu   = 100.0
    lag  = 0.5
    nsim = 200
    bins = [2, 5, 10, 20, 50]

    sim = az.SimLC(seed=463104)
    sim.add_model('powerlaw', [1e-2, -2.])
    sim.add_model('constant', lag, lag=True)

    lag = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n*4, dt, mu, norm='var')
        sim.apply_lag(phase=True)
        scale_fac = 1000
        x  = np.random.poisson(sim.x[:n]*scale_fac)/scale_fac
        y  = np.random.poisson(sim.y[:n]*scale_fac)/scale_fac
        lb = []
        for b in bins:
            l1 = az.LCurve.calculate_lag(y, x, dt, phase=True, fqbin={'by_n':[b, 1]},
                        rerr=(y/scale_fac)**0.5, Rerr=(x/scale_fac)**0.5)
            l1 = list(l1[:3]) + [l1[3][k] for k in ['coh', 'coh_e']]
            l2 = az.LCurve.calculate_lag(y, x, dt, phase=True, taper=True,
                fqbin={'by_n':[b, 1]}, rerr=(y/scale_fac)**0.5, Rerr=(x/scale_fac)**0.5)
            l2 = list(l2[:3]) + [l2[3][k] for k in ['coh', 'coh_e']]
            lb.append([l1,l2])
        lag.append(lb)


    plt.rcParams['figure.figsize'] = [14, 10]
    plt.rcParams['font.size'] = 7

    for ib in range(len(bins)):
        lb = np.array([[m for m in l[ib]] for l in lag])
        fq = lb[0,0,0]
        lp = np.percentile(lb,[50,16,100-16], 0)
        le = np.median(lb[:,:,2], 0)
        he = np.median(lb[:,:,4], 0)
        


        ax = plt.subplot(4, len(bins), ib+1)
        plt.plot(fq, lp[0,0,1], color='C0', lw=0.5)
        plt.fill_between(fq, lp[1,0,1], lp[2,0,1], alpha=0.5, color='C1')
        plt.fill_between(fq, lp[0,0,1]-le[0], lp[0,0,1]+le[0], alpha=0.5, color='C3')
        plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:], color='C2')
        ax.set_xscale('log'); ax.set_ylim([-1,1])
        plt.title('bin: %d, no-taper'%bins[ib])


        ax = plt.subplot(4, len(bins), len(bins)+ib+1)
        plt.plot(fq, lp[0,0,3], color='C0', lw=0.5)
        plt.fill_between(fq, lp[1,0,3], lp[2,0,3], alpha=0.5, color='C1')
        plt.fill_between(fq, lp[0,0,3]-he[0], lp[0,0,3]+he[0], alpha=0.5, color='C3')
        plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[0,1:]*0+1, color='C2')
        ax.set_xscale('log'); ax.set_ylim([-1,2])

    
        ax = plt.subplot(4, len(bins), 2*len(bins)+ib+1)
        plt.plot(fq, lp[0,1,1], color='C0', lw=0.5); plt.title('taper')
        plt.fill_between(fq, lp[1,1,1], lp[2,1,1], alpha=0.5, color='C1')
        plt.fill_between(fq, lp[0,1,1]-le[1], lp[0,1,1]+le[1], alpha=0.5, color='C3')
        plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:], color='C2')
        ax.set_xscale('log'); ax.set_ylim([-1,1])

        ax = plt.subplot(4, len(bins), 3*len(bins)+ib+1)
        plt.plot(fq, lp[0,1,3], color='C0', lw=0.5)
        plt.fill_between(fq, lp[1,1,3], lp[2,1,3], alpha=0.5, color='C1')
        plt.fill_between(fq, lp[0,1,3]-he[1], lp[0,1,3]+he[1], alpha=0.5, color='C3')
        plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[0,1:]*0+1, color='C2')
        ax.set_xscale('log'); ax.set_ylim([-1,2])

    plt.savefig('png/phase__4.png')


def lag__4():
    """Similar to phase__4, using lags

    SUMMRY:
        1. This is generally similar to the phase case, but
        2. The changing phase and phase wraps can mess things up. It could be
            worth doing calculations in phase.
    """
    n    = 2**14
    dt   = 1.0
    mu   = 100.0
    lag  = 30
    nsim = 200
    bins = [2, 10, 50]

    sim = az.SimLC(seed=463104)
    #sim.add_model('powerlaw', [1e-2, -2.])
    sim.add_model('broken_powerlaw', [5e-3, -1, -2, 1e-4])
    sim.add_model('constant', lag, lag=True)

    lag = []
    for isim in range(nsim):
        az.misc.print_progress(isim, nsim, isim==nsim-1)

        sim.simulate(n*4, dt, mu, norm='var')
        sim.apply_lag(phase=False)
        scale_fac = 1000
        x  = np.random.poisson(sim.x[:n]*scale_fac)/scale_fac
        y  = np.random.poisson(sim.y[:n]*scale_fac)/scale_fac
        lb = []
        for b in bins:
            l1 = az.LCurve.calculate_lag(y, x, dt, phase=False, fqbin={'by_n':[b, 1]},
                        rerr=(y/scale_fac)**0.5, Rerr=(x/scale_fac)**0.5)
            l1 = list(l1[:3]) + [l1[3][k] for k in ['coh', 'coh_e']]
            l2 = az.LCurve.calculate_lag(y, x, dt, phase=False, taper=True,
                fqbin={'by_n':[b, 1]}, rerr=(y/scale_fac)**0.5, Rerr=(x/scale_fac)**0.5)
            l2 = list(l2[:3]) + [l2[3][k] for k in ['coh', 'coh_e']]
            lb.append([l1,l2])
        lag.append(lb)


    plt.rcParams['figure.figsize'] = [14, 10]
    plt.rcParams['font.size'] = 7

    for ib in range(len(bins)):
        lb = np.array([[m for m in l[ib]] for l in lag])
        fq = lb[0,0,0]
        lp = np.percentile(lb,[50,16,100-16], 0)
        le = np.median(lb[:,:,2], 0)
        he = np.median(lb[:,:,4], 0)
        


        ax = plt.subplot(4, len(bins), ib+1)
        plt.plot(fq, lp[0,0,1], color='C0', lw=0.5)
        plt.fill_between(fq, lp[1,0,1], lp[2,0,1], alpha=0.5, color='C1')
        plt.fill_between(fq, lp[0,0,1]-le[0], lp[0,0,1]+le[0], alpha=0.5, color='C3')
        plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:], color='C2')
        ax.set_xscale('log'); ax.set_ylim([-100,100])
        plt.title('bin: %d, no-taper'%bins[ib])


        ax = plt.subplot(4, len(bins), len(bins)+ib+1)
        plt.plot(fq, lp[0,0,3], color='C0', lw=0.5)
        plt.fill_between(fq, lp[1,0,3], lp[2,0,3], alpha=0.5, color='C1')
        plt.fill_between(fq, lp[0,0,3]-he[0], lp[0,0,3]+he[0], alpha=0.5, color='C3')
        plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[0,1:]*0+1, color='C2')
        ax.set_xscale('log'); ax.set_ylim([-1,2])

    
        ax = plt.subplot(4, len(bins), 2*len(bins)+ib+1)
        plt.plot(fq, lp[0,1,1], color='C0', lw=0.5); plt.title('taper')
        plt.fill_between(fq, lp[1,1,1], lp[2,1,1], alpha=0.5, color='C1')
        plt.fill_between(fq, lp[0,1,1]-le[1], lp[0,1,1]+le[1], alpha=0.5, color='C3')
        plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[1,1:], color='C2')
        ax.set_xscale('log'); ax.set_ylim([-100,100])

        ax = plt.subplot(4, len(bins), 3*len(bins)+ib+1)
        plt.plot(fq, lp[0,1,3], color='C0', lw=0.5)
        plt.fill_between(fq, lp[1,1,3], lp[2,1,3], alpha=0.5, color='C1')
        plt.fill_between(fq, lp[0,1,3]-he[1], lp[0,1,3]+he[1], alpha=0.5, color='C3')
        plt.plot(sim.normalized_lag[0,1:], sim.normalized_lag[0,1:]*0+1, color='C2')
        ax.set_xscale('log'); ax.set_ylim([-1,2])

    plt.savefig('png/lag__4.png')



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


    p.add_argument('--calculate_psd__1', action='store_true', default=False,
            help="Simple psd simulation. Test fft calls.")
    p.add_argument('--calculate_psd__2', action='store_true', default=False,
            help="Simple psd simulation. Test noise level. ")
    p.add_argument('--calculate_psd__3', action='store_true', default=False,
            help="Simple psd simulation. Test tapering ")


    p.add_argument('--bin_psd__1', action='store_true', default=False,
            help="psd binning. Test log/linear averaging.")
    p.add_argument('--bin_psd__2', action='store_true', default=False,
            help="psd binning. Test log/linear averaging, with leak/taper.")
    p.add_argument('--bin_psd__3', action='store_true', default=False,
            help="psd binning. Test log/linear averaging, with leak/taper, with noise.")
    p.add_argument('--bin_psd__4', action='store_true', default=False,
            help="psd binning. test formula errors")

    p.add_argument('--phase__1', action='store_true', default=False,
            help="constant phase. leak")
    p.add_argument('--phase__2', action='store_true', default=False,
            help="constant phase. leak & noise")
    p.add_argument('--phase__3', action='store_true', default=False,
            help="constant phase. leak, noise & binning")
    p.add_argument('--phase__4', action='store_true', default=False,
            help="constant phase. error formula")

    p.add_argument('--lag__4', action='store_true', default=False,
            help="constant lag. error formula, similar to phase__4")


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


    ################

    # simple psd; fft calls #
    if args.calculate_psd__1: calculate_psd__1()

    # simple psd; noise level #
    if args.calculate_psd__2: calculate_psd__2()

    # simple psd; tapering #
    if args.calculate_psd__3: calculate_psd__3()


    ################

    # psd binning; log vs linear #
    if args.bin_psd__1: bin_psd__1()

    # psd binning; log vs linear with leak/taper #
    if args.bin_psd__2: bin_psd__2()

    # psd binning; log vs linear with leak/taper and noise #
    if args.bin_psd__3: bin_psd__3()

    # psd binning; formula error #
    if args.bin_psd__4: bin_psd__4()


    ################

    # constant phase; leak #
    if args.phase__1: phase__1()

    # constant phase; leak & noise #
    if args.phase__2: phase__2()

    # constant phase; leak, noise & binnings#
    if args.phase__3: phase__3()

    # constant phase; error formula 
    if args.phase__4: phase__4()


    # constant lag; error formula 
    if args.lag__4: lag__4()
