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
        ax = plt.subplot(1,2,iplt+1)
        ax.set_xscale('log'); ax.set_yscale('log')
        plt.errorbar(p[0,0], p[:,1].mean(0), p[:,1].std(0), 
            fmt='o', ms=3, alpha=0.5) 
        plt.plot(sim.normalized_psd[0,1:], sim.normalized_psd[1,1:], lw=0.5)
    #plt.show();exit(0)
    plt.savefig('png/psd_5.png')





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
