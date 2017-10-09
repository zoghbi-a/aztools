
import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import aztools as az


class SimLCTest(unittest.TestCase):
    """testing SimLC."""


    def test_powerlaw_psd(self):
        sim = az.SimLC()
        sim.add_model('powerlaw', [1e-2, -2])
        freq = np.arange(1, 6)/6.
        p = sim.calculate_model(freq)
        np.testing.assert_array_almost_equal(p, 1e-2 * freq**-2)


    def test_broekn_powerlaw_psd(self):
        sim = az.SimLC()
        sim.add_model('broken_powerlaw', [1e-2, -1, -2, 0.5])
        freq = np.arange(1, 6)/6.
        p = sim.calculate_model(freq)
        pp = np.zeros_like(freq)
        ii = freq<0.5
        pp[ii] = (freq[ii]/0.5) ** -1
        pp[~ii] = (freq[~ii]/0.5) ** -2
        pp *= 1e-2 * 0.5**-2
        np.testing.assert_array_almost_equal(p, pp)


    def test_adding_pl_bpl(self):
        sim = az.SimLC()
        sim.add_model('powerlaw', [1e-2, -2])
        sim.add_model('broken_powerlaw', [1e-2, -1, -2, 0.5], False)
        freq = np.arange(1, 6)/6.
        p = sim.calculate_model(freq)
        pp = np.zeros_like(freq)
        ii = freq<0.5
        pp[ii] = (freq[ii]/0.5) ** -1
        pp[~ii] = (freq[~ii]/0.5) ** -2
        pp *= 1e-2 * 0.5**-2
        pp += 1e-2 * freq**-2
        np.testing.assert_array_almost_equal(p, pp)

    
    def test_adding_step(self):
        sim = az.SimLC()
        sim.add_model('step', [[2/12, 4./12, 8/12.], [2., 1]])
        freq = np.arange(1, 12)/12.
        p = sim.calculate_model(freq)
        pp = np.array([2]*4 + [1.]*7)
        np.testing.assert_array_almost_equal(p, pp)


    def test_simulate(self):
        sim = az.SimLC()
        sim.add_model('powerlaw', [1e-2, -2])
        n, mu = 128, 100.

        norms = ['var', 'leahy', 'rms']
        expos = [0, 1, 2]
        for norm, expo in zip(norms, expos):
            sim.simulate(n, 1.0, mu, norm=norm)
            p = (2./(n*mu**expo)) * np.abs(np.fft.rfft(sim.x))**2
            np.testing.assert_almost_equal(
                (p[1:]/sim.normalized_psd[1][1:]).mean(),1,0)


    def test_lag_array(self):
        sim = az.SimLC()
        sim.add_model('powerlaw', [1e-2, -2])
        n, mu = 64, 100.
        sim.simulate(n, 1.0, mu, norm='var')

        lag = np.zeros(len(sim.x)//2+1) + 4
        y = sim.lag_array(sim.x, lag, False, sim.normalized_psd[0])
        np.testing.assert_array_almost_equal(sim.x[:-4], y[4:])


    def test_apply_lag(self):
        sim = az.SimLC()
        sim.add_model('powerlaw', [1e-2, -2])
        n, mu = 64, 100.
        sim.simulate(n, 1.0, mu, norm='var')

        sim.add_model('constant', 4, lag=True)
        sim.apply_lag(phase=False)
        np.testing.assert_array_almost_equal(sim.x[:-4], sim.y[4:])

