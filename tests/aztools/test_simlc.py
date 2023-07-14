import unittest

import numpy as np

from aztools import SimLC


class TestSimLC(unittest.TestCase):
    """Tests for SimLC"""

    @classmethod
    def setUpClass(cls):
        cls.sim = SimLC(seed=3443)

    def test_add_model__not_builting(self):
        with self.assertRaises(ValueError):
            self.sim.add_model('nomodel', [])


    def test_add_model__callable(self):
        def mymodel(freq, par):
            return freq + par[0]*0
        sim = SimLC(seed=334)
        pars = [1., 2.]
        sim.add_model(mymodel, pars)
        self.assertEqual(sim.psd_models[0][0], mymodel)
        self.assertEqual(sim.psd_models[0][1], pars)


    def test_add_model__user_array(self):
        sim = SimLC(seed=334)
        self.assertEqual(sim.psd_models, [])
        pars = [1., 2., 3.]
        sim.add_model('user_array', pars, lag=False)

        self.assertEqual(sim.psd_models[0][0], SimLC.user_array)
        self.assertEqual(sim.psd_models[0][1], pars)


    def test_add_model__wrong_npar(self):
        with self.assertRaises(ValueError):
            self.sim.add_model('powerlaw', [1., 2., 3])

        with self.assertRaises(ValueError):
            self.sim.add_model('broken_powerlaw', [1., 2., 3])

        with self.assertRaises(ValueError):
            self.sim.add_model('bending_powerlaw', [1., 2.])

        with self.assertRaises(ValueError):
            self.sim.add_model('lorentz', [1., 2.])


    def test_add_model(self):
        sim = SimLC(seed=334)
        pars = [1., 2.]
        sim.add_model('powerlaw', pars, lag=False)

        self.assertEqual(sim.psd_models[0][0], SimLC.powerlaw)
        self.assertEqual(sim.psd_models[0][1], pars)


        pars = [1., 2., 3., 4.]
        sim.add_model('broken_powerlaw', pars, clear=True, lag=False)

        self.assertEqual(sim.psd_models[0][0], SimLC.broken_powerlaw)
        self.assertEqual(sim.psd_models[0][1], pars)


        pars = [1., 2., 3.]
        sim.add_model('bending_powerlaw', pars, clear=True, lag=False)

        self.assertEqual(sim.psd_models[0][0], SimLC.bending_powerlaw)
        self.assertEqual(sim.psd_models[0][1], pars)


        pars = [1., 2., 3.]
        sim.add_model('lorentz', pars, clear=True, lag=False)

        self.assertEqual(sim.psd_models[0][0], SimLC.lorentz)
        self.assertEqual(sim.psd_models[0][1], pars)



    def test_add_model__powerlaw__lag(self):
        sim = SimLC(seed=334)
        pars = [1., 2.]
        sim.add_model('powerlaw', pars, lag=False)
        sim.add_model('powerlaw', pars, lag=True)

        self.assertEqual(sim.psd_models[0][0], SimLC.powerlaw)
        self.assertEqual(sim.psd_models[0][1], pars)
        self.assertEqual(sim.lag_models[0][0], SimLC.powerlaw)
        self.assertEqual(sim.lag_models[0][1], pars)


    def test_add_model_multiple_models(self):
        sim = SimLC(seed=334)
        sim.add_model('powerlaw', [1., 2], lag=False)
        sim.add_model('powerlaw', [-1., -2], lag=False, clear=False)

        self.assertEqual(len(sim.psd_models), 2)
        self.assertEqual(sim.psd_models[1][0], SimLC.powerlaw)
        self.assertEqual(sim.psd_models[1][1], [-1., -2.])


    def test_calculate_model__no_model_added(self):
        with self.assertRaises(ValueError):
            self.sim.calculate_model([1,2], [1,2])


    def test_calculate_model(self):
        sim = SimLC(seed=334)
        pars = [1., 2., 2.]
        sim.add_model('user_array', pars, lag=False)
        freq = np.array([0.1,0.2,0.3])
        mod = sim.calculate_model(freq, lag=False)
        self.assertTrue(np.all(sim.user_array(freq, pars) == mod))


    def test_powerlaw_psd(self):
        sim = SimLC(seed=321)
        sim.add_model('powerlaw', [1e-2, -2])
        freq = np.arange(1, 6)/6.
        model = sim.calculate_model(freq)
        np.testing.assert_array_almost_equal(model, 1e-2 * freq**-2)


    def test_broekn_powerlaw_psd(self):
        sim = SimLC()
        sim.add_model('broken_powerlaw', [1e-2, -1, -2, 0.5])
        freq = np.arange(1, 6)/6.
        mod1 = sim.calculate_model(freq)
        mod2 = np.zeros_like(freq)
        idx = freq<0.5
        mod2[idx] = (freq[idx]/0.5) ** -1
        mod2[~idx] = (freq[~idx]/0.5) ** -2
        mod2 *= 1e-2 * 0.5**-2
        np.testing.assert_array_almost_equal(mod1, mod2)


    def test_adding_pl_bpl(self):
        sim = SimLC()
        sim.add_model('powerlaw', [1e-2, -2])
        sim.add_model('broken_powerlaw', [1e-2, -1, -2, 0.5], clear=False)
        freq = np.arange(1, 6)/6.
        mod1 = sim.calculate_model(freq)
        mod2 = np.zeros_like(freq)
        idx = freq<0.5
        mod2[idx] = (freq[idx]/0.5) ** -1
        mod2[~idx] = (freq[~idx]/0.5) ** -2
        mod2 *= 1e-2 * 0.5**-2
        mod2 += 1e-2 * freq**-2
        np.testing.assert_array_almost_equal(mod1, mod2)


    def test_simulate(self):
        sim = SimLC()
        sim.add_model('powerlaw', [1e-2, -2])
        npoints, lcmean = 128, 100.

        norms = ['var', 'leahy', 'rms']
        expos = [0, 1, 2]
        for norm, expo in zip(norms, expos):
            sim.simulate(npoints, 1.0, lcmean, norm=norm)
            psd = (2./(npoints*lcmean**expo)) * np.abs(np.fft.rfft(sim.lcurve[1]))**2
            np.testing.assert_almost_equal(
                (psd[1:]/sim.normalized_psd[1][1:]).mean(),1,0)


    def test_lag_array(self):
        sim = SimLC()
        sim.add_model('powerlaw', [1e-2, -2])
        npoints, lcmean = 64, 100.
        sim.simulate(npoints, 1.0, lcmean, norm='var')

        lag = np.zeros(len(sim.lcurve[1])//2+1) + 4
        yarr = sim.lag_array(sim.lcurve[1], lag, False, sim.normalized_psd[0])
        np.testing.assert_array_almost_equal(sim.lcurve[1][:-4], yarr[4:])


    def test_apply_lag(self):
        sim = SimLC()
        sim.add_model('powerlaw', [1e-2, -2])
        npoints, lcmean = 64, 100.
        sim.simulate(npoints, 1.0, lcmean, norm='var')

        sim.add_model('constant', [4], lag=True)
        sim.apply_lag(phase=False)
        np.testing.assert_array_almost_equal(sim.lcurve[1][:-4], sim.lcurve[2][4:])
