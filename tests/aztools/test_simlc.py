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


    def test_add_model__powerlaw__wrong_npar(self):
        with self.assertRaises(ValueError):
            self.sim.add_model('powerlaw', [1., 2., 3])


    def test_add_model__powerlaw(self):
        sim = SimLC(seed=334)
        pars = [1., 2.]
        sim.add_model('powerlaw', pars, lag=False)

        self.assertEqual(sim.psd_models[0][0], SimLC.powerlaw)
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


    def test_add_model__broken_powerlaw__wrong_npar(self):
        with self.assertRaises(ValueError):
            self.sim.add_model('broken_powerlaw', [1., 2., 3])


    def test_add_model__broken_powerlaw(self):
        sim = SimLC(seed=334)
        pars = [1., 2., 3., 4.]
        sim.add_model('broken_powerlaw', pars, lag=False)

        self.assertEqual(sim.psd_models[0][0], SimLC.broken_powerlaw)
        self.assertEqual(sim.psd_models[0][1], pars)


    def test_add_model__bending_powerlaw__wrong_npar(self):
        with self.assertRaises(ValueError):
            self.sim.add_model('bending_powerlaw', [1., 2.])


    def test_add_model__bending_powerlaw(self):
        sim = SimLC(seed=334)
        pars = [1., 2., 3.]
        sim.add_model('bending_powerlaw', pars, lag=False)

        self.assertEqual(sim.psd_models[0][0], SimLC.bending_powerlaw)
        self.assertEqual(sim.psd_models[0][1], pars)


    def test_add_model__lorentz__wrong_npar(self):
        with self.assertRaises(ValueError):
            self.sim.add_model('lorentz', [1., 2.])


    def test_add_model__lorentz(self):
        sim = SimLC(seed=334)
        pars = [1., 2., 3.]
        sim.add_model('lorentz', pars, lag=False)

        self.assertEqual(sim.psd_models[0][0], SimLC.lorentz)
        self.assertEqual(sim.psd_models[0][1], pars)


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
