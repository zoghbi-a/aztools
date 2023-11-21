"""Test for aztools.misc"""
import os
import unittest

import numpy as np
from astropy.io import fits

from aztools import LCurve, misc


class MiscTest(unittest.TestCase):
    """testing aztools misc"""

    def test_split_array_1(self):
        """simple splitting
        t = [0,1,2,3,4,5]
        """
        tarr = np.arange(6, dtype=np.double)
        sarr,_ = misc.split_array(tarr, 2)
        barr = [[0,1], [2,3], [4,5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # strict=False #
        sarr,_ = misc.split_array(tarr, 4, strict=False)
        barr = [[0,1,2,3], [4,5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # strict = True #
        sarr,_ = misc.split_array(tarr, 4, strict=True)
        barr = [[0,1,2,3]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # approx = True #
        sarr,_ = misc.split_array(tarr, 4, approx=True)
        barr = [[0,1,2], [3,4,5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # approx = True #
        sarr,_ = misc.split_array(tarr, 5, approx=True)
        barr = [[0,1,2,3,4,5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])


    def test_split_array_2(self):
        """simple splitting
        tarr = [0,1,nan,3,4,5]
        """
        tarr = np.arange(6, dtype=np.double)
        tarr[2] = np.nan
        sarr,_ = misc.split_array(tarr, 2)
        barr = [[0,1], [3,4], [5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # approx=true #
        sarr,_ = misc.split_array(tarr, 2, approx=True)
        barr = [[0,1], [3,4], [5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # strict=true #
        sarr,_ = misc.split_array(tarr, 2, strict=1)
        barr = [[0,1], [3,4]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # another nan at index=0 #
        tarr[0] = np.nan
        sarr,_ = misc.split_array(tarr, 2)
        barr = [[1], [3,4], [5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # split_at_gaps = False #
        sarr,_ = misc.split_array(tarr, 2, split_at_gaps=0)
        barr = [[np.nan, 1], [np.nan,3], [4,5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])


    def test_split_array_3(self):
        """simple splitting
        tarr = [0,1,nan,3,4,5]
        """
        tarr = np.arange(6, dtype=np.double)
        tarr[2] = np.nan
        sarr,_ = misc.split_array(tarr, 2, overlap=1)
        barr = [[0,1], [1], [3,4], [4,5], [5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # strict = 1 #
        sarr,_ = misc.split_array(tarr, 2, overlap=1, strict=1)
        barr = [[0,1], [3,4], [4,5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])


    def test_group_array_1(self):
        """[0,1,2,3,4,5,6,7,8]"""

        xarr = np.arange(9)
        igrp = misc.group_array(xarr, by_n=[1, 1])
        grp = [[idx] for idx in xarr]
        for idx,_igrp in enumerate(igrp):
            np.testing.assert_array_almost_equal(_igrp, grp[idx])

        igrp = misc.group_array(xarr, by_n=[4, 1])
        grp = [[0,1,2,3], [4,5,6,7], [8]]
        for idx,_igrp in enumerate(igrp):
            np.testing.assert_array_almost_equal(_igrp, grp[idx])

        igrp = misc.group_array(xarr, by_n=[3, 2])
        grp = [[0,1,2], [3,4,5,6,7,8]]
        for idx,_igrp in enumerate(igrp):
            np.testing.assert_array_almost_equal(_igrp, grp[idx])


    def test_group_array_2(self):
        """[0,0,2,3,4,4,6,7,8]"""

        xarr = np.arange(9)
        xarr[1] = 0
        xarr[5] = 4
        igrp = misc.group_array(xarr, by_n=[1, 1])
        grp = [[0,1],[2],[3],[4,5],[6],[7],[8]]
        for idx,_igrp in enumerate(igrp):
            np.testing.assert_array_almost_equal(_igrp, grp[idx])

        igrp = misc.group_array(xarr, by_n=[4, 1])
        grp = [[0,1,2,3], [4,5,6,7], [8]]
        for idx,_igrp in enumerate(igrp):
            np.testing.assert_array_almost_equal(_igrp, grp[idx])

        # [0,0,0,0,4,4,6,7,8]
        xarr[2] = 0
        xarr[3] = 0
        igrp = misc.group_array(xarr, by_n=[2, 1.5])
        grp = [[0,1,2,3],[4,5,6], [7,8]]
        for idx,_igrp in enumerate(igrp):
            np.testing.assert_array_almost_equal(_igrp, grp[idx])


    def test_group_array_3(self):
        """[0,0,2,3,4,4,6,7,8]; bins"""

        xarr = np.arange(9)
        xarr[1] = 0
        xarr[5] = 4
        igrp = misc.group_array(xarr, bins=[0, 4, 8])
        grp = [range(4), [4,5,6,7]]
        for idx,_igrp in enumerate(igrp):
            np.testing.assert_array_almost_equal(_igrp, grp[idx])


    def test_sync_lcurve_1(self):
        """test misc.sync_lcurve"""
        t1arr = np.array([0,1,3,5])
        t2arr = np.array([0,2,3,4])

        sync = misc.sync_lcurve([t1arr.repeat(3).reshape(-1,3).T,
                                 t2arr.repeat(3).reshape(-1,3).T], tbase=None)
        np.testing.assert_array_almost_equal(
            sync, [np.array([[0,3], [0,3], [0,3]]),
                   np.array([[0,3], [0,3], [0,3]])]
        )

        # lists
        sync = misc.sync_lcurve([[t1arr, t1arr, t1arr],
                                 [t2arr, t2arr, t2arr]], tbase=None)
        np.testing.assert_array_almost_equal(
            sync, [np.array([[0,3], [0,3], [0,3]]),
                   np.array([[0,3], [0,3], [0,3]])]
        )

        # LCurve
        lcrv1 = LCurve(t1arr, t1arr, rerr=t1arr)
        lcrv2 = LCurve(t2arr, t2arr, rerr=t2arr)
        sync = misc.sync_lcurve([lcrv1, lcrv2], tbase=None)
        np.testing.assert_array_almost_equal(
            sync, [np.array([[0,3], [0,3], [0,3]]),
                   np.array([[0,3], [0,3], [0,3]])]
        )


    def test_lcurve_to_segments_1(self):
        """test misc.lcurve_to_segments"""
        tarr = np.arange(8)
        rarr = tarr*2
        rerr = tarr*0.1
        lcrv = LCurve(tarr, rarr, rerr=rerr, deltat=1.0)

        seg = misc.lcurve_to_segments(lcrv, seglen=3.0, strict=True)
        np.testing.assert_array_almost_equal(seg[0], [rarr[:3], rarr[3:6]])
        np.testing.assert_array_almost_equal(seg[1], [rerr[:3], rerr[3:6]])
        np.testing.assert_array_almost_equal(seg[2], [tarr[:3], tarr[3:6]])
        np.testing.assert_array_almost_equal(seg[3][0], np.array([[0,1,2], [3,4,5]]))

        # strict=False
        seg = misc.lcurve_to_segments(lcrv, seglen=3.0, strict=False)
        np.testing.assert_array_almost_equal(seg[0][-1], rarr[6:])
        np.testing.assert_array_almost_equal(seg[1][-1], rerr[6:])
        np.testing.assert_array_almost_equal(seg[2][-1], tarr[6:])
        np.testing.assert_array_almost_equal(seg[3][0][-1], np.array([6,7]))


    def test_lcurve_to_segments_2(self):
        """test misc.lcurve_to_segments"""
        tarr = np.array([0, 1, 4, 5, 6, 7])
        rarr = tarr*2
        rerr = tarr*0.1
        lcrv = LCurve(tarr, rarr, rerr=rerr, deltat=1.0)

        seg = misc.lcurve_to_segments(lcrv, seglen=3.0, strict=True, uneven=False)
        np.testing.assert_array_almost_equal(seg[0], [np.array([4,5,6])*2])

        seg = misc.lcurve_to_segments(lcrv, seglen=3.0, strict=False, uneven=False)
        for idx in [0,1,2]:
            np.testing.assert_array_almost_equal(
                seg[0][idx], [np.array([0,1])*2, np.array([4,5,6])*2, np.array([7])*2][idx])

        seg = misc.lcurve_to_segments(lcrv, seglen=3.0, strict=True, uneven=True)
        np.testing.assert_array_almost_equal(seg[0], [np.array([0,1,4])*2, np.array([5,6,7])*2])


    def test_read_fits_lcurve(self):
        """test read_fits_lcurve; write one and re-read it"""
        tarr  = np.arange(4)
        xarr  = np.arange(4)
        xerr = np.arange(1, 5)*1.
        hdu = fits.BinTableHDU.from_columns([
                fits.Column(name='TIME', format='E', array=tarr),
                fits.Column(name='RATE', format='E', array=xarr),
                fits.Column(name='ERROR', format='E', array=xerr),
            ])
        hdu.name = 'RATE'
        fname = 'tmp.fits'
        hdu.writeto(fname, overwrite=True)
        lcrv, _ = misc.read_fits_lcurve(fname)
        np.testing.assert_array_almost_equal(tarr,  lcrv[0])
        np.testing.assert_array_almost_equal(xarr,  lcrv[1])
        np.testing.assert_array_almost_equal(xerr, lcrv[2])
        if os.path.exists(fname):
            os.remove(fname)


    @staticmethod
    def _myf(avar, bvar=1.0):
        """_myf docs"""
        return avar+bvar
    @staticmethod
    def _myf2(avar, bvar=1.0, irun=None):
        irun = 0
        return avar+bvar+irun

    def test_parallelize_input(self):
        """test the input"""
        _myf_p = misc.parallelize(self._myf)
        with self.assertRaises(ValueError):
            _myf_p(1.0)

        with self.assertRaises(ValueError):
            _myf_p(1.0, 4.0)

        with self.assertRaises(ValueError):
            _myf_p([1.0,2.0], [3.0,4.,5.])

        with self.assertRaises(ValueError):
            _myf_p([1.0,2.0], bvar=[3.0,4.,5.])


    def test_parallelize_use_irun(self):
        """test use_run"""
        _myf_p = misc.parallelize(self._myf, use_irun=False)
        np.testing.assert_array_almost_equal(_myf_p([1.0,2.0], bvar=[3.0,4.]), [4., 6.])

        _myf_p = misc.parallelize(self._myf, use_irun=True)
        with self.assertRaises(TypeError):
            _myf_p([1.0,2.0], bvar=[3.0,4.])

        _myf_p = misc.parallelize(self._myf2, use_irun=True)
        np.testing.assert_array_almost_equal(_myf_p([1.0,2.0], bvar=[3.0,4.]), [4., 6.])


    def test_parallelize_docstring(self):
        """test __doc__"""
        _myf_p = misc.parallelize(self._myf)
        assert '_myf docs' in _myf_p.__doc__
        assert 'irun: None, int or list' in _myf_p.__doc__
