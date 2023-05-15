import unittest

import numpy as np

from aztools import utils


class UtilsTest(unittest.TestCase):
    """testing aztools utils"""

    def test_split_array_1(self):
        """simple splitting
        t = [0,1,2,3,4,5]
        """
        tarr = np.arange(6, dtype=np.double)
        sarr,_ = utils.split_array(tarr, 2)
        barr = [[0,1], [2,3], [4,5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # strict=False #
        sarr,_ = utils.split_array(tarr, 4, strict=False)
        barr = [[0,1,2,3], [4,5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # strict = True #
        sarr,_ = utils.split_array(tarr, 4, strict=True)
        barr = [[0,1,2,3]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # approx = True #
        sarr,_ = utils.split_array(tarr, 4, approx=True)
        barr = [[0,1,2], [3,4,5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # approx = True #
        sarr,_ = utils.split_array(tarr, 5, approx=True)
        barr = [[0,1,2,3,4,5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])


    def test_split_array_2(self):
        """simple splitting
        tarr = [0,1,nan,3,4,5]
        """
        tarr = np.arange(6, dtype=np.double)
        tarr[2] = np.nan
        sarr,_ = utils.split_array(tarr, 2)
        barr = [[0,1], [3,4], [5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # approx=true #
        sarr,_ = utils.split_array(tarr, 2, approx=True)
        barr = [[0,1], [3,4], [5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # strict=true #
        sarr,_ = utils.split_array(tarr, 2, strict=1)
        barr = [[0,1], [3,4]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # another nan at index=0 #
        tarr[0] = np.nan
        sarr,_ = utils.split_array(tarr, 2)
        barr = [[1], [3,4], [5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # split_at_gaps = False #
        sarr,_ = utils.split_array(tarr, 2, split_at_gaps=0)
        barr = [[np.nan, 1], [np.nan,3], [4,5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])


    def test_split_array_3(self):
        """simple splitting
        tarr = [0,1,nan,3,4,5]
        """
        tarr = np.arange(6, dtype=np.double)
        tarr[2] = np.nan
        sarr,_ = utils.split_array(tarr, 2, overlap=1)
        barr = [[0,1], [1], [3,4], [4,5], [5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])

        # strict = 1 #
        sarr,_ = utils.split_array(tarr, 2, overlap=1, strict=1)
        barr = [[0,1], [3,4], [4,5]]
        for idx,_sarr in enumerate(sarr):
            np.testing.assert_array_almost_equal(_sarr, barr[idx])


    def test_group_array_1(self):
        # [0,1,2,3,4,5,6,7,8]

        xarr = np.arange(9)
        igrp = utils.group_array(xarr, by_n=[1, 1])
        grp = [[idx] for idx in xarr]
        for idx,_igrp in enumerate(igrp):
            np.testing.assert_array_almost_equal(_igrp, grp[idx])

        igrp = utils.group_array(xarr, by_n=[4, 1])
        grp = [[0,1,2,3], [4,5,6,7], [8]]
        for idx,_igrp in enumerate(igrp):
            np.testing.assert_array_almost_equal(_igrp, grp[idx])

        igrp = utils.group_array(xarr, by_n=[3, 2])
        grp = [[0,1,2], [3,4,5,6,7,8]]
        for idx,_igrp in enumerate(igrp):
            np.testing.assert_array_almost_equal(_igrp, grp[idx])



    def test_group_array_2(self):
        # [0,0,2,3,4,4,6,7,8]

        xarr = np.arange(9)
        xarr[1] = 0
        xarr[5] = 4
        igrp = utils.group_array(xarr, by_n=[1, 1])
        grp = [[0,1],[2],[3],[4,5],[6],[7],[8]]
        for idx,_igrp in enumerate(igrp):
            np.testing.assert_array_almost_equal(_igrp, grp[idx])

        igrp = utils.group_array(xarr, by_n=[4, 1])
        grp = [[0,1,2,3], [4,5,6,7], [8]]
        for idx,_igrp in enumerate(igrp):
            np.testing.assert_array_almost_equal(_igrp, grp[idx])

        # [0,0,0,0,4,4,6,7,8]
        xarr[2] = 0
        xarr[3] = 0
        igrp = utils.group_array(xarr, by_n=[2, 1.5])
        grp = [[0,1,2,3],[4,5,6], [7,8]]
        for idx,_igrp in enumerate(igrp):
            np.testing.assert_array_almost_equal(_igrp, grp[idx])


    def test_group_array_3(self):
        # [0,0,2,3,4,4,6,7,8]; bins

        xarr = np.arange(9)
        xarr[1] = 0
        xarr[5] = 4
        igrp = utils.group_array(xarr, bins=[0, 4, 8])
        grp = [range(4), [4,5,6,7]]
        for idx,_igrp in enumerate(igrp):
            np.testing.assert_array_almost_equal(_igrp, grp[idx])
