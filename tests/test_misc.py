

import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import aztools as az


class miscTest(unittest.TestCase):
    """testing misc."""

    def test_split_array_1(self):
        """simple splitting
        t = [0,1,2,3,4,5]
        """
        t = np.arange(6, dtype=np.double)
        s,i = az.misc.split_array(t, 2)
        b = [[0,1], [2,3], [4,5]]
        for i in range(len(s)):
            np.testing.assert_array_almost_equal(s[i], b[i])

        # strict=False #
        s,i = az.misc.split_array(t, 4, strict=False)
        b = [[0,1,2,3], [4,5]]
        for i in range(len(s)):
            np.testing.assert_array_almost_equal(s[i], b[i])

        # strict = True #
        s,i = az.misc.split_array(t, 4, strict=True)
        b = [[0,1,2,3]]
        for i in range(len(s)):
            np.testing.assert_array_almost_equal(s[i], b[i])


    def test_split_array_2(self):
        """simple splitting
        t = [0,1,nan,3,4,5]
        """
        t = np.arange(6, dtype=np.double)
        t[2] = np.nan
        s,i = az.misc.split_array(t, 2)
        b = [[0,1], [3,4], [5]]
        for i in range(len(s)):
            np.testing.assert_array_almost_equal(s[i], b[i])

        # strict=true #
        s,i = az.misc.split_array(t, 2, strict=1)
        b = [[0,1], [3,4]]
        for i in range(len(s)):
            np.testing.assert_array_almost_equal(s[i], b[i])

        # another nan at index=0 #
        t[0] = np.nan
        s,i = az.misc.split_array(t, 2)
        b = [[1], [3,4], [5]]
        for i in range(len(s)):
            np.testing.assert_array_almost_equal(s[i], b[i])

        # split_at_gaps = False #
        s,i = az.misc.split_array(t, 2, split_at_gaps=0)
        b = [[np.nan, 1], [np.nan,3], [4,5]]
        for i in range(len(s)):
            np.testing.assert_array_almost_equal(s[i], b[i])


    def test_split_array_3(self):
        """simple splitting
        t = [0,1,nan,3,4,5]
        """
        t = np.arange(6, dtype=np.double)
        t[2] = np.nan
        s,i = az.misc.split_array(t, 2, overlap=1)
        b = [[0,1], [1], [3,4], [4,5], [5]]
        for i in range(len(s)):
            np.testing.assert_array_almost_equal(s[i], b[i])

        # strict = 1 #
        s,i = az.misc.split_array(t, 2, overlap=1, strict=1)
        b = [[0,1], [3,4], [4,5]]
        for i in range(len(s)):
            np.testing.assert_array_almost_equal(s[i], b[i])


