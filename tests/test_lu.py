"""Tests for LU.py."""
import numpy as np
from numpy.testing import assert_almost_equal
from hfs_fit.LU import LU, inverse


def test_lu():
    '''Ensure LU decomposes the given matrix, and inverse works.'''
    A = np.array([[  2,   1,   0,   0,   0],
                  [  3,   8,   4,   0,   0],
                  [  0,   9,  20,  10,   0],
                  [  0,   0,  22,  51, -25],
                  [  0,   0,   0, -55,  60]])
    
    # check LU multiplied is the same as original
    decomposed = LU(A)
    recombined = decomposed[0] @ decomposed[1]
    assert_almost_equal(recombined, A)
    
    # check A @ inverse A = Identity
    A_1 = inverse(A)
    identity = A_1 @ A
    assert_almost_equal(identity, np.identity(len(A)))
