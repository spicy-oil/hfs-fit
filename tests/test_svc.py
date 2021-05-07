"""Integration tests for the service as a whole."""

import os
from unittest.mock import patch

from numpy import testing
from hfs_fit import hfs


<<<<<<< HEAD
@patch('hfs_fit.hfs_fit.get_user_levels')
@patch('hfs_fit.hfs_fit.get_user_wavenumber')
@patch('hfs_fit.hfs_fit.get_user_noise')
=======
@patch('hfs_fit.get_user_levels')
@patch('hfs_fit.get_user_wavenumber')
@patch('hfs_fit.get_user_noise')
>>>>>>> b10b16cc946083ac2d3cf37242d37dcf2391a94d
def test_hfs(mock_user_noise, mock_user_wavenumber, mock_user_levels):
    """Run a full test of the script."""
    # setup
    mock_user_levels.return_value = 2, 2
    mock_user_noise.return_value = 37945, 37975
    mock_user_wavenumber.return_value = 'z5S2', 'a5P2', 37978, 37980

    # run svc
    obj = hfs('tests/sample_spectrum.txt', 'tests/fitLog.xlsx', nuclearSpin = 3.5)
    obj.NewFit()
    obj.PlotGuess()
    obj.Optimise(2)

    # validate
    testing.assert_almost_equal(obj.SNR, 52.386236188012326)
    testing.assert_almost_equal(obj.normFactor, 3.90336975182)
    testing.assert_almost_equal(obj.relIntensities[0], 0.16923077)
    testing.assert_almost_equal(obj.relIntensities[-2], 0.26923077)
    testing.assert_almost_equal(obj.relIntensities[-1], 1.)
    testing.assert_almost_equal(obj.fitParams[0], -5.03268524e-02)
    testing.assert_almost_equal(obj.fitParams[-2], 3.79790274e+04, decimal=3)
<<<<<<< HEAD

=======
    
>>>>>>> b10b16cc946083ac2d3cf37242d37dcf2391a94d
