# ############################################################################
# dirac_stream.py
# =======
# Authors : Adrien Besson [adribesson@gmail.com] and Matthieu Simeoni [matthieu.simeoni@gmail.com]
# ############################################################################
"""
Routines for generating and sampling Dirac streams.
"""

import numpy as np
from pyoneer.utils.math import sinc


def rnd_innovations(dirac_count: int, t_start: float = 0, t_end: float = 1, intensity_distribution: str = 'uniform',
                    width_intensity: float = 1, mean_intensity: float = 0, sigma_intensity: float = 1, df: int = 1,
                    relative_minimal_distance: float = None, seed: int = 1):
    """
    Generate Dirac innovations.
    :param dirac_count: int
    Number of diracs.
    :param t_start: float
    Start of the period interval.
    :param t_end: float
    End of the period interval.
    :param intensity_distribution: str
    One of {uniform, chisquare, lognormal}. Specifies the distribution for random intensities. The last two
    distributions produce sources with positive intensities only.
    :param width_intensity: float
    Width of interval for uniform distribution.
    :param mean_intensity: float
    Mean for uniform and lognormal distributions.
    :param sigma_intensity: float
    Standard deviation for normal and lognormal distributions.
    :param df: int
    Number of degrees of freedom for chi square distribution.
    :param seed: int
    Seed number for reproducibility
    :return: np.ndarray[dirac_count,], np.ndarray[dirac_count,]
    Locations and intensities of the Diracs.
    """

    # Fix the seed number
    np.random.seed(seed=seed)

    if intensity_distribution == 'uniform':
        intensities = width_intensity * np.random.random(size=(dirac_count,)) + (mean_intensity - 1 / 2)
    elif intensity_distribution == 'chisquare':
        intensities = np.random.chisquare(df=df, size=(dirac_count,))
    elif intensity_distribution == 'lognormal':
        intensities = np.random.lognormal(mean=mean_intensity, sigma=sigma_intensity, size=(dirac_count,))
    else:
        return print('Unsupported intensity distribution. Choose from [uniform, chisquare, lognormal].')
    if relative_minimal_distance is None:
        locations = (t_end - t_start) * np.random.random(size=(dirac_count,)) + t_start
    else:
        grid = np.arange(t_start, t_end, relative_minimal_distance * (t_end - t_start))
        locations = grid[np.random.permutation(np.arange(grid.size))[:dirac_count].astype(int)].reshape(-1)
    return locations, intensities


def fourier_series_coefficients(M: int, locations: np.ndarray, intensities: np.ndarray, period: float) -> np.ndarray:
    """
    Return the 2M+1 Fourier series coefficients of a stream of Diracs with specified period and innovations.
    :param M: int
    Maximum order of computed coefficients.
    :param locations: np.ndarray[K,]
    Locations of Diracs.
    :param intensities: np.ndarray[K,]
    Amplitudes of Diracs.
    :param period: float
    Period of the stream of Diracs.
    :return fs_coeffs: complex np.ndarray[2M+1,]
    Fourier series coefficients.
    """
    samples_m = np.arange(-M, M + 1)
    fs_coeffs = (1 / period) * np.sum(
        intensities * np.exp(-1j * 2 * np.pi * samples_m[:, None] * locations[None, :] / period), axis=-1)
    return fs_coeffs.reshape(-1)


def sinc_samples(samp_locs: np.ndarray, source_locs: np.ndarray, intensities: np.ndarray, M: int, period: float):
    """
    Compute non-uniform low-pass samples of Dirac stream
    :param samp_locs: np.ndarray[L,]
    Sample locations.
    :param source_locs: np.ndarray[K,]
    Dirac locations.
    :param intensities:  np.ndarray[K,]
    Dirac intensities.
    :param M: int
    Bandwidth of low-pass filter.
    :param period: float
    Period of Dirac stream
    :return: np.ndarray[L,]
    Non-uniform low-pass samples.
    """
    time_samples = np.sum(sinc(samp_locs[:, None] - source_locs[None, :], M, period=period) * intensities[None, :],
                          axis=-1)
    return time_samples
