# ############################################################################
# fri.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# ############################################################################
"""
Routines for FRI reconstruction.
"""

from pyoneer.operators.linear_operator import choose_toeplitz_class, build_toeplitz_operator, FRISampling
import numpy as np
import scipy.sparse.linalg as scp
import scipy.linalg as splin
import scipy.optimize as scop
import astropy.units as u
from astropy.coordinates import Angle


def total_least_squares(fs_coeff: np.ndarray, K: int):
    """
    Performs total least-squares to recover annihilating filter from input Fourier series coefficients.
    :param fs_coeff: np.ndarray[2*M+1,]
    Fourier series coefficients.
    :param K: int
    Number of sources to recover.
    :return mu, annihilating_filter: complex, np.ndarray[K+1,]
    Eigenvalue (close to zero) and complex filter coefficients array.

    Note: If the solvers from `scipy.sparse.linalg` fail to converge, the less efficient but more robust `svd` routine from
    `scipy.linalg` is used.
    """
    M = int(fs_coeff.size // 2)
    toeplitz_class, method = choose_toeplitz_class(P=K, M=M, measure=True)
    Toeplitz_matrix = build_toeplitz_operator(P=K, M=M, x=fs_coeff, toeplitz_class='standard', method=method)
    try:
        if Toeplitz_matrix.shape[0] == Toeplitz_matrix.shape[1]:
            conj_sym_coeffs = np.array_equal(np.flip(fs_coeff), np.conj(fs_coeff))
            if conj_sym_coeffs:
                mu, annihilating_filter = scp.eigsh(Toeplitz_matrix, k=1, which='SM')
            else:
                mu, annihilating_filter = scp.eigs(Toeplitz_matrix, k=1, which='SM')
        else:
            _, mu, annihilating_filter = scp.svds(Toeplitz_matrix, k=1, which='SM', return_singular_vectors='vh')
            annihilating_filter = annihilating_filter.conj()
    except:
        u, s, vh = splin.svd(Toeplitz_matrix.mat, check_finite=False, full_matrices=False)
        annihilating_filter = vh[-1, :].conj()
        mu = s[-1]
    return mu, annihilating_filter.reshape(-1)


def roots_to_locations(annihilating_filter: np.ndarray, period: float) -> np.ndarray:
    """
    Compute roots of annihilating filter z-transform and maps them to locations on the period interval.
    :param annihilating_filter: np.ndarray[K+1,]
    Annihilating filter.
    :param period: float,
    :return locations: np.ndarray[K,]
    Dirac locations.
    """
    roots = np.roots(np.flip(annihilating_filter, axis=0).reshape(-1))
    locations = Angle(np.angle(roots) * u.rad)
    locations = locations.wrap_at(2 * np.pi * u.rad)
    return period * locations.value.reshape(-1) / (2 * np.pi)


def estimate_amplitudes(locations: np.ndarray, fs_coeff: np.ndarray, period: float, threshold: float = 1e-6,
                        regularisation: str = 'ridge', penalty: float = 0.2) -> np.ndarray:
    """
    Least-square estimates of the Dirac amplitudes for given locations and Fourier series coefficients.
    :param locations: np.ndarray[K,]
    Dirac locations.
    :param fs_coeff: np.ndarray[N,]
    Fourier series coefficients.
    :param period: float
    :param threshold: float
    Cutoff for eigenvalues in pinv computation.
    :param regularisation: str
    Type of regularisation.
    :param penalty: float
    Penalty strength.
    :return: np.ndarray[K,]
    Dirac amplitudes.
    """
    M = fs_coeff.size // 2
    frequencies = np.arange(-M, M + 1)
    vandermonde_mat = FRISampling(frequencies=frequencies, time_samples=locations,
                                  period=period).mat.conj().transpose() / (period ** 2)
    if regularisation is 'ridge':
        penalty = (1 + 1j) * penalty * (np.linalg.norm(vandermonde_mat) ** 2)
        gram = vandermonde_mat.conj().transpose() @ vandermonde_mat + penalty * np.eye(vandermonde_mat.shape[1],
                                                                                       vandermonde_mat.shape[1])
        gram_inv = np.linalg.pinv(gram, rcond=threshold)
        intensities = gram_inv @ (vandermonde_mat.conj().transpose() @ fs_coeff[:, None])
    else:
        vandermonde_pinv = np.linalg.pinv(vandermonde_mat, rcond=threshold)
        intensities = vandermonde_pinv @ fs_coeff[:, None]
    return np.real(intensities)


def match_to_ground_truth(true_locations: np.ndarray, estimated_locations: np.ndarray, period: float):
    """
    Match estimated sources to ground truth with a bipartite graph matching algorithm.
    :param true_locations: np.ndarray[K,],true  dirac locations.
    :param estimated_locations: np.ndarray[K,], estimated dirac locations.
    :return: estimated locations is reordered to match true locations. Average cost of matching also returned
    (positionning error).
    """
    true_locations = true_locations.reshape(-1)
    distance = np.abs(true_locations[:, None] - estimated_locations[None, :])
    cost = np.fmin(distance, period - distance)
    row_ind, col_ind = scop.linear_sum_assignment(cost)
    return estimated_locations[col_ind], cost[row_ind, col_ind].mean()


def coeffs_to_matched_diracs(fs_coeff: np.ndarray, K: int, period: float, locations: np.ndarray):
    """
    Get Dirac locations from Fourier coefficients by sequentially running the routines `total_least_squares`,
    `roots_to_locations` and `match_to_ground_truth`.
    :param fs_coeff: np.ndarray
    FS coefficients.
    :param K: int
    Number of Diracs.
    :param period: float
    :param locations: np.ndarray
    True Dirac locations.
    :return: estimated locations and positionning error.
    """
    mu, annihilating_filter = total_least_squares(fs_coeff, K)
    estimated_locations = roots_to_locations(annihilating_filter, period=period)
    estimated_locations, cost = match_to_ground_truth(locations, estimated_locations, period=period)
    return estimated_locations, cost
