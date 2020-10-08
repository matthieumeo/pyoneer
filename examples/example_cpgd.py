import numpy as np
import pyoneer.model.dirac_stream as mod
from benchmarking.plots.plot_routines import loclocplot
from pyoneer.operators.linear_operator import ToeplitzificationOperator, FRISampling
from scipy.linalg import lstsq
from pyoneer.utils.fri import coeffs_to_matched_diracs
from pyoneer.algorithms.cpgd import CPGDAlgorithm
import matplotlib.pyplot as plt

# Parameters
K = 9  # int, number of Diracs
beta = 3 # np.array([1, 2, 3, 4, 5])  # np.ndarray, oversampling parameter
M = beta * K  # np.ndarray, bandwidth parameter
P = M  # np.ndarray, parameter P in [1].
N = 2 * M + 1  # np.ndarray, sizes of seeked Fourier series coefficients.
L = 2 * 4 * K + 1  # float, number of measurements
period = 1  # float, period of Dirac stream
PSNR = -10  # np.ndarray, peak signal-to-noise ratios.
seed = 4  # int, seed of random number generator for reproducibility.
tol = 1e-9   # float, tolerance for stopping criterion.
eig_tol = 1e-8  # float, tolerance for low-rank approximation if `backend_cadzow` is 'scipy.sparse'.
nb_cadzow_iter = 10  # int, number of iterations in Cadzow denoising (typically smaller than 20).
backend_cadzow = 'scipy'  # str,  backend for low-rank approximation.

# Settings dictionaries used as inputs to algorithms and routines
settings_dirac = {'dirac_count': K, 'sigma_intensity': 0.5, 't_end': (1 - 0.01) * period,
                  't_start': 0.01 * period,
                  'mean_intensity': 0, 'relative_minimal_distance': 0.01,
                  'intensity_distribution': 'lognormal',
                  'seed': seed}
settings_cpgd = {'nb_iter': 5000, 'rank': K, 'nb_cadzow_iter': nb_cadzow_iter, 'denoise_verbose': False,
                 'nb_init': 1, 'tol': tol, 'eig_tol': eig_tol, 'tau_init_type': 'safest',
                 'random_state': seed, 'cadzow_backend': backend_cadzow, 'tau_weight': 0.5}

# Generate signal innovations
locations, intensities = mod.rnd_innovations(**settings_dirac)

# Generate sampling locations and standardised noise.
sampling_locations, _ = mod.rnd_innovations(L, t_end=period,
                                            relative_minimal_distance=0.005, seed=1)
rng = np.random.RandomState(seed=1)
std_noise = rng.standard_normal(size=(sampling_locations.size,))

print(f'********** N={N}, L={L} **********')
# Generate noiseless Fourier coefficients
frequencies = np.arange(-M, M + 1)
fs_coeff = mod.fourier_series_coefficients(M, locations, intensities, period=period)

# Generate the irregular sampling operator
G = FRISampling(frequencies, sampling_locations, period)
print(f'Cond. num. of G of size {G.shape}: {np.linalg.cond(G.mat):.2f}')
settings_cpgd['linear_op'] = G
data_noiseless = G(fs_coeff)

# Create Toeplitzification Operator
Tp = ToeplitzificationOperator(P=P, M=M)
settings_cpgd['toeplitz_op'] = Tp

noise_lvl = np.max(intensities) * np.exp(-PSNR / 10)
data_noisy = data_noiseless + noise_lvl * std_noise

# CPGD
cpgd = CPGDAlgorithm(**settings_cpgd)
fs_coeff_recovered = cpgd.reconstruct(data_noisy, verbose=True)
total_iterations = cpgd.total_iterations
total_time = cpgd.total_time
estimated_locations, cost = coeffs_to_matched_diracs(fs_coeff_recovered, K, period, locations)

loclocplot(K=K, recovered_locations=estimated_locations, dirac_locations=locations)
plt.show()
