# ############################################################################
# reproduce_execution_times.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# ############################################################################
"""
This script compares the execution times of CPGD and GenFRI for various oversampling parameters.
This is also the script for reproducing the results of [Section V.B, 1].
* Results of the simulations are saved in the folder `../results`, in the subfolder specified by `save_folder`.
* For re-running the simulations, set `run_simu` to True.
* For re-generating the plots of past simulations, set `run_simu` to False and specify in `save_folder` the name of the
subfolder in `../results` containing the results.

Note: Simulations may take a several minutes to execute on a laptop! To reduce computation
time, consider smaller oversampling parameters beta.

Reference:
[1] Simeoni, M., Besson, A., Hurley, P. & Vetterli, M. (2020). Cadzow Plug-and-Play Gradient Descent for Generalised FRI.
Under review.
"""

import pickle
import numpy as np
import os, datetime
import pyoneer.model.dirac_stream as mod
from benchmarking.plots.plot_routines import timing_plots
from pyoneer.operators.linear_operator import ToeplitzificationOperator, FRISampling
from pyoneer.algorithms.genfri import GenFRIAlgorithm
from pyoneer.algorithms.cpgd import CPGDAlgorithm

def algorithmic_contest(data_noisy: np.ndarray, algo_names: list,
                        settings_cpgd: dict, settings_genfri: dict):
    # Input and output local variables
    times = np.zeros(shape=(len(algo_names),))

    for i, algo_name in enumerate(algo_names):
        # CPGD
        if algo_name == 'CPGD':
            cpgd = CPGDAlgorithm(**settings_cpgd)
            _ = cpgd.reconstruct(data_noisy, verbose=False)
            total_time = cpgd.total_time
        # GenFRI
        else:
            genfri = GenFRIAlgorithm(**settings_genfri)
            _ = genfri.reconstruct(data_noisy, verbose=False)
            total_time = genfri.total_time
        times[i] = float(total_time)
    return times


if __name__ == '__main__':
    run_simu = True  # bool, Re-run simulations
    filename_general_settings = 'general_settings.pickle'  # str, name of pickle file storing settings
    filename_results = 'results.pickle'  # str, name of pickle file storing results
    save_folder = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")  # str, name of folder in which to save the results.
    # The results of the paper are saved in the folder  `../results/paper_execution_times`.

    # Check if results folder exists or create it:
    cwd = os.getcwd()
    results_dir = os.path.join(cwd, 'results', save_folder)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    if run_simu:
        # Parameters
        K = 9  # int, number of Diracs
        beta = np.array([1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300])  # np.ndarray, oversampling parameter
        M = beta * K  # np.ndarray, bandwidth parameter
        P = M  # np.ndarray, parameter P in [1].
        N = 2 * M + 1  # np.ndarray, sizes of seeked Fourier series coefficients.
        L = N  # np.ndarray, number of measurements
        period = 1  # float, period of Dirac stream
        PSNR = 20  # float, peak signal-to-noise ratios.
        seed = 3  # int, seed of random number generator for reproducibility.
        tol = 1e-4  # float, tolerance for stopping criterion.
        eig_tol = 1e-8  # float, tolerance for low-rank approximation if `backend_cadzow` is 'scipy.sparse'.
        nb_cadzow_iter = 10  # int, number of iterations in Cadzow denoising (typically smaller than 20).
        backend_cadzow = 'scipy.sparse'  # str,  backend for low-rank approximation.

        algo_markers = {'CPGD': 'D', 'GenFRI': 'X'}
        algo_names = list(algo_markers.keys())
        nb_algorithms_in_contest = len(algo_names)
        # Settings dictionaries
        settings_dirac = {'dirac_count': K, 'sigma_intensity': 0.5, 't_end': (1 - 0.01) * period,
                          't_start': 0.01 * period,
                          'mean_intensity': 0, 'relative_minimal_distance': 0.01,
                          'intensity_distribution': 'lognormal',
                          'seed': seed}
        settings_cpgd = {'nb_iter': 1, 'rank': K, 'nb_cadzow_iter': nb_cadzow_iter, 'denoise_verbose': False,
                         'nb_init': 1, 'tol': tol, 'eig_tol': eig_tol, 'tau_init_type': 'safest',
                         'random_state': seed, 'cadzow_backend': backend_cadzow, 'tau_weight': 0.5}
        settings_genfri = {'nb_iter': 1, 'nb_init': 1, 'tol': tol, 'random_state': seed, 'rcond': 1e-4}
        settings_experiment = {'K': K, 'beta': beta, 'M': M, 'P': P, 'N': N, 'L': L, 'period': period, 'PSNR': PSNR,
                               'seed': seed}
        ## Save settings

        with open(os.path.join(results_dir, filename_general_settings), 'wb') as file:
            pickle.dump(dict(genfri=settings_genfri, cpgd=settings_cpgd,
                             dirac=settings_dirac, experiment=settings_experiment, algo_names=algo_names,
                             algo_markers=algo_markers), file)

        # Generate signal innovations
        locations, intensities = mod.rnd_innovations(**settings_dirac)

        # Create the output variables
        store_times = np.zeros(shape=(len(beta), nb_algorithms_in_contest))
        fs_coeff_list = []
        data_noiseless_list = []
        sampling_locations_list = []

        # Set random number generator
        rng = np.random.RandomState(seed=1)

        # Run simulations for various values of beta:
        for n in range(N.size):
            print(f'**********Iteration {n}: N={N[n]}, L={L[n]} **********')
            # Generate noiseless Fourier coefficients
            frequencies = np.arange(-M[n], M[n] + 1)
            fs_coeff = mod.fourier_series_coefficients(M[n], locations, intensities, period=period)
            fs_coeff_list.append(fs_coeff)

            # Generate sampling locations
            sampling_locations, _ = mod.rnd_innovations(L[n], t_end=period, seed=1)
            sampling_locations_list.append(sampling_locations)
            # Generate the irregular sampling operator
            G = FRISampling(frequencies, sampling_locations, period)
            settings_cpgd['linear_op'] = G
            settings_genfri['linear_op'] = G
            data_noiseless = G(fs_coeff)
            data_noiseless_list.append(data_noiseless)

            # Noisy data
            noise_lvl = np.max(intensities) * np.exp(-PSNR / 10)
            std_noise = rng.standard_normal(size=(sampling_locations.size,))
            data_noisy = data_noiseless + noise_lvl * std_noise

            # Create Toeplitzification Operator
            Tp = ToeplitzificationOperator(P=P[n], M=M[n])
            settings_cpgd['toeplitz_op'] = Tp
            settings_genfri['toeplitz_op'] = Tp

            store_times[n] = algorithmic_contest(data_noisy, algo_names, settings_cpgd, settings_genfri)

        # Store the results to be able to regenerate the plots
        with open(os.path.join(results_dir, filename_results), 'wb') as file:
            results = {'store_times': store_times,
                       'data_noiseless_list': data_noiseless_list, 'sampling_locations_list': sampling_locations_list,
                       'dirac_locations': locations, 'dirac_intensities': intensities,
                       'fs_coeff_list': fs_coeff_list}
            pickle.dump(results, file)

    # Generate and save the plots
    timing_plots(os.path.join(results_dir, filename_general_settings), os.path.join(results_dir, filename_results),
                 results_dir, cmap_name='tab10', color_ind=(0, 6))
