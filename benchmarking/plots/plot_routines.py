# ############################################################################
# plot_routines.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# ############################################################################
"""
Plotting routines.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom_style.mplstyle'))
cmap = plt.get_cmap("tab10")


def plot_stream_of_diracs(dirac_locations: np.ndarray, dirac_intensities: np.ndarray, period: float = 1,
                          linefmt: str = 'dimgray', marker: str = 'o', edgecolors: str = 'dimgray', zorder: int = 1):
    """
    Plot a Dirac stream over its fundamental period.
    :param dirac_locations: np.ndarray[K,]
    :param dirac_intensities: np.ndarray[K,]
    :param period: float
    :param linefmt: str
    Line format supported by `matplotlib.pyplot.stem`.
    :param marker: str
    Marker supported by `matplotlib`.
    :param edgecolors: str
    Color supported by `matplotlib`.
    :param zorder: int
    Vertical order of the plot for superposition.
    :return: handle
    Handle of the scatter plot for inputting into legends.
    """
    plt.stem(dirac_locations, dirac_intensities, linefmt=linefmt, use_line_collection=True)
    scatter_hdl = plt.scatter(dirac_locations, dirac_intensities, marker=marker,
                              s=100, c=cmap(np.arange(dirac_intensities.size)), zorder=zorder, edgecolors=edgecolors)
    plt.xlim(0, period)
    return scatter_hdl


def loclocplot(K: int, recovered_locations: np.ndarray, dirac_locations: np.ndarray, marker: str = 'o', zorder=3,
               s: int = 60, linewidths: int = 1, alpha: float = 0.5, mad_positions=None, linestyle: str = '-'):
    """
    Plot the recovered vs actual Dirac locations.
    :param K: int
    :param recovered_locations: np.ndarray[K,]
    Recovered Dirac locations.
    :param dirac_locations: np.ndarray[K,]
    Actual Dirac locations.
    :param marker: str
    Marker supported by `matplotlib`.
    :param zorder: int
    Vertical order of the plot for superposition.
    :param s:
    Size of markers.
    :param linewidths: int
    :param alpha: float
    Controls transparency.
    :param mad_positions: {None,np.ndarray[K,]}
    If not None, print the mean absolute deviations of the recovered locations.
    :param linestyle: str
    Linestyle supported by `matplotlib`.
    """
    for j in range(K):
        color_marker_face = list(cmap(j))
        color_marker_face[-1] = alpha
        plt.scatter(recovered_locations[j], dirac_locations[j], marker=marker, s=s, zorder=zorder,
                    c=np.array(color_marker_face)[None, :], linewidths=linewidths, edgecolors=cmap(j))
        # if mad_positions is not None:
        #     plt.plot([recovered_locations[j] - mad_positions[j], recovered_locations[j] + mad_positions[j]],
        #              [dirac_locations[j], dirac_locations[j]], color=cmap(j), linewidth=linewidths + 1,
        #              linestyle=linestyle, zorder=zorder, alpha=alpha)
        # else:
        #     pass


def profiles_with_quantiles(x: np.ndarray, median: np.ndarray, percentile_bottom: np.ndarray,
                            percentile_top: np.ndarray,
                            cmap, color_ind: tuple, markers: list, algo_names: list, markersize: int = 6,
                            linewidth: int = 2, alpha: float = 0.3, xlabel: str = 'PSNR (dB)',
                            ylabel: str = 'Positioning Error (\% of $T$)'):
    """
    Performance plots (median + inter-percentile distance).
    :param x: np.ndarray[n,]
    Absciss.
    :param median: np.ndarray[n,k]
    Median of ordinates for each algorithm.
    :param percentile_bottom: np.ndarray[n,k]
    Lower percentile of ordinates for each algorithm.
    :param percentile_top: np.ndarray[n,k]
    Upper percentile of ordinates for each algorithm.
    :param cmap: cmap
    Colormap object.
    :param color_ind: tuple
    Colors from the colormap to be used.
    :param markers: str
    Marker supported by `matplotlib`.
    :param algo_names: list[str] of size k
    Names of the algorithms.
    :param markersize: int
    :param linewidth: float
    :param alpha: float
    :param xlabel: str
    :param ylabel: str
    """
    for k, algo_name in enumerate(algo_names):
        plt.plot(x, median[:, k], '-', fillstyle='full', markersize=markersize, linewidth=linewidth,
                 color=cmap(color_ind[k]), marker=markers[algo_name])
        plt.fill_between(x, percentile_bottom[:, k], percentile_top[:, k], linewidth=0, color=cmap(color_ind[k]),
                         alpha=alpha)
    plt.legend(algo_names, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.xlim((np.min(x), np.max(x)))
    plt.ylabel(ylabel, fontsize=16)


def simu_plots(settings: str, results: str, save_folder: str, percentile: int = 25, cmap_name: str = 'tab10',
               color_ind: tuple = (4, 0, 6)):
    """
    Routine for plotting the various diagnostic plots from [Section V,1].
    :param settings: str
    Path to the pickle file containing the settings of the simulation.
    :param results: str
    Path to the pickle file containing the results of the simulation.
    :param save_folder: str
    Path to the folder where plots should be saved.
    :param percentile: int
    Percentile to be used for inter-percentiles regions.
    :param cmap_name: str
    Name of colormap.
    :param color_ind: tuple
    Which colors from the colormap are used.

    Reference:
    [1] Simeoni, M., Besson, A., Hurley, P. & Vetterli, M. (2020). Cadzow Plug-and-Play Gradient Descent for Generalised FRI.
    Under review.
    """
    # Load simulation settings
    with open(settings, 'rb') as file:
        settings_dict = pickle.load(file)
    # settings_dirac = settings_dict['dirac']
    settings_experiment = settings_dict['experiment']
    algo_names = settings_dict['algo_names']
    algo_markers = settings_dict['algo_markers']
    PSNR = settings_experiment['PSNR']
    beta = settings_experiment['beta']
    K = settings_experiment['K']
    M = settings_experiment['M']
    N = settings_experiment['N']
    L = settings_experiment['L']
    period = settings_experiment['period']
    nb_exp = settings_experiment['nb_exp']

    # Load results
    with open(results, 'rb') as file:
        results_dict = pickle.load(file)
    store_positions = results_dict['store_positions']
    store_results = results_dict['store_results']
    store_iters = results_dict['store_iters']
    store_times = results_dict['store_times']
    sampling_locations = results_dict['sampling_locations']
    dirac_locations = results_dict['dirac_locations']
    dirac_intensities = results_dict['dirac_intensities']
    data_noiseless_list = results_dict['data_noiseless_list']
    low_pass_signal_list = results_dict['low_pass_signal_list']

    # Compute median and quatiles of the results
    median_results = np.median(store_results, axis=2)
    percentile_bottom_results = np.percentile(store_results, percentile, axis=2)
    percentile_top_results = np.percentile(store_results, 100 - percentile, axis=2)

    median_iters = np.median(store_iters, axis=2)
    percentile_bottom_iters = np.percentile(store_iters, percentile, axis=2)
    percentile_top_iters = np.percentile(store_iters, 100 - percentile, axis=2)

    median_times = np.median(store_times, axis=2)
    percentile_top_times = np.percentile(store_times, 100 - percentile, axis=2)
    percentile_bottom_times = np.percentile(store_times, percentile, axis=2)

    median_positions = np.median(store_positions, axis=2)

    # Decide on colors:
    cmap = plt.get_cmap(cmap_name)

    for b in range(beta.size):
        # Plot diracs and noiseless samples
        plt.figure()
        plt.plot(np.linspace(0, period, low_pass_signal_list[b].size), low_pass_signal_list[b], '-', color='silver')
        markerline, stemlines, baseline = plt.stem(sampling_locations, np.real(data_noiseless_list[b]) / N[b],
                                                   linefmt='None', markerfmt='D')
        markerline.set_markerfacecolor('silver')
        markerline.set_markeredgecolor('dimgray')
        plt.stem(dirac_locations, dirac_intensities, linefmt='dimgray')
        plt.scatter(dirac_locations, dirac_intensities, marker='o',
                    s=100, c=cmap(np.arange(dirac_intensities.size)), edgecolors='dimgray', zorder=4)
        plt.xlim((0, 1))
        # plt.legend(['Noiseless Samples', 'Dirac Stream'], fontsize=16)
        plt.savefig(os.path.join(save_folder, f'irregular_time_samples_K={K}_L={L}_M={M[b]}.pdf'), dpi=600,
                    transparent=False, bbox_inches='tight')

        if N[b] > L:
            algo_names = algo_names[:-1]
        # Reconstruction Error
        plt.figure()
        profiles_with_quantiles(PSNR, 100 * median_results[b], 100 * percentile_bottom_results[b],
                                100 * percentile_top_results[b], cmap,
                                color_ind, algo_markers, algo_names, markersize=6, linewidth=2, alpha=0.3,
                                xlabel='PSNR (dB)', ylabel='Positioning Error (\% of $T$)')
        plt.ylim((np.min(100 * percentile_bottom_results[np.isnan(percentile_bottom_results) == False]),
                  np.max(100 * percentile_top_results[np.isnan(percentile_top_results) == False])))
        plt.yscale('log')
        plt.savefig(
            os.path.join(save_folder, f'reconstructuion_error_irr_samp_K={K}_N={N[b]}_L={L}_rep={nb_exp}.pdf'),
            dpi=600, transparent=False, bbox_inches='tight')

        # Number of Iterations
        plt.figure()
        profiles_with_quantiles(PSNR, median_iters[b], percentile_bottom_iters[b], percentile_top_iters[b], cmap,
                                color_ind, algo_markers, algo_names, markersize=6, linewidth=2, alpha=0.3,
                                xlabel='PSNR (dB)', ylabel='Number of Iterations')
        plt.ylim((0.8 * np.min(percentile_bottom_iters[np.isnan(percentile_bottom_iters) == False]),
                  np.max(percentile_top_iters[np.isnan(percentile_top_iters) == False])))
        plt.yscale('log')
        plt.savefig(
            os.path.join(save_folder, f'nb_iters_irr_samp_K={K}_N={N[b]}_L={L}_rep={nb_exp}.pdf'),
            dpi=600, transparent=False, bbox_inches='tight')

        # Times
        plt.figure()
        profiles_with_quantiles(PSNR, median_times[b], percentile_bottom_times[b], percentile_top_times[b], cmap,
                                color_ind, algo_markers, algo_names, markersize=6, linewidth=2, alpha=0.3,
                                xlabel='PSNR (dB)', ylabel='Execution Time (s)')
        plt.ylim((0.8 * np.min(percentile_bottom_times[np.isnan(percentile_bottom_times) == False]),
                  np.max(percentile_top_times[np.isnan(percentile_top_times) == False])))
        plt.yscale('log')
        plt.savefig(
            os.path.join(save_folder, f'times_irr_samp_K={K}_N={N[b]}_L={L}_rep={nb_exp}.pdf'),
            dpi=600, transparent=False, bbox_inches='tight')

        # Create localization plots
        sub_psnr = [0, 3, -1]  # Selected PSNR

        mad_positions = np.median(np.abs(store_positions - median_positions[:, :, None, ...]), axis=2)
        for i in sub_psnr:
            plt.figure()
            # Mock artists for legend
            hdl1 = plt.scatter([-1], [0], marker=algo_markers['LS-Cadzow'], s=60, c='#333337')
            hdl2 = plt.scatter([-1], [0], marker=algo_markers['CPGD'], s=60, c='#333337')
            if N[b] <= L:
                hdl3 = plt.scatter([-1], [0], marker=algo_markers['GenFRI'], s=60, c='#333337')
            hdl4 = plt.plot([0, 1], [0, 1], '-', color='#333337', linewidth=2, zorder=2)

            for k, name in enumerate(algo_names):
                loclocplot(K, median_positions[b, i, k], dirac_locations, marker=algo_markers[name],
                           zorder=3, s=60, linewidths=1, alpha=0.5, mad_positions=mad_positions[b, i, k], linestyle='-')

            plt.axis('square')
            plt.xlim((0, period))
            plt.ylim((0, period))
            plt.xlabel('Recovered Dirac Locations', fontsize=14)
            plt.ylabel('True Dirac Locations', fontsize=14)
            if N[b] <= L:
                plt.legend([hdl1, hdl2, hdl3, hdl4], algo_names, fontsize=14)
            else:
                plt.legend([hdl1, hdl2, hdl4], algo_names, fontsize=14)
            plt.savefig(os.path.join(save_folder, f'loc_loc_plt_irr_samp_K={K}_L={L}_N={N[b]}_PSNR={int(PSNR[i])}.pdf'),
                        dpi=600, transparent=False)


def timing_plots(settings: str, results: str, save_folder: str, cmap_name: str = 'tab10', color_ind: tuple = (0, 6)):
    """
    Routine for plotting the timing plots from [Section VI,1].
    :param settings: str
    Path to the pickle file containing the settings of the simulation.
    :param results: str
    Path to the pickle file containing the results of the simulation.
    :param save_folder: str
    Path to the folder where plots should be saved.
    :param cmap_name: str
    Name of colormap.
    :param color_ind: tuple
    Which colors from the colormap are used.

    Reference:
    [1] Simeoni, M., Besson, A., Hurley, P. & Vetterli, M. (2020). Cadzow Plug-and-Play Gradient Descent for Generalised FRI.
    Under review.
    """
    # Load simulation settings
    with open(settings, 'rb') as file:
        settings_dict = pickle.load(file)
    # settings_dirac = settings_dict['dirac']
    settings_experiment = settings_dict['experiment']
    algo_names = settings_dict['algo_names']
    algo_markers = settings_dict['algo_markers']
    # PSNR = settings_experiment['PSNR']
    beta = settings_experiment['beta']
    K = settings_experiment['K']
    M = settings_experiment['M']
    N = settings_experiment['N']
    L = settings_experiment['L']
    period = settings_experiment['period']

    # Load results
    with open(results, 'rb') as file:
        results_dict = pickle.load(file)
    store_times = results_dict['store_times']
    sampling_locations_list = results_dict['sampling_locations_list']
    dirac_locations = results_dict['dirac_locations']
    dirac_intensities = results_dict['dirac_intensities']
    data_noiseless_list = results_dict['data_noiseless_list']

    # Decide on colors:
    cmap = plt.get_cmap(cmap_name)
    factor = [100, 50 * 15]
    ind_slopes = [9, 4]
    leg = []
    lines = ['--', '-.']
    # Times
    plt.figure()
    for k, algo_name in enumerate(algo_names):
        plt.plot(N, factor[k] * store_times[:, k], '-', fillstyle='full', markersize=6, linewidth=2,
                 color=cmap(color_ind[k]), marker=algo_markers[algo_name])
        log_last_times = np.log(factor[k] * store_times[-7:, k])
        log_last_N = np.log(N[-7:])
        design_matrix = np.stack([log_last_N, 0 * log_last_N + 1], axis=1)
        coeff = np.linalg.pinv(design_matrix) @ log_last_times
        print(design_matrix)
        print(coeff)
        print(f'Growth order of {algo_name}: {coeff[0]}. Intercept: {np.exp(coeff[1])}.')
        plt.plot(N, np.exp(coeff[1]) * N ** (coeff[0]), lines[k], linewidth=2,
                 color=cmap(ind_slopes[k]))
        leg.extend([algo_name, f'$\propto N^{{{np.round(coeff[0], 2)}}}$'])
    plt.legend(leg, fontsize=16)
    plt.xlabel('Size $N$', fontsize=16)
    plt.xlim((np.min(N), np.max(N)))
    plt.ylabel('Execution Time (s)', fontsize=16)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(
        os.path.join(save_folder, f'times_N_varying_K={K}.pdf'), dpi=600, transparent=False, bbox_inches='tight')
