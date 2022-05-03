# imports
from os.path import join
from os import listdir
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sympy import symbols, solve
import matplotlib.pyplot as plt

from utils import functions, fit, bin


"""
Blue: #0C5DA5
Green: #00B945
"""
sciblue = '#0C5DA5'
scigreen = '#00B945'

plt.style.use(['science', 'ieee', 'std-colors'])
figg, axx = plt.subplots()
size_x_inches, size_y_inches = figg.get_size_inches()
plt.close(figg)


def evaluate_particles_intensity_and_diameter(base_dir, mag_eff, working_distance_mm, microns_per_pixel,
                                              plot_particle_ids=None,
                                              plot_all_intensity=False,
                                              plot_sampling_frequency=None,
                                              path_test_coords_sampling_frequency=None,
                                              ):
    """

    :param path_test_coords_sampling_frequency:
    :param base_dir:
    :param mag_eff:
    :param working_distance_mm:
    :param microns_per_pixel:
    :param plot_particle_ids: 'all': plot all particles, [list]: plot particle IDs, 'None': skip
    :param plot_all_intensity:
    :param plot_sampling_frequency: input the number of particles assessed by IDPT to compare SPCT sampling.
    :return:
    """

    # setup filepaths
    path_calib_coords = join(base_dir, 'coords/calib-coords')
    path_figs = join(base_dir, 'figs')

    path_spct_stats = [join(path_calib_coords, f) for f in listdir(path_calib_coords) if f.startswith('calib_spct_stats_')][0]


    # read spct stats
    dfraw = pd.read_excel(path_spct_stats)
    dfraw = dfraw.dropna()

    # filter on number of frames
    dfcounts = dfraw.groupby('id').count().reset_index()
    max_counts = dfcounts.z_corr.max()
    remove_ids = dfcounts[dfcounts['z_corr'] < max_counts * 0.75].id.unique()
    df = dfraw[~dfraw.id.isin(remove_ids)]

    df = df.sort_values('id')
    pids = df.id.unique()

    # modifiers
    plots_per_fig = 10

    if plot_all_intensity:

        plt.style.use(['science', 'ieee', 'muted'])

        # create directory
        path_particle_intensities = join(path_figs, 'particle_intensity_plots')
        if not os.path.exists(path_particle_intensities):
            os.makedirs(path_particle_intensities)

        # structure data
        num_figs = int(np.ceil(len(pids) / plots_per_fig))

        for i in range(num_figs):

            fig, ax = plt.subplots(figsize=(size_x_inches * 1.125, size_y_inches * 1.125))

            for j in pids[i * plots_per_fig:(i + 1) * plots_per_fig]:

                dfpid = dfraw[dfraw['id'] == j]
                dfpid = dfpid.reset_index()
                zf = np.round(dfpid.iloc[dfpid.peak_int.idxmax()].z_corr, 2)
                ax.plot(dfpid.z_corr, dfpid.peak_int, '-o', ms=3, label='{}: {}'.format(j, zf))

            ax.legend(title=r'$p_{ID}: z_{f}$')
            plt.tight_layout()
            plt.savefig(path_particle_intensities + '/particles_{}_intensities.png'.format(i))
            plt.show()
            plt.close()

    if plot_particle_ids is not None:

        plt.style.use(['science', 'ieee', 'std-colors'])

        # create directory
        path_particle_diameter_and_intensities = join(path_figs, 'particle_diameter_and_intensity_plots')
        if not os.path.exists(path_particle_diameter_and_intensities):
            os.makedirs(path_particle_diameter_and_intensities)

        path_particle_diameter_x_and_y = join(path_figs, 'particle_diameter_x_and_y_plots')
        if not os.path.exists(path_particle_diameter_x_and_y):
            os.makedirs(path_particle_diameter_x_and_y)

        path_particle_sigma_x_and_y = join(path_figs, 'particle_sigma_x_and_y_plots')
        if not os.path.exists(path_particle_sigma_x_and_y):
            os.makedirs(path_particle_sigma_x_and_y)

        if plot_particle_ids == 'all':
            plot_particle_ids = pids

        s0 = working_distance_mm * 1e3

        for pid in plot_particle_ids:

            df = dfraw[dfraw['id'] == pid]

            z_corr = df.z_corr.to_numpy()
            peak_int_norm = df.peak_int.to_numpy() - df.peak_int.min()

            raw_amplitude, raw_c, raw_sigma = functions.get_amplitude_center_sigma(z_corr, peak_int_norm)

            guess_params = [raw_amplitude, raw_c, raw_sigma]

            popt, pcov = curve_fit(fit.gauss_1d_function, z_corr, peak_int_norm, p0=guess_params)
            fit_amplitude, fit_xc, fit_sigma = popt[0], popt[1], popt[2]

            # constrain the z range by Gaussian sigma
            z_range = fit_sigma * 5
            df = df[(df['z_corr'] > -z_range) & (df['z_corr'] < z_range)]

            # center on z_f = 0
            df['z_corr'] = df['z_corr'] - fit_xc

            # --- fit diameter function
            z_corr = df.z_corr.to_numpy()
            peak_int_norm = df.peak_int.to_numpy() - df.peak_int.min()
            gauss_diameter = df.gauss_diameter.to_numpy()
            gauss_dia_x = df.gauss_dia_x.to_numpy()
            gauss_dia_y = df.gauss_dia_y.to_numpy()

            def particle_diameter_function(z, c1, c2):
                return mag_eff * np.sqrt(c1 ** 2 * z ** 2 + c2 ** 2)

            guess_c1, guess_c2 = 0.15, 0.65

            poptxy, pcovxy = curve_fit(particle_diameter_function,
                                       z_corr,
                                       gauss_diameter,
                                       p0=[guess_c1, guess_c2],
                                       bounds=([0, 0], [1, 1])
                                       )

            poptx, pcovx = curve_fit(particle_diameter_function,
                                     z_corr,
                                     gauss_dia_x,
                                     p0=[guess_c1, guess_c2],
                                     bounds=([0, 0], [1, 1])
                                     )

            popty, pcovy = curve_fit(particle_diameter_function,
                                     z_corr,
                                     gauss_dia_y,
                                     p0=[guess_c1, guess_c2],
                                     bounds=([0, 0], [1, 1])
                                     )

            def gaussian_particle_intensity_distribution_2d(data, a):
                z = data[0]
                diameter = data[1]
                return a / (diameter ** 2 * (s0 + z) ** 2)

            z_corr = df.z_corr.to_numpy()
            peak_int_norm = df.peak_int.to_numpy() - df.peak_int.min()
            gauss_diameter = df.gauss_diameter.to_numpy()

            # fit 2D Gaussian distribution: function(z, r)
            data = [z_corr * 1e-6, gauss_diameter * 1e-6 * microns_per_pixel]
            poptj, pcovj = curve_fit(gaussian_particle_intensity_distribution_2d, data, peak_int_norm)

            # fit z
            z_fit_intensity = np.linspace(z_corr.min(), z_corr.max(), 250) * 1e-6
            resampled_diameter = particle_diameter_function(z_fit_intensity * 1e6, *poptxy)
            data_fit = [z_fit_intensity, resampled_diameter * 1e-6 * microns_per_pixel]

            # plot - diameter + intensity
            fig, ax = plt.subplots()

            ax.plot(z_corr, gauss_diameter * microns_per_pixel, 'o', ms=2, color=sciblue, fillstyle='none')
            ax.plot(z_fit_intensity * 1e6, resampled_diameter * microns_per_pixel, linewidth=0.5, color='midnightblue')
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_ylabel(r'$d_{e} \: (\mu m)$')

            axr = ax.twinx()
            axr.plot(z_corr, peak_int_norm, 's', ms=2, color='darkgray', fillstyle='none')
            axr.plot(z_fit_intensity * 1e6, gaussian_particle_intensity_distribution_2d(data_fit, *poptj),
                     linewidth=0.5, color='dimgray')
            axr.set_ylabel(r'$I_{max} \: (A.U.)$', color='gray')
            plt.tight_layout()
            plt.savefig(path_particle_diameter_and_intensities + '/pid{}_diameter_and_intensity.png'.format(pid))
            plt.close()

            # plot - diameter x + diameter y

            gauss_dia_x = df.gauss_dia_x.to_numpy()
            resampled_dia_x = particle_diameter_function(z_fit_intensity * 1e6, *poptx)

            gauss_dia_y = df.gauss_dia_y.to_numpy()
            resampled_dia_y = particle_diameter_function(z_fit_intensity * 1e6, *popty)

            # plot
            fig, ax = plt.subplots()

            ax.plot(z_corr, gauss_dia_x * microns_per_pixel,
                    'o', ms=2, color=sciblue, fillstyle='none', label=r'$a_{x}$')
            ax.plot(z_fit_intensity * 1e6, resampled_dia_x * microns_per_pixel, linewidth=0.5, color='tab:blue')

            ax.plot(z_corr, gauss_dia_y * microns_per_pixel,
                    's', ms=2, color=scigreen, fillstyle='none', label=r'$a_{y}$')
            ax.plot(z_fit_intensity * 1e6, resampled_dia_y * microns_per_pixel, linewidth=0.5, color='tab:green')

            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_ylabel(r'$a \: (\mu m)$')
            ax.legend(loc='lower left')
            plt.tight_layout()
            plt.savefig(path_particle_diameter_x_and_y + '/pid{}_diameter_x_and_y.png'.format(pid))
            plt.close()

            # plot - diameter x / diameter y

            fig, ax = plt.subplots()
            ax.plot(z_corr, df.gauss_dia_x_y.to_numpy(), 'o', ms=2)
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_ylabel(r'$a_{x}/a_{y}$')
            plt.tight_layout()
            plt.savefig(path_particle_diameter_x_and_y + '/pid{}_diameter_x_y.png'.format(pid))
            plt.close()

            # plot - sigma x + sigma y

            gauss_sigma_x = df.gauss_sigma_x.to_numpy()
            gauss_sigma_y = df.gauss_sigma_y.to_numpy()
            fig, ax = plt.subplots()
            ax.plot(z_corr, gauss_sigma_x, 'o', ms=2, color=sciblue, fillstyle='none', label=r'$\sigma_{x}$')
            ax.plot(z_corr, gauss_sigma_y, 's', ms=2, color=scigreen, fillstyle='none', label=r'$\sigma_{y}$')
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_ylabel(r'$a \: (pixels)$')
            ax.legend(loc='lower left')
            plt.tight_layout()
            plt.savefig(path_particle_sigma_x_and_y + '/pid{}_sigma_x_and_y.png'.format(pid))
            plt.close()

            # plot - sigma x + sigma y
            fig, ax = plt.subplots()
            ax.plot(z_corr, df.gauss_sigma_x_y.to_numpy(), '-o', ms=2)
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_ylabel(r'$a_{x}/a_{y}$')
            plt.tight_layout()
            plt.savefig(path_particle_sigma_x_and_y + '/pid{}_sigma_x_y.png'.format(pid))
            plt.close()

    if plot_sampling_frequency is not None:

        path_sampling_frequency = join(path_figs, 'compare_sampling_frequency')
        if not os.path.exists(path_sampling_frequency):
            os.makedirs(path_sampling_frequency)

        # --- SPCT Nyquist sampling

        # read test coords to get number of particles per frame (z_true)
        dft_spct = pd.read_excel(path_test_coords_sampling_frequency)
        dft_spct = dft_spct[dft_spct['cm'] > 0.5]
        spct_num_per_z_true = dft_spct.groupby('frame').count().z.to_numpy()

        # calculate mean particle spacing and Nyquist sampling (z)
        arr_spct_mean_particle_to_particle_spacing = np.sqrt((512 * microns_per_pixel) ** 2 / spct_num_per_z_true)
        arr_spct_nyquist_measured = arr_spct_mean_particle_to_particle_spacing * 2

        # bin spct_stats by z_corr to get corrected z values and gaussian diameter
        column_to_bin = 'z_corr'
        column_to_count = 'id'
        bins = len(dfraw.frame.unique())
        round_to_decimal = 4
        dfzm, dfzstd = bin.bin_generic(dfraw, column_to_bin, column_to_count, bins, round_to_decimal)

        # SPCT Nyquist minimum w/o overlap
        spct_nyquist_no_gaussian_overlap = dfzm.gauss_diameter.to_numpy() * 2
        spct_nyquist_no_contour_overlap = dfzm.diameter_contour.to_numpy() * 2

        # IDPT Nyquist sampling
        square_microns_per_particle = (512 * microns_per_pixel) ** 2 / plot_sampling_frequency
        mean_particle_to_particle_spacing = np.sqrt(square_microns_per_particle)
        idpt_nyquist = mean_particle_to_particle_spacing * 2 * np.ones_like(dfzm.z_corr)

        # plot
        ms = 2

        # measured IDPT vs. SPCT Nyquist sampling
        fig, ax = plt.subplots()

        ax.plot(dfzm.z_corr, idpt_nyquist, '-o', ms=ms, label=r'$IDPT$')
        ax.plot(dfzm.z_corr, arr_spct_nyquist_measured, '-o', ms=ms, label=r'$SPCT$')

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$f_{Nyquist} \: (\mu m)$')
        ax.set_ylim([0, arr_spct_nyquist_measured.max() * 1.05])
        ax.legend()
        plt.tight_layout()
        plt.savefig(path_sampling_frequency + '/compare_IDPT-SPCT_Nyq_sampling.png')
        plt.show()
        plt.close()

        # measured IDPT vs. SPCT Nyquist sampling + ideal Gaussian and contour
        fig, ax = plt.subplots()

        ax.plot(dfzm.z_corr, idpt_nyquist, '-o', ms=ms, label=r'$IDPT$')
        ax.plot(dfzm.z_corr, arr_spct_nyquist_measured, '-o', ms=ms, label=r'$SPCT$')
        ax.plot(dfzm.z_corr, spct_nyquist_no_gaussian_overlap, '-o', ms=ms, label=r'$SPCT_{N.O.}(Gaussian)$')
        ax.plot(dfzm.z_corr, spct_nyquist_no_contour_overlap, '-o', ms=ms, label=r'$SPCT_{N.O.}(contour)$')

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$f_{Nyquist} \: (\mu m)$')
        ax.set_ylim([0, arr_spct_nyquist_measured.max() * 1.05])
        ax.legend()
        plt.tight_layout()
        plt.savefig(path_sampling_frequency + '/compare_IDPT-SPCT_Nyq_sampling_gauss-contour-dia.png')
        plt.show()
        plt.close()

        # measured IDPT vs. SPCT Nyquist sampling + ideal contour
        fig, ax = plt.subplots()

        ax.plot(dfzm.z_corr, idpt_nyquist, '-o', ms=ms, label=r'$IDPT$')
        ax.plot(dfzm.z_corr, arr_spct_nyquist_measured, '-o', ms=ms, label=r'$SPCT$')
        ax.plot(dfzm.z_corr, spct_nyquist_no_contour_overlap, '-o', ms=ms, label=r'$SPCT_{N.O.}$')

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$f_{Nyquist} \: (\mu m)$')
        ax.set_ylim([0, arr_spct_nyquist_measured.max() * 1.05])
        ax.legend()
        plt.tight_layout()
        plt.savefig(path_sampling_frequency + '/compare_IDPT-SPCT_Nyq_sampling_contour-dia.png')
        plt.show()
        plt.close()

        # measured IDPT vs. SPCT Nyquist sampling + ideal Gaussian
        fig, ax = plt.subplots()

        ax.plot(dfzm.z_corr, idpt_nyquist, '-o', ms=ms, label=r'$IDPT$')
        ax.plot(dfzm.z_corr, arr_spct_nyquist_measured, '-o', ms=ms, label=r'$SPCT$')
        ax.plot(dfzm.z_corr, spct_nyquist_no_gaussian_overlap, '-o', ms=ms, label=r'$SPCT_{N.O.}$')

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$f_{Nyquist} \: (\mu m)$')
        ax.set_ylim([0, arr_spct_nyquist_measured.max() * 1.05])
        ax.legend()
        plt.tight_layout()
        plt.savefig(path_sampling_frequency + '/compare_IDPT-SPCT_Nyq_sampling_gauss-dia.png')
        plt.show()
        plt.close()



base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/FINAL-04.25.22_SPCT_1um-calib_5um-test'

mag_eff = 10
working_distance_mm = 7.8
microns_per_pixel = 1.6
plot_particle_ids = [6, 44, 60, 61]

evaluate_particles_intensity_and_diameter(base_dir, mag_eff, working_distance_mm, microns_per_pixel,
                                          plot_particle_ids=plot_particle_ids,
                                          plot_all_intensity=False,
                                          plot_sampling_frequency=None
                                          )

print("Analysis completed without errors")