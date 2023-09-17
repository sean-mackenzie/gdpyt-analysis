# 02.06.22 - local axial and radial displacements per membrane

# imports
import os
from os.path import join

import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from utils import functions, bin, modify
from utils.plotting import lighten_color

# ---

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'
scipurple = '#845B97'
sciblack = '#474747'
scigray = '#9e9e9e'
sci_color_list = [sciblue, scigreen, scired, sciorange, scipurple, sciblack, scigray]

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 0. Experimental Parameters

mag_eff = 5.0
numerical_aperture = 0.3
pixel_size = 16
depth_of_focus = functions.depth_of_field(mag_eff, numerical_aperture, 600e-9, 1.0, pixel_size=pixel_size * 1e-6) * 1e6
microns_per_pixel = 3.2
exposure_time = 40e-3
frame_rate = 24.444
time_per_frame = 1 / frame_rate
start_frame = 39
start_time = start_frame / frame_rate

# exclude outliers
pids_saturated = [12, 13, 18, 34, 39, 49, 66, 78]
excluded_pids = [39, 61]
bad_pids = [12, 13, 18, 34, 39, 49, 61, 66, 78]

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 1. FILES PATHS

base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/02.06.22_radial_displacement'
path_data = join(base_dir, 'data')
path_results = join(base_dir, 'results')

# ----------------------------------------------------------------------------------------------------------------------

# ---

# setup
tid = 1
read_raw_or_reconstruction = 'raw'
memb_id = 1
memb_str = 'lr'
exclude_pids = excluded_pids

# ----------------------------------------------------------------------------------------------------------------------
# 2. PROCESSING - IDPT COORDS
read_current_local_displacement_and_export_minimum_dataset = False
if read_current_local_displacement_and_export_minimum_dataset:

    def read_current_and_export_minimum(tid, read_raw_or_reconstruction, memb_id, exclude_pids):

        # modifiers
        if memb_id == 1:
            save_id = 'lr'
        elif memb_id == 2:
            save_id = 'ul'
        else:
            raise ValueError("Membrane ID currently only implemented for 1 and 2.")

        # read
        data_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/02.06.22_local_displacement'
        path_slope_corrected_coords = join(data_dir, 'data/slope-corrected', 'dz{}'.format(tid))
        dfd = pd.read_excel(path_slope_corrected_coords + '/df_{}_corr_displacement_lr-ul_nd.xlsx'.format(read_raw_or_reconstruction))

        # ---

        # processing

        # filter - membrane
        dfd = dfd[dfd['memb_id'].isin([memb_id])]

        # filter - time
        dfd = dfd[dfd['t'] > start_time]

        # filter - bad pids
        dfd = dfd[~dfd['id'].isin(exclude_pids)]

        # get only necessary columns
        dfd = dfd[
            ['memb_id', 'frame', 't', 'id', 'cm',
             'r', 'rg',  # initial r position, and tracked r-positions
             'drg', 'dz',  # r- and z-displacement of particles (relative zero deflection)
             'apparent_dr', 'drg_corr',  # apparent r-displacement and slope-corrected r-displacement (mid-plane)
             'nd_r', 'nd_rg',  # non-dimensional initial r and tracked r positions
             'd_dz',  # MODEL: dimensional z-displacement
             'memb_radius']
        ]

        # ---

        # append peak deflection 'w0' to dataframe

        # read
        data_dir2 = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/' \
                    'analyses/results-09.15.22_idpt-subpix'
        path_read2 = data_dir2 + '/results/dz{}'.format(tid)
        dfr = pd.read_excel(path_read2 + '/id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest.xlsx'.format(tid))
        dfr = dfr[['frame', 'rz_lr', 'rz_ul']]

        # append peak deflection to dataframe
        map_to_col = 'w0'
        map_on_col = 'frame'
        mapping_col = 'rz_{}'.format(save_id)

        dfd = modify.map_column_between_dataframes(map_to_df=dfd,
                                                   map_to_col=map_to_col,
                                                   map_on_col=map_on_col,
                                                   mapping_df=dfr,
                                                   mapping_col=mapping_col,
                                                   )

        dfd.to_excel(path_data + '/df_{}_radial_displacement_{}_nd.xlsx'.format(read_raw_or_reconstruction, save_id))

    # ---

    read_current_and_export_minimum(tid, read_raw_or_reconstruction, memb_id, exclude_pids)

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 2. PROCESSING - COMSOL COORDS
parse_comsol_txt_and_export = False
if parse_comsol_txt_and_export:
    # positive deflections
    fptxt = path_data + '/COMSOL/txt/1D_r-displacement_by_+P0.txt'
    df_comsol = pd.read_csv(fptxt, delimiter='          ', names=['r', 'dr'])

    fptxt = path_data + '/COMSOL/txt/1D_r-displacement_by_+P0_midplane.txt'
    dfm_comsol = pd.read_csv(fptxt, delimiter='          ', names=['r', 'dr'])

    plt_lbls = [100, 300, 600]
    plt_length = 501
    i = 0

    dfcs = []
    for lbl in plt_lbls:
        dfc = df_comsol.iloc[i:i+plt_length]
        dfc['P0'] = lbl
        dfc['dr_corr'] = dfm_comsol.iloc[i:i+plt_length].dr
        dfcs.append(dfc)
        i += plt_length

    df_cmsl_pos = pd.concat(dfcs)

    # ---

    # negative deflections
    fptxt = path_data + '/COMSOL/txt/1D_r-displacement_by_-P0.txt'
    df_comsol = pd.read_csv(fptxt, delimiter='          ', names=['r', 'dr'])

    fptxt = path_data + '/COMSOL/txt/1D_r-displacement_by_-P0_midplane.txt'
    dfm_comsol = pd.read_csv(fptxt, delimiter='          ', names=['r', 'dr'])

    plt_lbls = [-600, -300, -100]
    plt_length = 501
    i = 0

    dfcs = []
    for lbl in plt_lbls:
        dfc = df_comsol.iloc[i:i+plt_length]
        dfc['P0'] = lbl
        dfc['dr_corr'] = dfm_comsol.iloc[i:i+plt_length].dr
        dfcs.append(dfc)
        i += plt_length

    df_cmsl_neg = pd.concat(dfcs)

    df_cmsl = pd.concat([df_cmsl_pos, df_cmsl_neg])
    df_cmsl = df_cmsl.sort_values(['P0', 'r'])

    # ---

    # z-displacement
    fptxt = path_data + '/COMSOL/txt/1D_z-displacement_by_r.txt'
    df_comsol = pd.read_csv(fptxt, delimiter='          ', names=['r', 'dz'])

    plt_lbls = [-600, -300, -100, 0, 100, 300, 600]
    plt_length = 501
    i = 0

    dfcs = []
    for lbl in plt_lbls:
        dfc = df_comsol.iloc[i:i+plt_length]
        dfc['P0'] = lbl
        if lbl != 0:
            dfcs.append(dfc)
        i += plt_length

    df_cmsl_z = pd.concat(dfcs)
    df_cmsl['dz'] = df_cmsl_z.dz
    df_cmsl.to_excel(path_data + '/COMSOL/1D_dz-dr-dr_corr_by_r_P0.xlsx'.format(lbl), index=False)

    # ---

# ---


# ----------------------------------------------------------------------------------------------------------------------
# 2. PLOTTING
plot_radial_displacement = True
if plot_radial_displacement:

    r_col = 'nd_r'
    dr_col = 'drg'
    dr_corr_col = dr_col + '_corr'

    df = pd.read_excel(path_data + '/df_{}_radial_displacement_{}_nd.xlsx'.format(read_raw_or_reconstruction, memb_str))
    df_cmsl = pd.read_excel(path_data + '/COMSOL/1D_dz-dr-dr_corr_by_r_P0.xlsx')

    # ---

    # fitting functions
    def sine_decay(x, A, f, b):
        """
        :param A: amplitude
        :param f: frequency - the number of cycles per second; bounds = (0, 1)
        :param b: decay rate
        """
        return A * np.sin(2 * np.pi * f * x) * np.exp(-x * b) * (1 - x ** 12)

    # ---

    # plotting

    # plot positive deflection
    plot_positive = True
    if plot_positive:

        # peak positive deflection groups:
        """
        Peak positive deflection groups: [[75, 80], [130, 134, 145], [185, 188, 196]]
            peak 1: [frame, deflection]: [75, 80], [66, 98]
            peak 2: [frame, deflection]: [130, 134, 145], [66, 99, 133]
            peak 3: [frame, deflection]: [185, 188, 196], [66, 99, 130]
            
        Peak negative deflection groups: [[44, 48, 50], [103, 107, 115], [161, 163, 170]]
            peak 1: [frame, deflection]: [44, 48, 50], [-60 -102, -112]
            peak 2: [frame, deflection]: [103, 107, 115], [-73, -100, -119]
            peak 3: [frame, deflection]: [161, 163, 170], [-74, -97, -126]
            
        """

        froi_groups = [
            [75, 80], [130, 134, 145], [185, 188, 196],  # positive deflections
            [44, 48, 50], [103, 107, 115], [161, 163, 170],  # negative deflections
                       ]
        comsol_groups = [
            [100, 300], [100, 300, 600], [100, 300, 600],  # positive deflections
            [-100, -300, -600], [-100, -300, -600], [-100, -300, -600],  # negative deflections
        ]

        plot_comsol = True
        fit_curve = False
        n_points = 150
        r_fit = np.linspace(0, 1, n_points)  # 1.025 = 20 microns
        guess = [1, 0.5, 1]
        bounds = ([1.3, 0.8, 0], [2, 5, 5])

        for plot_corrected_dr in [False, True]:
            if plot_corrected_dr:
                dr_col = 'drg_corr'
                ylbl = r'$\Delta r_{mid-plane} \: (\mu m)$'
            else:
                ylbl = r'$\Delta r \: (\mu m)$'

            for plot_normalized_r in [True, False]:
                if plot_normalized_r:
                    xlbl = r'$r / a$'
                    xticks = [0, 0.5, 1]
                    memb_radius = 1
                    cmsl_radius = 1
                else:
                    xlbl = r'$r \: (\mu m)$'
                    xticks = [0, 200, 400, 600, 800]
                    memb_radius = df.iloc[0]['memb_radius'] * microns_per_pixel
                    cmsl_radius = 800

                for frois, cmsl_P0s in zip(froi_groups, comsol_groups):

                    fig, ax = plt.subplots()
                    markers = ['o', 's', 'd', '*', 'p', 'D', '+', 'x']
                    clrs = cm.plasma(np.linspace(0.9, 0.1, len(frois)))

                    for froi, cmsl_P0, mrk, clr in zip(frois, cmsl_P0s, markers, clrs):
                        fit_ = False
                        y_fit = np.zeros_like(r_fit)

                        # get dataframe of this frame
                        dfr = df[df['frame'] == froi].sort_values(r_col).reset_index()

                        # membrane data
                        w0 = int(np.round(dfr.iloc[0]['w0'], 0))

                        # particle data
                        dfr = dfr.dropna(subset=[r_col, dr_col])
                        x = dfr[r_col].to_numpy()
                        y = dfr[dr_col].to_numpy()

                        if len(y) < 10:
                            continue

                        # fit curve
                        if fit_curve:
                            try:
                                popt, pcov = curve_fit(sine_decay, x, y, p0=guess, xtol=1.49012e-07, maxfev=1000)
                                y_fit = sine_decay(r_fit, *popt)
                                fit_ = True
                            except RuntimeError:
                                pass

                        # plot
                        p1, = ax.plot(x * memb_radius, y * microns_per_pixel, marker=mrk, ms=2, color=lighten_color(clr, 1.1),
                                      linestyle='', label=w0)
                        if fit_:
                            ax.plot(r_fit * memb_radius, y_fit * microns_per_pixel, linewidth=0.75, color=clr, alpha=0.5)

                        if plot_comsol:
                            dfc = df_cmsl[df_cmsl['P0'] == cmsl_P0]
                            dfc = dfc[dfc['r'] < 850]
                            dfc['nd_r'] = dfc['r'] / 800
                            dfc['drg'] = dfc['dr']
                            dfc['drg_corr'] = dfc['dr_corr']

                            r_cmsl = dfc[r_col].to_numpy()
                            dr_cmsl = dfc[dr_col].to_numpy()

                            ax.plot(r_cmsl * cmsl_radius, dr_cmsl, linewidth=0.75, linestyle=':', color=clr, alpha=0.5)

                    ax.set_ylabel(ylbl)
                    ax.set_xlabel(xlbl)
                    ax.set_xticks(xticks)
                    ax.legend(title=r'$w_{o} \: (\mu m)$',
                              markerscale=1.5, labelspacing=0.35, handletextpad=0.15, borderaxespad=0.25)

                    plt.tight_layout()
                    plt.savefig(path_results +
                                '/dr_peak-fr{}_w0={}_norm={}_corr={}.png'.format(froi,
                                                                                 w0,
                                                                                 plot_normalized_r,
                                                                                 plot_corrected_dr))
                    plt.show()
                    plt.close()

                # ---

            # ---

        # ---

    # ---

# ---