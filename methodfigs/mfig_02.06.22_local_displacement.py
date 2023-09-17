# 02.06.22 - local axial and radial displacements per membrane

# imports
import os
from os.path import join

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, Akima1DInterpolator

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

import analyze
from utils import boundary, functions, io, bin
from utils.functions import fSphericalUniformLoad, fNonDimensionalNonlinearSphericalUniformLoad
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
E_silpuran = 500e3
poisson = 0.5
t_membrane = 20e-6
t_membrane_norm = 20

# pressure application
start_frame = 39
start_time = start_frame / frame_rate

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 1. Processing Parameters
padding_during_idpt_test_calib = 15  # 10
image_length = 512
img_xc = 256
img_yc = 256

""" --- MEMBRANE SPECIFIC PARAMETERS --- """

# mask lower right membrane
xc_lr, yc_lr, r_edge_lr = 423, 502, 252
circle_coords_lr = [xc_lr, yc_lr, r_edge_lr]

# mask upper left membrane
xc_ul, yc_ul, r_edge_ul = 167, 35, 157
circle_coords_ul = [xc_ul, yc_ul, r_edge_ul]

# mask left membrane
xc_ll, yc_ll, r_edge_ll = 12, 289, 78
circle_coords_ll = [xc_ll, yc_ll, r_edge_ll]

# mask middle
xc_mm, yc_mm, r_edge_mm = 177, 261, 31
circle_coords_mm = [xc_mm, yc_mm, r_edge_mm]

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 2. Setup Files and Variables

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/02.06.22_local_displacement'

path_test_coords = join(base_dir, 'data/test-coords')
path_reconstruction_coords = join(base_dir, 'data/reconstruction-coords')
path_boundary = join(base_dir, 'data/boundary')
path_mask_boundary_npy = join(path_boundary, 'mask/mask_boundary.npy')

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 2.5 IMPORTANT PHYSICAL POSITIONS

# axial positions
z_f_from_calib = 140
z_offset_lr = 5
z_offset_ul = 2
z_inital_focal_plane_bias_errors = np.max([z_offset_lr, z_offset_ul]) + 5

lr_w0_max = 133
ul_w0_max = 64

"""
dz1:
z_i_mean_lr = -3.5
z_i_mean_ul = -0
z_i_mean_ll = -8.5
z_i_mean_mm = -10
z_mean_boundary = -3.5
z allowance = 2.5

dz22: is actually a dz1:
z_i lr = -4.5
z i ul = -1
z allowance = 2.5

dz23: is actually a dz1:
z_i lr = -4
z i ul = -2
z allowance = 1.5

dz24: is actually a dz1:
z_i lr = -2.5
z i ul = +1
z allowance = 2.5
"""

# exclude outliers
"""
Saturated pids for ttemp <= 11: [12, 13, 18, 34, 39, 49, 66, 78]
Saturated pids for ttemp == 13: [11, 12, 17, 33, 38, 48, 65, 77]
"""
pids_saturated = [12, 13, 18, 34, 39, 49, 66, 78]
exclude_pids = [39, 61]

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 3. Analyze and Plot
analyze_local_displacement_and_plot = False

if analyze_local_displacement_and_plot:

    # ---

    # define: initialize plate model
    def initialize_plate_model(df_test, df_results, membrane_radius, p_col, k_col,
                               nonlinear_only=False, exclude_outside_membrane_radius=False):
        """ dft_, fND_ = initialize_plate_model(df_test, df_results, membrane_radius, p_col, k_col) """
        a_membrane = membrane_radius * microns_per_pixel
        fND = fNonDimensionalNonlinearSphericalUniformLoad(r=a_membrane * 1e-6,
                                                           h=t_membrane,
                                                           youngs_modulus=E_silpuran,
                                                           poisson=poisson)

        # 2. calculate non-dimensional pressure and pre-tension
        nd_P, nd_k = fND.non_dimensionalize_p_k(d_p0=df_results[p_col].to_numpy(),
                                                d_n0=df_results[k_col].to_numpy()
                                                )
        df_results['nd_p'] = nd_P
        df_results['nd_k'] = nd_k

        # 4. Append nd_P, nd_k columns to 'dft'

        # 4.1 - columns to be mapped
        df_test['d_p'] = df_test['frame']
        df_test['d_k'] = df_test['frame']
        df_test['nd_p'] = df_test['frame']
        df_test['nd_k'] = df_test['frame']

        # 4.2 - create mapping dict
        mapper_dict = df_results[['frame', p_col, k_col, 'nd_p', 'nd_k']].set_index('frame').to_dict()

        # 4.3 - map nd_P, nd_k to 'dft' by 'frame'
        df_test = df_test.replace({'d_p': mapper_dict[p_col]})
        df_test = df_test.replace({'d_k': mapper_dict[k_col]})
        df_test = df_test.replace({'nd_p': mapper_dict['nd_p']})
        df_test = df_test.replace({'nd_k': mapper_dict['nd_k']})

        # ---

        # 5. COMPUTE NON-LINEAR NON-DIMENSIONAL

        # 5.1 - Calculate nd_z, nd_slope, nd_curvature using nd_P, nd_k.

        nd_P = df_test['nd_p'].to_numpy()
        nd_k = df_test['nd_k'].to_numpy()

        nd_r = df_test['r'].to_numpy() * microns_per_pixel / a_membrane
        nd_z = fND.nd_nonlinear_clamped_plate_p_k(nd_r, nd_P, nd_k)

        nd_theta = fND.nd_nonlinear_theta(nd_r, nd_P, nd_k)
        # nd_curve = fND.nd_nonlinear_curvature_lva(nd_r, nd_P, nd_k)
        nd_curve = np.where(nd_k > 0.0,
                            fND.nd_nonlinear_curvature_lva(nd_r, nd_P, nd_k),
                            fND.nd_nonlinear_theta_plate(nd_r, nd_P),
                            )

        df_test['nd_r'] = nd_r
        df_test['nd_rg'] = df_test['rg'].to_numpy() * microns_per_pixel / a_membrane
        df_test['nd_dr'] = df_test['drg'] * microns_per_pixel / t_membrane_norm
        df_test['nd_dz'] = nd_z
        df_test['nd_dz_corr'] = (df_test['z_corr'] + df_test['z_offset']) / t_membrane_norm
        df_test['d_dz_corr'] = df_test['z_corr'] + df_test['z_offset']
        df_test['d_dz'] = nd_z * t_membrane_norm
        df_test['nd_theta'] = nd_theta
        df_test['nd_curve'] = nd_curve

        # 7. Calculate error: (z_corr - d_z); and squared error for rmse
        df_test['dz_error'] = df_test['d_dz_corr'] - df_test['d_dz']
        df_test['z_rmse'] = df_test['dz_error'] ** 2

        # -

        if nonlinear_only:
            df_test = df_test[df_test['nd_k'] > 0.0001]

        if exclude_outside_membrane_radius:
            df_test = df_test[df_test['r'] < membrane_radius]

        # -

        return df_test, fND


    # -

    def func_analyze_local_displacement_and_plot(tid, norm_r_bins, read_raw_or_reconstruction,
                                                 remove_initial_fpb_errors=False,
                                                 remove_saturated_pids=False):
        fn = 'test_coords_stats_dz{}'.format(tid)
        fnr = 'df_reconstruction_dz{}'.format(tid)
        path_results = join(base_dir, 'results')
        filetype = '.xlsx'

        # ---
        path_results_tid_coords = join(base_dir, 'results', 'dz{}'.format(tid))
        path_results_tid = join(base_dir, 'results',
                                'dz{}/{}/bins-{}'.format(tid, read_raw_or_reconstruction, len(norm_r_bins)))
        path_figs_tid = join(path_results_tid, 'figs')

        if not os.path.exists(path_results_tid):
            os.makedirs(path_results_tid)
        if not os.path.exists(path_figs_tid):
            os.makedirs(path_figs_tid)

        # ---

        # read file
        if read_raw_or_reconstruction == 'raw':
            df = pd.read_excel(join(path_test_coords, fn + filetype))

            keep_columns = ['frame', 't', 'id', 'z', 'z_corr', 'cm',
                            'x', 'y', 'xg', 'yg', 'gauss_xc', 'gauss_yc', 'pdf_xc', 'pdf_yc',
                            'pdf_A', 'pdf_sigma_x', 'pdf_sigma_y', 'pdf_rho']

        elif read_raw_or_reconstruction == 'reconstruction':
            df = pd.read_excel(join(path_reconstruction_coords, fnr + filetype))

            keep_columns = ['frame', 't', 'memb_id', 'id', 'z', 'z_corr', 'cm',
                            'x', 'y', 'xg', 'yg', 'gauss_xc', 'gauss_yc', 'pdf_xc', 'pdf_yc',
                            'pdf_A', 'pdf_sigma_x', 'pdf_sigma_y', 'pdf_rho',
                            'mmb_xc', 'mmb_yc',
                            'rr0', 'drr']
        else:
            raise ValueError("Test coords not understood.")

        # -

        # read from: analyses/results... --> data processing
        data_dir2 = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/' \
                    'analyses/results-09.15.22_idpt-subpix'
        path_read2 = data_dir2 + '/results/dz{}'.format(tid)

        # read dataframe
        dfr = pd.read_excel(path_read2 + '/id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest.xlsx'.format(tid))
        dfr = dfr[['frame', 'time',
                   'rz_lr', 'fit_lr_pressure', 'fit_lr_pretension',
                   'rz_ul', 'fit_ul_pressure', 'fit_ul_pretension',
                   ]]

        # ---

        # exclude pids
        if remove_saturated_pids:
            df = df[~df['id'].isin(exclude_pids)]

        # ---

        # X-Y columns
        xy_cols_list = ['x', 'y', 'xg', 'yg']  # , 'gauss_xc', 'gauss_yc', 'pdf_xc', 'pdf_yc']

        analyze_cols = ['frame', 'id', 'z', 'cm',
                        'x', 'y', 'xg', 'yg']  # , 'gauss_xc', 'gauss_yc', 'pdf_xc', 'pdf_yc',
        # 'pdf_A', 'pdf_sigma_x', 'pdf_sigma_y', 'pdf_rho']

        xy_cols = [['x', 'y'], ['xg', 'yg']]  # , ['gauss_xc', 'gauss_yc'], ['pdf_xc', 'pdf_yc']]
        r_cols = ['r', 'rg']  # , 'gauss_r', 'pdf_r']

        # ----------------------------------------------------------------------------------------------------------------------
        # 3. Adjust for padding and identify particles on membranes

        # 3.0 - Remove baseline frame
        # df = df[df['frame'] >= 1]  # NOTE: baseline frame removal no longer necessary.

        # 3.1 - Adjust for padding
        if read_raw_or_reconstruction == 'raw':
            """ NOTE: reconstruction coords are already corrected for padding. """
            if padding_during_idpt_test_calib != 0:
                for xy_col in xy_cols_list:
                    df[xy_col] = df[xy_col] - padding_during_idpt_test_calib

        # 3.3 - Add time column
        if 't' not in df.columns:
            df['t'] = df['frame'] / frame_rate

        # 3.4 - Add 'z_corr' column (corrected to focal plane)
        if 'z_corr' not in df.columns:
            """ NOTE: z_f_from_calib = 140 was also used to generate reconstruction coords. Don't change this param. """
            df['z_corr'] = df['z'] - z_f_from_calib

        # 3.2 - Add 'r' columns
        """for r_col, xy_col in zip(r_cols, xy_cols):
            df[r_col] = functions.calculate_radius_at_xy(df[xy_col[0]], df[xy_col[1]], xc=img_xc, yc=img_yc)"""

        # 3.4 - Get only necessary columns
        df = df[keep_columns]

        # ---

        # ----------------------------------------------------------------------------------------------------------------------
        # 3.5. Filter particles with impossible positions

        # The below filter is necessary for Gaussian fitting x-y localization but not template matchin.
        # for xy_col in xy_cols_list:
        # df = df[(df[xy_col] > 0) & (df[xy_col] < image_length)]

        # 3.5b - filter particles with inital focal plane bias errors
        if remove_initial_fpb_errors:
            df.loc[(df['z_corr'] > z_inital_focal_plane_bias_errors) & (df['t'] < start_time), 'z_corr'] = np.nan
            print("Setting 'z_corr' to NaN for all particles @"
                  " 'z_corr' > {} for 't' < {}".format(z_inital_focal_plane_bias_errors, start_time))
            df = df.dropna(subset=['z_corr'])

        # ---

        # ----------------------------------------------------------------------------------------------------------------------
        # 4. Get particles on features

        if not os.path.exists(join(path_boundary, 'boundary_pids.txt')):
            # 4.0a - load mask
            mask_boundary = np.load(path_mask_boundary_npy)

            # 4.0b - get boundary particles
            boundary_pids, interior_pids = boundary.get_boundary_particles(mask_boundary,
                                                                           df,
                                                                           return_interior_particles=True,
                                                                           flip_xy_coords=True,
                                                                           flip_xy_coords_minus_mask_size=False,
                                                                           )
            # 4.0c - package and export to avoid computing later
            io.write_list_to_txt_file(list_values=boundary_pids, filename='boundary_pids.txt', directory=path_boundary)
            io.write_list_to_txt_file(list_values=interior_pids, filename='interior_pids.txt', directory=path_boundary)

        # 4.1 - get dataframe of each feature
        if os.path.exists(join(path_results + 'df_pids_per_membrane.xlsx')):
            export_pids_per_membrane = False
        else:
            export_pids_per_membrane = False

        dflr, dful, dfll, dfmm, dfbd, lr_pids, ul_pids, ll_pids, mm_pids, boundary_pids = \
            boundary.get_pids_on_features(df, path_results, path_boundary, export_pids_per_membrane)

        # ---

        # ----------------------------------------------------------------------------------------------------------------------
        # 5. Plot particles prior to start time
        compare_or_inspect_initial = False
        if compare_or_inspect_initial:

            """
            In order to plot 'r' features, you must activate 3.2 and deactivate 3.4 in the code above (and copied below for ref)
            
            # 3.2 - Add 'r' columns
            for r_col, xy_col in zip(r_cols, xy_cols):
                df[r_col] = functions.calculate_radius_at_xy(df[xy_col[0]], df[xy_col[1]], xc=img_xc, yc=img_yc)
        
            # 3.4 - Get only necessary columns
            df = df[keep_columns]
            """

            # modifiers
            compare_features_initial_r_std = False
            compare_features_initial_z_r_std = False
            compare_features_initial_z_xy_std = False
            compare_features_initial_z = False
            inspect_features_initial_r = False
            inspect_features_initial_xy = False
            inspect_features_initial_3d = False

            # setup
            dfs = [dfbd, dflr, dful]
            lbls = ['Boundary', 'L.R.', 'U.L.']
            idx = np.arange(len(dfs))

            # compare features before start time
            if compare_features_initial_r_std:
                for r_col, xy_col in zip(r_cols, xy_cols):

                    fig, ax = plt.subplots(nrows=3, figsize=(size_x_inches * 1.25, size_y_inches * 1.2))

                    for dfst, lbl, i in zip(dfs, lbls, idx):
                        dfst_mean = dfst[dfst['t'] < start_time].groupby('id').mean()
                        dfst_std = dfst[dfst['t'] < start_time].groupby('id').std()

                        ax[i].plot(dfst_mean.index, dfst_std[r_col] * 2, 'o',
                                   ms=1,
                                   label=lbl)
                        ax[i].set_xlabel(r'$p_{ID}$')
                        ax[i].set_ylabel(r_col)
                        ax[i].legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$2\sigma_{r}$')

                    plt.tight_layout()
                    plt.show()
                    plt.close()

            # ---

            if compare_features_initial_z_r_std:
                for r_col, xy_col in zip(r_cols, xy_cols):

                    fig, ax = plt.subplots(nrows=3, figsize=(size_x_inches * 1.25, size_y_inches * 1.2))

                    for dfst, lbl, i in zip(dfs, lbls, idx):
                        dfst_mean = dfst[dfst['t'] < start_time].groupby('id').mean()
                        dfst_std = dfst[dfst['t'] < start_time].groupby('id').std()

                        ax[i].errorbar(dfst_mean[r_col], dfst_mean['z_corr'],
                                       yerr=dfst_std['z_corr'], xerr=dfst_std[r_col],
                                       fmt='o', markersize=2, elinewidth=1, capsize=2,
                                       label=lbl)
                        ax[i].set_xlabel(r_col)
                        ax[i].set_ylabel('z_corr')
                        ax[i].legend(loc='upper left', bbox_to_anchor=(1, 1))

                    plt.tight_layout()
                    plt.show()
                    plt.close()

            # ---

            if compare_features_initial_z_xy_std:
                for xy_col in xy_cols_list:

                    fig, ax = plt.subplots(nrows=3, figsize=(size_x_inches * 1.2, size_y_inches * 1.1))

                    for dfst, lbl, i in zip(dfs, lbls, idx):
                        dfst_mean = dfst[dfst['t'] < start_time].groupby('id').mean()
                        dfst_std = dfst[dfst['t'] < start_time].groupby('id').std()

                        ax[i].errorbar(dfst_mean[xy_col], dfst_mean['z_corr'],
                                       yerr=dfst_std['z_corr'], xerr=dfst_std[xy_col],
                                       fmt='o', markersize=2, elinewidth=1, capsize=2,
                                       label=lbl)
                        ax[i].set_xlabel(xy_col)
                        ax[i].set_ylabel('z_corr')
                        ax[i].legend(loc='upper left', bbox_to_anchor=(1, 1))

                    plt.tight_layout()
                    plt.show()
                    plt.close()

            # ---

            if compare_features_initial_z:
                for xy_col in xy_cols:

                    fig, ax = plt.subplots(nrows=3)

                    for dfst, lbl, i in zip(dfs, lbls, idx):
                        ax[i].scatter(dfst[dfst['t'] < start_time][xy_col[0]], dfst[dfst['t'] < start_time]['z_corr'],
                                      c=dfst[dfst['t'] < start_time]['id'],
                                      s=5,
                                      label=lbl)
                        ax[i].set_ylabel(xy_col[0])
                        ax[i].legend(loc='upper left', bbox_to_anchor=(1, 1))

                    plt.tight_layout()
                    plt.show()
                    plt.close()

            # ---

            # inspect each feature before start time
            if inspect_features_initial_r:

                for dfst, lbl in zip(dfs, lbls):
                    for r_col in r_cols:
                        fig, ax = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches))
                        ax.scatter(dfst[dfst['t'] < start_time][r_col], dfst[dfst['t'] < start_time]['z_corr'],
                                   c=dfst[dfst['t'] < start_time]['id'],
                                   s=5,
                                   label=lbl)
                        ax.set_xlabel(r_col)
                        ax.legend(loc='lower right')
                        ax.set_ylabel('z_corr')

                        plt.tight_layout()
                        plt.show()
                        plt.close()

            # ---

            # inspect each feature before start time
            if inspect_features_initial_xy:

                for dfst, lbl in zip(dfs, lbls):
                    for xy_col in xy_cols:
                        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,
                                                       figsize=(size_x_inches * 1.5, size_y_inches))
                        ax1.scatter(dfst[dfst['t'] < start_time][xy_col[0]], dfst[dfst['t'] < start_time]['z_corr'],
                                    c=dfst[dfst['t'] < start_time]['id'],
                                    s=5,
                                    label=lbl)
                        ax1.set_xlabel(xy_col[0])
                        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
                        ax1.set_ylabel('z_corr')

                        ax2.scatter(dfst[dfst['t'] < start_time][xy_col[1]], dfst[dfst['t'] < start_time]['z_corr'],
                                    c=dfst[dfst['t'] < start_time]['id'],
                                    s=5,
                                    label=lbl)
                        ax2.set_xlabel(xy_col[1])

                        plt.tight_layout()
                        plt.show()
                        plt.close()

            # ---

            # inspect each feature before start time
            if inspect_features_initial_3d:
                elev, azim = 5, -40

                for dfst, lbl in zip(dfs, lbls):
                    for xy_col in [['xg', 'yg']]:  # xy_cols:
                        fig = plt.figure(figsize=(6, 6))
                        ax = fig.add_subplot(projection='3d')

                        ax.scatter(dfst[dfst['t'] < start_time][xy_col[0]],
                                   dfst[dfst['t'] < start_time][xy_col[1]],
                                   dfst[dfst['t'] < start_time]['z_corr'],
                                   c=dfst[dfst['t'] < start_time]['id'],
                                   s=5,
                                   label=lbl)
                        ax.set_xlabel(xy_col[0])
                        ax.set_ylabel(xy_col[1])
                        ax.set_zlabel('z_corr')
                        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                        ax.view_init(elev, azim)

                        plt.tight_layout()
                        plt.show()
                        plt.close()

            # ---

        # ---

        # ----------------------------------------------------------------------------------------------------------------------
        # 6. Calculate per-frame radial displacement

        # --- --- Stated earlier so no need to restate
        # xy_cols = [['x', 'y'], ['xg', 'yg'], ['gauss_xc', 'gauss_yc'], ['pdf_xc', 'pdf_yc']]
        # r_cols = ['r', 'rg', 'gauss_r', 'pdf_r']
        start_time_col = 't'
        min_num_measurements = 3
        memb_id_col = 'memb_id'

        # ---

        if not os.path.exists(
                path_results_tid_coords + '/df_{}_local_displacement_lr-ul_nd.xlsx'.format(
                    read_raw_or_reconstruction)):
            dfd_membs = []
            for memb_id, df_memb, disc_coords, z_offset in zip([1, 2],
                                                               [dflr, dful],
                                                               [circle_coords_lr, circle_coords_ul],
                                                               [z_offset_lr, z_offset_ul],
                                                               ):
                dfd_memb = analyze.disc_calculate_local_displacement(df_memb, xy_cols, r_cols, disc_coords,
                                                                     start_time, start_time_col,
                                                                     min_num_measurements,
                                                                     memb_id, memb_id_col,
                                                                     z_offset,
                                                                     )
                dfd_membs.append(dfd_memb)

            dfd = pd.concat(dfd_membs)

            # ---

            # add non-dimensional

            # LR membrane
            dft_lr = dfd[dfd.id.isin(lr_pids)]
            dft_lr, fND_lr = initialize_plate_model(df_test=dft_lr,
                                                    df_results=dfr,
                                                    membrane_radius=r_edge_lr,
                                                    p_col='fit_lr_pressure',
                                                    k_col='fit_lr_pretension',
                                                    nonlinear_only=False,
                                                    exclude_outside_membrane_radius=True,
                                                    )

            # UL membrane
            dft_ul = dfd[dfd.id.isin(ul_pids)]
            dft_ul, fND_ul = initialize_plate_model(df_test=dft_ul,
                                                    df_results=dfr,
                                                    membrane_radius=r_edge_ul,
                                                    p_col='fit_ul_pressure',
                                                    k_col='fit_ul_pretension',
                                                    nonlinear_only=False,
                                                    exclude_outside_membrane_radius=True,
                                                    )

            dfd = pd.concat([dft_lr, dft_ul])

            # ---

            # export
            dfd.to_excel(
                path_results_tid_coords + '/df_{}_local_displacement_lr-ul_nd.xlsx'.format(
                    read_raw_or_reconstruction),
                index=False)

            # radial displacement columns: ['drr', 'dr', 'drg', 'dgauss_r', 'dpdf_r']
            #   > Note, 'drr' is the original radial displacement calculated during model fitting.

        else:
            dfd = pd.read_excel(
                path_results_tid_coords + '/df_{}_local_displacement_lr-ul_nd.xlsx'.format(read_raw_or_reconstruction))

        # ---

        # ----------------------------------------------------------------------------------------------------------------------
        # 7. (OPTIONAL) - PLOT LOCAL DISPLACEMENT
        plot_local_displacement = False
        if plot_local_displacement:

            dz_col = 'z_corr'
            dr_cols = ['drg']
            r_disp_col = 'drg'
            lgnd_cols = 2

            columns_to_bin = ['frame', 'r']
            column_to_count = 'id'
            norm_z_r = np.array([t_membrane, t_membrane]) * 1e6
            norm_cols_str = '_norm_h'
            memb_pids_list = None
            memb_id_col = 'memb_id'
            export_results = False

            # ---

            # Lower Right Membrane
            memb_radius = r_edge_lr
            memb_id = 1

            # ---

            df = dfd
            df = df[df[memb_id_col] == memb_id]
            df = df[df['t'] > start_time]

            # ---
            # ---

            # evaluate each particle trajectory and filter on residuals of fit
            apply_pid_filtering = False
            if apply_pid_filtering:
                x_col = 't'
                x0 = 1.6
                y1_col = dz_col
                y2_col = r_disp_col
                id_col = 'id'

                fit_function = 'empirical'  # 'Akima', 'interp1d', 'poly', 'sine'
                poly_deg = 16
                y1_residuals_col = 'pf_residuals_z'
                y2_residuals_col = 'pf_residuals_r'
                y1_residuals_limit = 15

                # evaluate a single "indicator" particle to get fitting function
                indicator_pid = 55
                df_indicator = df[df[id_col] == indicator_pid].sort_values(x_col).reset_index()
                arr_x = df_indicator[x_col].to_numpy()
                arr_y1 = df_indicator[y1_col].to_numpy()
                fit_func = Akima1DInterpolator(arr_x, arr_y1)

                def fit_amplitude_on_empirical_function(xdata, amplitude):
                    return amplitude * fit_func(xdata)

                def fit_sine_filter_residuals(dfpid, fitting_function, pid=None, iteration=None, save_figs=False,
                                              show_figs=False):
                    """ fit_func, df_good, df_bad = fit_sine_filter_residuals(dfpid)

                    :param fitting_function: 'poly', 'sine', 'interp1d', 'Akima', 'empirical'
                    """
                    # get data
                    arr_x = dfpid[x_col].to_numpy()
                    arr_y1 = dfpid[y1_col].to_numpy()
                    arr_y2 = dfpid[y2_col].to_numpy() * microns_per_pixel

                    # resample x
                    fit_x = np.linspace(arr_x.min(), arr_x.max(), 250)

                    # data range
                    arr_y1_range = arr_y1.max() - arr_y1.min()

                    # estimate function
                    est_A = arr_y1_range / 2 * -1
                    est_w = 2.65
                    est_p = 1.8
                    est_c = -5

                    # estimate range
                    dA = np.abs(est_A * 0.05)
                    dw = np.abs(est_w * 0.2)
                    dp = np.abs(est_p * 0.2)
                    dc = np.abs(est_c) * 2.5

                    # fit
                    if fitting_function == 'sine':
                        # fit function
                        guess = [est_A, est_w, est_p, est_c]
                        bounds = ([est_A - dA, est_w - dw, est_p - dp, est_c - dc],
                                  [est_A + dA, est_w + dw, est_p + dp, est_c + dc])
                        dict_res = functions.fit_sin(arr_x, arr_y1, guess, bounds)

                        # results
                        A = dict_res['amp']
                        w = dict_res['omega']
                        p = dict_res['phase']
                        c = dict_res['offset']
                        period = dict_res['period']
                        fit_func = dict_res['fitfunc']

                        # label
                        lbl = r'$(A, w, \phi, c)=$' + '({}, {}, {}, {})'.format(np.round(A, 1), np.round(w, 3),
                                                                                np.round(p, 3), np.round(c, 3))

                    elif fitting_function == 'poly':
                        pcoeff, residuals, rank, singular_values, rcond = np.polyfit(arr_x, arr_y1, deg=poly_deg,
                                                                                     full=True)
                        fit_func = np.poly1d(pcoeff)
                        lbl = 'deg. {}'.format(poly_deg)
                    elif fitting_function == 'interp1d':
                        fit_func = interp1d(arr_x, arr_y1)
                    elif fitting_function == 'Akima':
                        fit_func = Akima1DInterpolator(arr_x, arr_y1)
                        lbl = 'Akima'
                    elif fitting_function == 'empirical':
                        popt, pcov = curve_fit(fit_amplitude_on_empirical_function, arr_x, arr_y1, p0=[1],
                                               bounds=([0], [10]))
                        fit_func = lambda t: fit_amplitude_on_empirical_function(t, *popt)
                        lbl = 'A={}'.format(popt[0])

                    # evaluate residuals
                    y_model = fit_func(arr_x)
                    y_residuals = y_model - arr_y1
                    y_precision = np.mean(np.std(y_residuals))
                    y_rmse = np.sqrt(np.mean(y_residuals ** 2))

                    # add residuals column
                    dfpid[y1_residuals_col] = y_residuals

                    # resample
                    fit_y1 = fit_func(fit_x)

                    # good / bad residuals
                    df_good = dfpid[dfpid[y1_residuals_col].abs() < y1_residuals_limit]
                    df_bad = dfpid[dfpid[y1_residuals_col].abs() >= y1_residuals_limit]

                    if save_figs or show_figs:
                        fig, ax = plt.subplots()
                        ax.plot(fit_x, fit_y1, color='black', linewidth=0.5, label=lbl)
                        ax.scatter(df_good[x_col], df_good[y1_col], color='green', label='good')
                        ax.scatter(df_bad[x_col], df_bad[y1_col], color='red', label='bad')
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y1_col)
                        ax.legend(title=r'$p_{ID}=$' + str(pid))
                        ax.set_title(r'$\sigma_{z} = $' + ' {} '.format(np.round(y_rmse, 2)) + r'$\mu m$')
                        plt.tight_layout()
                        if save_figs:
                            plt.savefig(
                                path_figs_tid + '/pid{}_iteration{}_fit-{}.png'.format(pid, iteration,
                                                                                       fitting_function))
                        if show_figs:
                            plt.show()
                        plt.close()

                    return fit_func, df_good, df_bad

                df_pids = []
                bad_pids = []

                for pid in df[id_col].unique():
                    df_pid = df[df[id_col] == pid].sort_values(x_col).reset_index()

                    """arr_y1 = df_pid[y1_col].to_numpy()
                    arr_y1_range = arr_y1.max() - arr_y1.min()
                    if arr_y1_range < 50:
                        continue"""

                    # ---

                    # iterative filter
                    ffit_, df_pid_good, df_pid_bad = fit_sine_filter_residuals(df_pid, fit_function, pid, iteration=1,
                                                                               save_figs=True, show_figs=False)

                    # if >50% of points are bad, skip
                    if len(df_pid_good) < len(df_pid_bad):
                        bad_pids.append(pid)
                        continue

                    ffit, df_g, df_b = fit_sine_filter_residuals(df_pid_good, fit_function, pid, iteration=2,
                                                                 save_figs=True, show_figs=False)

                    # ---

                    # evaluate residuals using iteratively filtered function
                    arr_x = df_pid[x_col].to_numpy()
                    arr_y1 = df_pid[y1_col].to_numpy()

                    # y_model == function
                    y_model = ffit(arr_x)
                    y_residuals = y_model - arr_y1
                    y_precision = np.mean(np.std(y_residuals))
                    y_rmse = np.sqrt(np.mean(y_residuals ** 2))

                    # --- Confidence Bands via Time-Shifting
                    apply_time_shifting = False
                    if apply_time_shifting:
                        dt = 0.15

                        # y_model_lower == function - dt
                        y_model_lower = ffit(arr_x - dt)
                        y_residuals_lower = y_model_lower - arr_y1

                        # y_model_upper == function + dt
                        y_model_upper = ffit(arr_x + dt)
                        y_residuals_upper = y_model_upper - arr_y1

                        # add residuals column
                        df_pid[y1_residuals_col] = y_residuals
                        df_pid[y1_residuals_col + '_lower'] = y_residuals_lower
                        df_pid[y1_residuals_col + '_upper'] = y_residuals_upper

                        # good / bad residuals
                        df_good = df_pid[
                            (df_pid[y1_residuals_col].abs() < y1_residuals_limit) |
                            (df_pid[y1_residuals_col + '_lower'].abs() < y1_residuals_limit) |
                            (df_pid[y1_residuals_col + '_upper'].abs() < y1_residuals_limit)
                            ]
                        df_bad = df_pid[~df_pid['frame'].isin(df_good['frame'].unique())]
                    else:
                        df_pid[y1_residuals_col] = y_residuals
                        df_good = df_pid[df_pid[y1_residuals_col].abs() < y1_residuals_limit]
                        df_bad = df_pid[df_pid[y1_residuals_col].abs() >= y1_residuals_limit]

                    # store
                    df_pids.append(df_good)

                    # ---

                    # resample
                    fit_x = np.linspace(arr_x.min(), arr_x.max(), 250)
                    fit_y1 = ffit(fit_x)

                    # plot only large z-range
                    plot_pid_fit = False
                    if plot_pid_fit:
                        fig, ax = plt.subplots()
                        ax.plot(fit_x, fit_y1, color='black', linewidth=0.5, )
                        # label=r'$(A, w, \phi, c)=$' + '({}, {}, {}, {})'.format(np.round(A, 1), np.round(w, 3), np.round(p, 3), np.round(c, 3)))
                        ax.scatter(df_good[x_col], df_good[y1_col], color='green', label='good')
                        ax.scatter(df_bad[x_col], df_bad[y1_col], color='red', label='bad')
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y1_col)
                        ax.legend(title=r'$p_{ID}=$' + str(pid))
                        ax.set_title(r'$\sigma_{z} = $' + ' {} '.format(np.round(y_rmse, 2)) + r'$\mu m$')
                        plt.tight_layout()
                        plt.show()

                    # ---

                print("{} p ID's were 'bad'.".format(len(bad_pids)))
                df = pd.concat(df_pids)

            # ---
            # ---

            # 2d binning
            bin_frames = df[columns_to_bin[0]].unique()
            bin_r = np.round(np.array(norm_r_bins) * memb_radius, 0)
            bins = [bin_frames, bin_r]
            round_to_decimals = [0, 1]
            min_num_bin = 1
            return_groupby = True
            dfm, dfstd = bin.bin_generic_2d(df,
                                            columns_to_bin,
                                            column_to_count,
                                            bins,
                                            round_to_decimals,
                                            min_num_bin,
                                            return_groupby
                                            )

            # setup
            export_binned_df = True
            plot_each_bin = False
            plot_all_bins = True
            plot_all_pids = True
            plot_figs_separately = True
            plot_figs_separately_norm = True
            include_legend = False
            save_figs = True
            show_figs = False
            clr_map = cm.plasma(np.linspace(0.95, 0.15, len(bin_r)))

            # ---

            # export
            if export_binned_df:
                dfm['z_offset'] = z_offset_lr
                dfm['norm_r_by_microns'] = memb_radius * microns_per_pixel
                dfm['norm_dz_dr_by'] = t_membrane_norm
                dfm['r_bin_microns'] = dfm['bin_ll'] * microns_per_pixel
                dfm['r_bin_norm'] = dfm['bin_ll'] / memb_radius
                dfm['dz'] = dfm[dz_col] + z_offset_lr
                dfm['dz_norm'] = dfm['dz'] / t_membrane_norm
                dfm['dr_microns'] = dfm[r_disp_col] * microns_per_pixel
                dfm['dr_norm_microns'] = dfm['dr_microns'] / t_membrane_norm
                dfm['dr_norm_pixels'] = dfm[r_disp_col] / t_membrane_norm

                dfm.to_excel(path_figs_tid + '/dfm_2b-bin-frame-{}_disc-lr_num-bins={}_{}.xlsx'.format(
                    r_disp_col,
                    len(norm_r_bins),
                    read_raw_or_reconstruction),
                             )

            # ---

            if plot_each_bin:
                for br, br_lbl, clr in zip(bin_r, norm_r_bins, clr_map):
                    dfbr = dfm[dfm['bin_ll'] == br]
                    if len(dfbr) == 0:
                        continue

                    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                                   figsize=(size_x_inches * 1.5, size_y_inches * 1.25),
                                                   gridspec_kw={'height_ratios': [1, 1]})
                    # plot variables
                    arr_x = dfbr.t.to_numpy()
                    arr_z = dfbr[dz_col].to_numpy()
                    arr_r = dfbr[r_disp_col].to_numpy() * microns_per_pixel
                    ax1.plot(arr_x, arr_z, 'o', ms=1, color=clr, label=br_lbl)
                    ax2.plot(arr_x, arr_r, 'o', ms=1, lw=0.75, color=clr)

                    ax1.set_ylabel(r'$\Delta z^{\delta} \: (\mu m)$')
                    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.05), ncol=lgnd_cols, markerscale=2,
                               title=r'$r/a$')
                    ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
                    ax2.set_xlabel(r'$t \: (s)$')

                    plt.tight_layout()
                    if save_figs:
                        plt.savefig(path_figs_tid + '/2b-bin-frame-{}_disc-lr_bin={}_plasma.png'.format(r_disp_col, br))
                    if show_figs:
                        plt.show()
                    plt.close()

            # ---

            if plot_all_bins:
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                               figsize=(size_x_inches * 1.5, size_y_inches * 1.25),
                                               gridspec_kw={'height_ratios': [1, 1]})
                for br, br_lbl, clr in zip(bin_r, norm_r_bins, clr_map):
                    dfbr = dfm[dfm['bin_ll'] == br]
                    if len(dfbr) == 0:
                        continue

                    # plot variables
                    arr_x = dfbr.t.to_numpy()
                    arr_z = dfbr[dz_col].to_numpy()
                    arr_r = dfbr[r_disp_col].to_numpy() * microns_per_pixel

                    ax1.plot(arr_x, arr_z, 'o', ms=1, color=clr, label=br_lbl)
                    ax2.plot(arr_x, arr_r, 'o', ms=1, lw=0.75, color=clr)

                ax1.set_ylabel(r'$\Delta z^{\delta} \: (\mu m)$')
                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.05), ncol=lgnd_cols, markerscale=2,
                           title=r'$r/a$')
                ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
                ax2.set_xlabel(r'$t \: (s)$')

                plt.tight_layout()
                if save_figs:
                    plt.savefig(
                        path_figs_tid + '/2b-bin-frame-{}_disc-lr_num-bins={}_plasma_per-pid-filtering.png'.format(
                            r_disp_col,
                            len(norm_r_bins)),
                    )
                if show_figs:
                    plt.show()
                plt.close()

            # ---

            if plot_all_pids:
                df = df.sort_values('r')
                clr_map_pids = cm.plasma(np.linspace(0.95, 0.15, len(df.id.unique())))

                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                               figsize=(size_x_inches * 1.5, size_y_inches * 1.25),
                                               gridspec_kw={'height_ratios': [1, 1]})
                for pid, clr in zip(df.id.unique(), clr_map_pids):
                    dfbr = df[df['id'] == pid]
                    # plot variables
                    arr_x = dfbr.t.to_numpy()
                    arr_z = dfbr[dz_col].to_numpy()
                    arr_r = dfbr[r_disp_col].to_numpy() * microns_per_pixel

                    ax1.plot(arr_x, arr_z, 'o', ms=1, color=clr)
                    ax2.plot(arr_x, arr_r, 'o', ms=1, lw=0.75, color=clr)

                ax1.set_ylabel(r'$\Delta z^{\delta} \: (\mu m)$')
                ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
                ax2.set_xlabel(r'$t \: (s)$')

                plt.tight_layout()
                if save_figs:
                    plt.savefig(
                        path_figs_tid + '/2b-bin-frame-{}_disc-lr_plot-all-pids_plasma_per-pid-filtering.png'.format(
                            r_disp_col))
                if show_figs:
                    plt.show()
                plt.close()

            # ---

            # ---

            if plot_figs_separately:

                # plot z
                fig, ax1 = plt.subplots(figsize=(size_x_inches * 1., size_y_inches * 1.35 / 2))

                for br, br_lbl, clr in zip(bin_r, norm_r_bins, clr_map):
                    dfbr = dfm[dfm['bin_ll'] == br]
                    if len(dfbr) == 0:
                        ax1.plot(2, 500, 'o', ms=1, color=clr, label=br_lbl)
                        continue

                    # plot variables
                    arr_x = dfbr.t.to_numpy()
                    arr_z = dfbr[dz_col].to_numpy()

                    ax1.plot(arr_x, arr_z + z_offset_lr, 'o', ms=1, color=clr, label=br_lbl)

                if include_legend:
                    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.05), ncol=1, markerscale=2, title=r'$r/a$')
                ax1.set_ylabel(r'$\Delta z^{\delta} \: (\mu m)$')
                ax1.set_ylim([-140 + z_offset_lr, 140 + z_offset_lr])
                ax1.set_xlabel(r'$t \: (s)$')
                ax1.set_xticks([2, 4, 6, 8])

                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_figs_tid + '/2b-bin-frame-{}_disc-lr_num-bins={}_{}_only-z_units.png'.format(
                        r_disp_col,
                        len(norm_r_bins),
                        read_raw_or_reconstruction),

                                )
                if show_figs:
                    plt.show()
                plt.close()

                # ---

                # plot r
                fig, ax2 = plt.subplots(figsize=(size_x_inches * 1., size_y_inches * 1.35 / 2))

                for br, br_lbl, clr in zip(bin_r, norm_r_bins, clr_map):
                    dfbr = dfm[dfm['bin_ll'] == br]
                    if len(dfbr) == 0:
                        ax2.plot(2, 500, 'o', ms=1, color=clr, label=br_lbl)
                        continue

                    # plot variables
                    arr_x = dfbr.t.to_numpy()
                    arr_r = dfbr[r_disp_col].to_numpy() * microns_per_pixel

                    ax2.plot(arr_x, arr_r, 'o', ms=1, lw=0.75, color=clr, label=br_lbl)

                if include_legend:
                    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1.05), ncol=1, markerscale=2, title=r'$r/a$')
                ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
                ax2.set_ylim([-4, 8])
                ax2.set_xlabel(r'$t \: (s)$')
                ax2.set_xticks([2, 4, 6, 8])

                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_figs_tid + '/2b-bin-frame-{}_disc-lr_num-bins={}_{}_only-r_units.png'.format(
                        r_disp_col,
                        len(norm_r_bins),
                        read_raw_or_reconstruction),

                                )
                if show_figs:
                    plt.show()
                plt.close()

            # ---

            # ---

            if plot_figs_separately_norm:

                # plot z
                fig, ax1 = plt.subplots(figsize=(size_x_inches * 1., size_y_inches * 1.35 / 2))

                for br, br_lbl, clr in zip(bin_r, norm_r_bins, clr_map):
                    dfbr = dfm[dfm['bin_ll'] == br]
                    if len(dfbr) == 0:
                        ax1.plot(2, 500, 'o', ms=1, color=clr, label=br_lbl)
                        continue

                    # plot variables
                    arr_x = dfbr.t.to_numpy()
                    arr_z = dfbr[dz_col].to_numpy()

                    ax1.plot(arr_x, (arr_z + z_offset_lr) / t_membrane_norm, 'o', ms=1, color=clr, label=br_lbl)

                if include_legend:
                    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.05), ncol=1, markerscale=2, title=r'$r/a$')
                ax1.set_ylabel(r'$\Delta z^{\delta}/t_{m}$')
                ax1.set_ylim([(-140 + z_offset_lr) / t_membrane_norm, (140 + z_offset_lr) / t_membrane_norm])
                ax1.set_xlabel(r'$t \: (s)$')
                ax1.set_xticks([2, 4, 6, 8])

                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_figs_tid + '/2b-bin-frame-{}_disc-lr_num-bins={}_{}_only-z_norm.png'.format(
                        r_disp_col,
                        len(norm_r_bins),
                        read_raw_or_reconstruction),

                                )
                if show_figs:
                    plt.show()
                plt.close()

                # ---

                # plot r
                fig, ax2 = plt.subplots(figsize=(size_x_inches * 1., size_y_inches * 1.35 / 2))

                for br, br_lbl, clr in zip(bin_r, norm_r_bins, clr_map):
                    dfbr = dfm[dfm['bin_ll'] == br]
                    if len(dfbr) == 0:
                        ax2.plot(2, 500, 'o', ms=1, color=clr, label=br_lbl)
                        continue

                    # plot variables
                    arr_x = dfbr.t.to_numpy()
                    arr_r = dfbr[r_disp_col].to_numpy() * microns_per_pixel

                    ax2.plot(arr_x, arr_r / t_membrane_norm, 'o', ms=1, lw=0.75, color=clr, label=br_lbl)

                if include_legend:
                    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1.05), ncol=1, markerscale=2, title=r'$r/a$')
                ax2.set_ylabel(r'$\Delta r^{\delta}/t_{m}$')
                ax2.set_ylim([-4 / t_membrane_norm, 8 / t_membrane_norm])
                ax2.set_xlabel(r'$t \: (s)$')
                ax2.set_xticks([2, 4, 6, 8])

                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_figs_tid + '/2b-bin-frame-{}_disc-lr_num-bins={}_{}_only-r_norm.png'.format(
                        r_disp_col,
                        len(norm_r_bins),
                        read_raw_or_reconstruction),

                                )
                if show_figs:
                    plt.show()
                plt.close()

            # ---

            # ---
            # ---

            # Upper Left Membrane
            memb_radius = r_edge_ul
            memb_id = 2
            """dfm_ul = analyze.disc_bin_plot_local_displacement(dfd,
                                                              columns_to_bin, column_to_count,
                                                              dz_col, z_offset,
                                                              dr_cols,
                                                              norm_r_bins, norm_z_r, norm_cols_str,
                                                              memb_radius, memb_pids_list, memb_id, memb_id_col,
                                                              export_results, path_results,
                                                              show_plots=True,
                                                              save_plots=False,
                                                              microns_per_pixel=microns_per_pixel,
                                                              units_pixels=False,
                                                              )"""

            # ---

            df = dfd
            df = df[df[memb_id_col] == memb_id]
            df = df[df['t'] > start_time]

            # 2d binning
            bin_frames = df[columns_to_bin[0]].unique()
            bin_r = np.round(np.array(norm_r_bins) * memb_radius, 0)
            bins = [bin_frames, bin_r]
            round_to_decimals = [0, 1]
            min_num_bin = 1
            return_groupby = True
            dfm, dfstd = bin.bin_generic_2d(df,
                                            columns_to_bin,
                                            column_to_count,
                                            bins,
                                            round_to_decimals,
                                            min_num_bin,
                                            return_groupby
                                            )

            # setup
            """plot_each_bin = False
            plot_all_bins = False
            plot_all_pids = False
            plot_figs_separately = False
            plot_figs_separately_norm = False
            include_legend = False
            save_figs = True
            show_figs = False"""
            clr_map = cm.plasma(np.linspace(0.95, 0.15, len(bin_r)))

            # ---

            # export
            if export_binned_df:
                dfm['z_offset'] = z_offset_ul
                dfm['norm_r_by_microns'] = memb_radius * microns_per_pixel
                dfm['norm_dz_dr_by'] = t_membrane_norm
                dfm['r_bin_microns'] = dfm['bin_ll'] * microns_per_pixel
                dfm['r_bin_norm'] = dfm['bin_ll'] / memb_radius
                dfm['dz'] = dfm[dz_col] + z_offset_ul
                dfm['dz_norm'] = dfm['dz'] / t_membrane_norm
                dfm['dr_microns'] = dfm[r_disp_col] * microns_per_pixel
                dfm['dr_norm_microns'] = dfm['dr_microns'] / t_membrane_norm
                dfm['dr_norm_pixels'] = dfm[r_disp_col] / t_membrane_norm

                dfm.to_excel(path_figs_tid + '/dfm_2b-bin-frame-{}_disc-ul_num-bins={}_{}.xlsx'.format(
                    r_disp_col,
                    len(norm_r_bins),
                    read_raw_or_reconstruction),
                             )

            # ---

            if plot_each_bin:
                for br, br_lbl, clr in zip(bin_r, norm_r_bins, clr_map):
                    dfbr = dfm[dfm['bin_ll'] == br]
                    if len(dfbr) == 0:
                        continue

                    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                                   figsize=(size_x_inches * 1.5, size_y_inches * 1.25),
                                                   gridspec_kw={'height_ratios': [1, 1]})

                    # plot variables
                    arr_x = dfbr.t.to_numpy()
                    arr_z = dfbr[dz_col].to_numpy()
                    arr_r = dfbr[r_disp_col].to_numpy() * microns_per_pixel
                    ax1.plot(arr_x, arr_z, 'o', ms=1, color=clr, label=br_lbl)
                    ax2.plot(arr_x, arr_r, 'o', ms=1, lw=0.75, color=clr)

                    ax1.set_ylabel(r'$\Delta z^{\delta} \: (\mu m)$')
                    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.05), ncol=lgnd_cols, markerscale=2,
                               title=r'$r/a$')
                    ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
                    ax2.set_xlabel(r'$t \: (s)$')

                    plt.tight_layout()
                    if save_figs:
                        plt.savefig(path_figs_tid + '/2b-bin-frame-{}_disc-ul_bin={}_plasma.png'.format(r_disp_col, br))
                    if show_figs:
                        plt.show()
                    plt.close()

            # ---

            if plot_all_bins:
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                               figsize=(size_x_inches * 1.5, size_y_inches * 1.25),
                                               gridspec_kw={'height_ratios': [1, 1]})
                for br, br_lbl, clr in zip(bin_r, norm_r_bins, clr_map):
                    dfbr = dfm[dfm['bin_ll'] == br]
                    if len(dfbr) == 0:
                        continue

                    # plot variables
                    arr_x = dfbr.t.to_numpy()
                    arr_z = dfbr[dz_col].to_numpy()
                    arr_r = dfbr[r_disp_col].to_numpy() * microns_per_pixel

                    ax1.plot(arr_x, arr_z, 'o', ms=1, color=clr, label=br_lbl)
                    ax2.plot(arr_x, arr_r, 'o', ms=1, lw=0.75, color=clr)

                ax1.set_ylabel(r'$\Delta z^{\delta} \: (\mu m)$')
                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.05), ncol=lgnd_cols, markerscale=2,
                           title=r'$r/a$')
                ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
                ax2.set_xlabel(r'$t \: (s)$')

                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_figs_tid + '/2b-bin-frame-{}_disc-ul_num-bins={}_plasma.png'.format(r_disp_col,
                                                                                                         len(norm_r_bins)),
                                )
                if show_figs:
                    plt.show()
                plt.close()

            # ---

            if plot_all_pids:
                df = df.sort_values('r')
                clr_map_pids = cm.plasma(np.linspace(0.95, 0.15, len(df.id.unique())))

                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                               figsize=(size_x_inches * 1.5, size_y_inches * 1.25),
                                               gridspec_kw={'height_ratios': [1, 1]})
                for pid, clr in zip(df.id.unique(), clr_map_pids):
                    dfbr = df[df['id'] == pid]
                    # plot variables
                    arr_x = dfbr.t.to_numpy()
                    arr_z = dfbr[dz_col].to_numpy()
                    arr_r = dfbr[r_disp_col].to_numpy() * microns_per_pixel

                    ax1.plot(arr_x, arr_z, 'o', ms=1, color=clr)
                    ax2.plot(arr_x, arr_r, 'o', ms=1, lw=0.75, color=clr)

                ax1.set_ylabel(r'$\Delta z^{\delta} \: (\mu m)$')
                ax2.set_ylabel(r'$\Delta r \: (\mu m)$')
                ax2.set_xlabel(r'$t \: (s)$')

                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_figs_tid + '/2b-bin-frame-{}_disc-ul_plot-all-pids_plasma.png'.format(r_disp_col))
                if show_figs:
                    plt.show()
                plt.close()

            # ---
            # ---

            if plot_figs_separately:

                # plot z
                fig, ax1 = plt.subplots(figsize=(size_x_inches * 1., size_y_inches * 1.35 / 2))

                for br, br_lbl, clr in zip(bin_r, norm_r_bins, clr_map):
                    dfbr = dfm[dfm['bin_ll'] == br]
                    if len(dfbr) == 0:
                        ax1.plot(2, 500, 'o', ms=1, color=clr, label=br_lbl)
                        continue

                    # plot variables
                    arr_x = dfbr.t.to_numpy()
                    arr_z = dfbr[dz_col].to_numpy()

                    ax1.plot(arr_x, arr_z + z_offset_ul, 'o', ms=1, color=clr, label=br_lbl)

                if include_legend:
                    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.05), ncol=1, markerscale=2, title=r'$r/a$')
                ax1.set_ylabel(r'$\Delta z^{\delta} \: (\mu m)$')
                ax1.set_ylim([-75 + z_offset_ul, 75 + z_offset_ul])
                ax1.set_xlabel(r'$t \: (s)$')
                ax1.set_xticks([2, 4, 6, 8])

                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_figs_tid + '/2b-bin-frame-{}_disc-ul_num-bins={}_{}_only-z_units.png'.format(
                        r_disp_col,
                        len(norm_r_bins),
                        read_raw_or_reconstruction),

                                )
                if show_figs:
                    plt.show()
                plt.close()

                # ---

                # plot r
                fig, ax2 = plt.subplots(figsize=(size_x_inches * 1., size_y_inches * 1.35 / 2))

                for br, br_lbl, clr in zip(bin_r, norm_r_bins, clr_map):
                    dfbr = dfm[dfm['bin_ll'] == br]
                    if len(dfbr) == 0:
                        ax2.plot(2, 500, 'o', ms=1, color=clr, label=br_lbl)
                        continue

                    # plot variables
                    arr_x = dfbr.t.to_numpy()
                    arr_r = dfbr[r_disp_col].to_numpy() * microns_per_pixel

                    ax2.plot(arr_x, arr_r, 'o', ms=1, lw=0.75, color=clr, label=br_lbl)

                if include_legend:
                    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1.05), ncol=1, markerscale=2, title=r'$r/a$')
                ax2.set_ylabel(r'$\Delta r^{\delta} \: (\mu m)$')
                ax2.set_ylim([-3, 3.75])
                ax2.set_xlabel(r'$t \: (s)$')
                ax2.set_xticks([2, 4, 6, 8])

                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_figs_tid + '/2b-bin-frame-{}_disc-ul_num-bins={}_{}_only-r_units.png'.format(
                        r_disp_col,
                        len(norm_r_bins),
                        read_raw_or_reconstruction),

                                )
                if show_figs:
                    plt.show()
                plt.close()

            # ---

            # ---

            if plot_figs_separately_norm:

                # plot z
                fig, ax1 = plt.subplots(figsize=(size_x_inches * 1., size_y_inches * 1.35 / 2))

                for br, br_lbl, clr in zip(bin_r, norm_r_bins, clr_map):
                    dfbr = dfm[dfm['bin_ll'] == br]
                    if len(dfbr) == 0:
                        ax1.plot(2, 500, 'o', ms=1, color=clr, label=br_lbl)
                        continue

                    # plot variables
                    arr_x = dfbr.t.to_numpy()
                    arr_z = dfbr[dz_col].to_numpy()

                    ax1.plot(arr_x, (arr_z + z_offset_ul) / t_membrane_norm, 'o', ms=1, color=clr, label=br_lbl)

                if include_legend:
                    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.05), ncol=1, markerscale=2, title=r'$r/a$')
                ax1.set_ylabel(r'$\Delta z^{\delta}/t_{m}$')
                ax1.set_ylim([(-75 + z_offset_ul) / t_membrane_norm, (75 + z_offset_ul) / t_membrane_norm])
                ax1.set_xlabel(r'$t \: (s)$')
                ax1.set_xticks([2, 4, 6, 8])

                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_figs_tid + '/2b-bin-frame-{}_disc-ul_num-bins={}_{}_only-z_norm.png'.format(
                        r_disp_col,
                        len(norm_r_bins),
                        read_raw_or_reconstruction),

                                )
                if show_figs:
                    plt.show()
                plt.close()

                # ---

                # plot r
                fig, ax2 = plt.subplots(figsize=(size_x_inches * 1., size_y_inches * 1.35 / 2))

                for br, br_lbl, clr in zip(bin_r, norm_r_bins, clr_map):
                    dfbr = dfm[dfm['bin_ll'] == br]
                    if len(dfbr) == 0:
                        ax2.plot(2, 500, 'o', ms=1, color=clr, label=br_lbl)
                        continue

                    # plot variables
                    arr_x = dfbr.t.to_numpy()
                    arr_z = dfbr[dz_col].to_numpy()
                    arr_r = dfbr[r_disp_col].to_numpy() * microns_per_pixel

                    ax2.plot(arr_x, arr_r / t_membrane_norm, 'o', ms=1, lw=0.75, color=clr, label=br_lbl)

                if include_legend:
                    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1.05), ncol=1, markerscale=2, title=r'$r/a$')
                ax2.set_ylabel(r'$\Delta r^{\delta}/t_{m}$')
                ax2.set_ylim([-3 / t_membrane_norm, 3.75 / t_membrane_norm])
                ax2.set_xlabel(r'$t \: (s)$')
                ax2.set_xticks([2, 4, 6, 8])

                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_figs_tid + '/2b-bin-frame-{}_disc-ul_num-bins={}_{}_only-r_norm.png'.format(
                        r_disp_col,
                        len(norm_r_bins),
                        read_raw_or_reconstruction),

                                )
                if show_figs:
                    plt.show()
                plt.close()

            # ---


    # ---

    tids = [1]  # , 3]
    read_raw_or_reconstructions = ['raw', 'reconstruction']  # ['raw']  # , 'reconstruction']
    norm_r_binss = [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]  # , [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]

    for tid in tids:
        for read_raw_or_reconstruction in read_raw_or_reconstructions:
            for norm_r_bins in norm_r_binss:
                func_analyze_local_displacement_and_plot(tid, norm_r_bins, read_raw_or_reconstruction,
                                                         remove_initial_fpb_errors=False)

# ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 4. EXPORT SLOPE-CORRECTED TEST COORDS

export_slope_corrected_coords = True

if export_slope_corrected_coords:
    """ NOTE: no particle filtering is performed at this stage. Only processing. """

    tid = 1
    read_raw_or_reconstructions = ['reconstruction', 'raw']  # 'raw'  #  'reconstruction'

    for read_raw_or_reconstruction in read_raw_or_reconstructions:

        path_results_tid_coords = join(base_dir, 'results', 'dz{}'.format(tid))
        path_pubfigs = join(base_dir, 'pubfigs', str(tid), read_raw_or_reconstruction)

        if not os.path.exists(path_pubfigs):
            os.makedirs(path_pubfigs)

        dfd = pd.read_excel(
            path_results_tid_coords + '/df_{}_local_displacement_lr-ul_nd.xlsx'.format(read_raw_or_reconstruction))

        # ---

        # plot columns
        z_col = 'z_corr'
        dz_col = 'dz'
        r_disp_col = 'drg'
        r_disp_corr_col = r_disp_col + '_corr'

        # correct r-displacement for surface-slope-dependent apparent displacement
        dfd['apparent_dr'] = np.arcsin(np.deg2rad(dfd['nd_theta'])) * t_membrane_norm / 2
        dfd[r_disp_corr_col] = dfd[r_disp_col] + dfd['apparent_dr']

        # correct columns
        dfd['dz'] = dfd[z_col] + dfd['z_offset']

        # add columns to normalize by
        dfd['norm_dz_dr_by'] = t_membrane_norm

        # normalize columns
        dfd['r_norm'] = dfd['r'] / dfd['memb_radius']
        dfd['dz_norm'] = dfd['dz'] / dfd['norm_dz_dr_by']
        dfd['dr_norm'] = dfd[r_disp_col] / dfd['norm_dz_dr_by']
        dfd['dr_corr_norm'] = dfd[r_disp_corr_col] / dfd['norm_dz_dr_by']

        # ---

        # export

        path_slope_corrected_coords = join(base_dir, 'data/slope-corrected', 'dz{}'.format(tid))

        if not os.path.exists(path_slope_corrected_coords):
            os.makedirs(path_slope_corrected_coords)

        dfd.to_excel(path_slope_corrected_coords + '/df_{}_corr_displacement_lr-ul_nd.xlsx'.format(read_raw_or_reconstruction))

# ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 4. PLOT PUBLICATION FIGURES

plot_pubfigs = False

if plot_pubfigs:

    filter_saturated = True

    tid = 1
    read_raw_or_reconstructions = ['reconstruction', 'raw']  # 'raw'  #  'reconstruction'

    for read_raw_or_reconstruction in read_raw_or_reconstructions:

        path_results_tid_coords = join(base_dir, 'results', 'dz{}'.format(tid))
        path_pubfigs = join(base_dir, 'pubfigs', str(tid), read_raw_or_reconstruction)

        if not os.path.exists(path_pubfigs):
            os.makedirs(path_pubfigs)

        dfd = pd.read_excel(
            path_results_tid_coords + '/df_{}_local_displacement_lr-ul_nd.xlsx'.format(read_raw_or_reconstruction))

        # ---

        # filter
        if filter_saturated:
            dfd = dfd[~dfd['id'].isin(pids_saturated)]

        # ----------------------------------------------------------------------------------------------------------------------

        # plot columns
        z_col = 'z_corr'
        dz_col = 'dz'
        r_disp_col = 'drg'
        r_disp_corr_col = r_disp_col + '_corr'

        # correct r-displacement for surface-slope-dependent apparent displacement
        dfd['apparent_dr'] = np.arcsin(np.deg2rad(dfd['nd_theta'])) * t_membrane_norm / 2
        dfd[r_disp_corr_col] = dfd[r_disp_col] + dfd['apparent_dr']

        # correct columns
        dfd['dz'] = dfd[z_col] + dfd['z_offset']

        # add columns to normalize by
        dfd['norm_dz_dr_by'] = t_membrane_norm

        # normalize columns
        dfd['r_norm'] = dfd['r'] / dfd['memb_radius']
        dfd['dz_norm'] = dfd['dz'] / dfd['norm_dz_dr_by']
        dfd['dr_norm'] = dfd[r_disp_col] / dfd['norm_dz_dr_by']
        dfd['dr_corr_norm'] = dfd[r_disp_corr_col] / dfd['norm_dz_dr_by']

        # ---

        x_col = 't'
        memb_id_col = 'memb_id'

        save_figs = True
        show_figs = True
        include_legend = False
        include_colobar = True

        # plot as scatter
        p_scatter = True
        marker_size = 0.25
        # cmap = 'plasma_r'
        # cmap = mpl.cm.plasma
        # norm = mpl.colors.Normalize(vmin=0, vmax=1)

        vmin, vmax = 0, 1  # dfd['r_norm'].min(), dfd['r_norm'].max()
        cmap = mpl.cm.plasma_r
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        # plot individual pids
        clr_map_pids = cm.plasma(np.linspace(0.95, 0.15, len(dfd.id.unique())))


        # ---

        def plot_pubfig_local_displacement(df, memb_id, p_id, normamlize_dzr, norm_z_by_max_dz,
                                           include_tilt_correction):

            if normamlize_dzr:
                norm_z = t_membrane_norm
                ylbl_z = r'$\Delta z / t_{m}$'

                norm_r = t_membrane_norm
                ylbl_r = r'$\Delta r / t_{m}$'
                ylbl_r_corr = r'$\Delta r_{mid} / t_{m}$'
            else:
                norm_z = 1
                ylbl_z = r'$\Delta z \: (\mu m)$'

                norm_r = 1
                ylbl_r = r'$\Delta r \: (\mu m)$'
                ylbl_r_corr = r'$\Delta r_{mid} \: (\mu m)$'

            # prepare data
            if p_id is not None:
                df = df[df['id'] == p_id]
                save_id = 'pid-{}'.format(p_id)
            else:
                df = df[df[memb_id_col] == memb_id]
                save_id = 'disc-{}'.format(memb_id)

            df = df[df[x_col] > start_time]
            df = df.sort_values('x')

            # modify norm_z
            if normamlize_dzr and norm_z_by_max_dz is not None:
                norm_z = norm_z_by_max_dz  # df[dz_col].abs().max()
                ylbl_z = r'$\Delta z / w_{o}^{*}$'

            # plot
            if include_tilt_correction:
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True,
                                                    figsize=(size_x_inches * 1.1, size_y_inches * 1.25),
                                                    gridspec_kw={'height_ratios': [1, 1, 1]})

                ax1.scatter(df[x_col].to_numpy(), df[dz_col].to_numpy() / norm_z,
                            c=cmap(norm(df['r_norm'])),
                            # cmap=cmap,
                            marker='o', s=marker_size)

                ax2.scatter(df[x_col].to_numpy(), df[r_disp_col].to_numpy() * microns_per_pixel / norm_r,
                            c=cmap(norm(df['r_norm'])),
                            marker='o', s=marker_size)

                ax3.scatter(df[x_col].to_numpy(), df[r_disp_corr_col].to_numpy() * microns_per_pixel / norm_r,
                            c=cmap(norm(df['r_norm'])),
                            marker='o', s=marker_size)
                ax3.set_ylabel(ylbl_r_corr)
                ax3.set_xlabel(r'$t \: (s)$')

            else:
                fig, (ax1, ax2) = plt.subplots(nrows=2,
                                               sharex=True,
                                               figsize=(size_x_inches * 1.1, size_y_inches),
                                               gridspec_kw={'height_ratios': [1, 1]})

                ax1.scatter(df[x_col].to_numpy(), df[dz_col].to_numpy() / norm_z,
                            c=cmap(norm(df['r_norm'])),
                            # cmap=cmap,
                            marker='o', s=marker_size)

                ax2.scatter(df[x_col].to_numpy(), df[r_disp_col].to_numpy() * microns_per_pixel / norm_r,
                            c=cmap(norm(df['r_norm'])),
                            marker='o', s=marker_size)
                ax2.set_xlabel(r'$t \: (s)$')

            ax1.set_ylabel(ylbl_z)
            ax2.set_ylabel(ylbl_r)
            plt.tight_layout()

            if include_colobar:
                cbl, cbb, cbw, cbh = 0.85, 0.1, 0.075, 0.8
                cbl = cbl + 0.05
                cbw = cbw / 2
                plt.subplots_adjust(bottom=0.1, right=cbl - 0.05, top=0.9)
                cax = plt.axes([cbl, 0.1, cbw, 0.8])  # left, bottom, width, height
                plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                             aspect=40,
                             label=r'$r/a$', ticks=[0, 1])

                # plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical', label=r'$r/a$')

            if save_figs:
                plt.savefig(path_pubfigs +
                            '/{}_plot-{}_{}_{}_{}-corr_by_{}_norm{}-norm-max-dz{}_cbar{}_slope-corrected-{}.png'.format(
                                save_id,
                                read_raw_or_reconstruction,
                                dz_col,
                                r_disp_col, r_disp_col,
                                x_col,
                                normamlize_dzr,
                                norm_z_by_max_dz,
                                include_colobar,
                                include_tilt_correction,
                            ),
                            )
            if show_figs:
                plt.show()
            plt.close()


        # ---

        # plot by particle_id
        plot_per_pid = False
        if plot_per_pid:
            for memb_idd, max_w0 in zip([1, 2], [lr_w0_max, ul_w0_max]):
                dfdm = dfd[dfd['memb_id'] == memb_idd]

                disc_boundary_pids = dfdm.id.unique()  # dfdm[dfdm['r'] / dfdm['memb_radius'] > 0.8].id.unique()

                for p_idd in disc_boundary_pids:
                    for normalize_rz_disp in [True]:
                        plot_pubfig_local_displacement(df=dfd,
                                                       memb_id=None,
                                                       p_id=p_idd,
                                                       normamlize_dzr=normalize_rz_disp,
                                                       norm_z_by_max_dz=max_w0,
                                                       include_tilt_correction=True,
                                                       )

        # ---

        # plot by membrane_id
        for memb_idd in [1, 2]:
            dfd[r_disp_col] = dfd[r_disp_col].where(~dfd['id'].isin([60, 61]), np.nan)
            for normalize_rz_disp in [False, True]:
                plot_pubfig_local_displacement(df=dfd,
                                               memb_id=memb_idd,
                                               p_id=None,
                                               normamlize_dzr=normalize_rz_disp,
                                               norm_z_by_max_dz=normalize_rz_disp,
                                               include_tilt_correction=False,
                                               )

        # ---

    # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 4. PLOT PARTICLE VELOCITY BY FITTING A CURVE

plot_particle_velocity = False

if plot_particle_velocity:

    tid = 1
    read_raw_or_reconstruction = 'reconstruction'

    path_results_velocity = join(base_dir, 'results', 'dz{}'.format(tid), 'velocity')
    if not os.path.exists(path_results_velocity):
        os.makedirs(path_results_velocity)

    path_results_tid_coords = join(base_dir, 'results', 'dz{}'.format(tid))
    dfd = pd.read_excel(
        path_results_tid_coords + '/df_{}_local_displacement_lr-ul_nd.xlsx'.format(read_raw_or_reconstruction))

    dfd['dz'] = dfd['z_corr'] + dfd['z_offset']
    dfd = dfd[dfd['t'] > start_time]

    # ---

    from numpy.polynomial import Polynomial

    # one particle per plot
    plot_each_pid = False
    if plot_each_pid:
        for memb_id in [1, 2]:

            df = dfd[dfd['memb_id'] == memb_id]

            for pid in [80, 94, 50, 69, 71, 81, 10, 30, 41, 53, 57, 60]:
                dfpid = df[df['id'] == pid][['t', 'dz']]
                dfpid = dfpid.dropna()

                if len(dfpid) < 10:
                    continue

                # fit
                pf = Polynomial.fit(dfpid['t'], dfpid['dz'], 12)
                dpf = pf.deriv()
                tf = np.linspace(dfpid['t'].min(), dfpid['t'].max(), 300)
                texp = np.arange(dfpid['t'].min(), dfpid['t'].max() + 1 / frame_rate, 1 / frame_rate)

                # plot
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

                ax1.plot(dfpid['t'], dfpid['dz'], 'o', ms=1, color=sciblue, label='Exp.')
                ax1.plot(tf, pf(tf), color=sciblack, label='Fit')

                ax3 = ax2.twinx()
                ax3.plot(texp, dpf(texp) * exposure_time, 'o', ms=1, color=lighten_color('r', 1.1))

                ax2.plot(tf, dpf(tf), color='k')

                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), markerscale=1.5, handlelength=0.4)
                ax1.set_ylabel(r'$z \: (\mu m)$')
                ax2.set_ylabel(r'$\vec{z} \: (\mu m / s)$')
                ax3.set_ylabel(r'$\Delta z_{exposure} \: (\mu m)$', color=lighten_color('r', 1.1))
                ax2.set_xlabel(r'$t \: (s)$')

                ax1.tick_params(axis='y', which='minor',
                                bottom=False, top=False, left=False, right=False,
                                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                ax2.tick_params(axis='y', which='minor',
                                bottom=False, top=False, left=False, right=False,
                                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                ax3.tick_params(axis='y', which='minor',
                                bottom=False, top=False, left=False, right=False,
                                labelbottom=False, labeltop=False, labelleft=False, labelright=False)

                plt.tight_layout()
                plt.savefig(path_results_velocity + '/pid-{}_plot_dz-dz-vec-dz-exposure_by_t.png'.format(pid))
                plt.show()

    # ---

    # plot histogram of velocities
    from numpy.polynomial import Polynomial
    pids = dfd.id.unique()

    dz_exps = []
    for pid in pids:
        dfpid = df[df['id'] == pid][['t', 'dz']]
        dfpid = dfpid.dropna()

        # fit
        pf = Polynomial.fit(dfpid['t'], dfpid['dz'], 12)
        dpf = pf.deriv()

        # axial displacement per-exposure
        texp = np.arange(dfpid['t'].min(), dfpid['t'].max() + 1 / frame_rate, 1 / frame_rate)
        dz_exp = dpf(texp) * exposure_time


# ---

# ----------------------------------------------------------------------------------------------------------------------
# 4. Plot frames of interest using binned dataframe of local displacement

post_process_local_displacement_and_plot = False

if post_process_local_displacement_and_plot:

    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/02.06.22_local_displacement/' \
               'results/dz1/figs/pubfigs'

    fplr = '/dfm_2b-bin-frame-drg_disc-lr_num-bins=11_raw.xlsx'
    fpul = '/dfm_2b-bin-frame-drg_disc-ul_num-bins=11_raw.xlsx'

    dflr = pd.read_excel(base_dir + fplr)
    dful = pd.read_excel(base_dir + fpul)

    # create some columns
    dflr['r_bin_norm'] = dflr['bin_ll'] / r_edge_lr
    dful['r_bin_norm'] = dful['bin_ll'] / r_edge_ul

    """dfm['z_offset'] = z_offset
    dfm['norm_r_by_microns'] = memb_radius * microns_per_pixel
    dfm['norm_dz_dr_by'] = t_membrane_norm
    dfm['r_bin_microns'] = dfm['bin_ll'] * microns_per_pixel
    dfm['r_bin_norm'] = dfm['bin_ll'] / memb_radius
    dfm['dz'] = dfm[dz_col] + z_offset
    dfm['dz_norm'] = dfm['dz'] / t_membrane_norm
    dfm['dr_microns'] = dfm[r_disp_col] * microns_per_pixel
    dfm['dr_norm_microns'] = dfm['dr_microns'] / t_membrane_norm
    dfm['dr_norm_pixels'] = dfm[r_disp_col] / t_membrane_norm"""

    times_of_interest = [[4.0, 4.2], [4.5, 4.75], [5.15, 5.35], [5.75, 6]]

    for toi in times_of_interest:
        dfl = dflr[(dflr['t'] > toi[0]) & (dflr['t'] < toi[1])].groupby('bin_ll').mean().reset_index()
        dfu = dful[(dful['t'] > toi[0]) & (dful['t'] < toi[1])].groupby('bin_ll').mean().reset_index()

        fig, ax = plt.subplots()

        ax.plot(dfl.r_bin_norm, dfl.dr_norm_microns, '-o', label=800)
        ax.plot(dfu.r_bin_norm, dfu.dr_norm_microns, '-o', label=500)

        ax.legend()
        ax.set_xlabel(r'$r/a$')
        ax.set_ylabel(r'$\Delta r^{\delta} / t_{m}$')
        ax.grid()

        plt.tight_layout()
        plt.show()

print("Analysis completed without errors.")