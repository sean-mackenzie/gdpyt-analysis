# 02.06.22 - local axial and radial displacements per membrane

# imports
import os
from os.path import join

import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, Akima1DInterpolator

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

import analyze
from utils import boundary, functions, io, bin, plotting
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
exclude_pids = [39, 61]
bad_pids = [12, 13, 18, 34, 39, 49, 61, 66, 78]

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 1. FILES PATHS

base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/02.06.22_exposure_displacement'
path_data = join(base_dir, 'data')
path_results = join(base_dir, 'results')


# ----------------------------------------------------------------------------------------------------------------------

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 2. PROCESSING
calculate_exposure_displacement = False
if calculate_exposure_displacement:

    tid = 1
    data_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/02.06.22_local_displacement'
    path_results_tid_coords = join(data_dir, 'results', 'dz{}'.format(tid))
    read_raw_or_reconstruction = 'reconstruction'
    dfd = pd.read_excel(path_results_tid_coords + '/df_{}_local_displacement_lr-ul_nd.xlsx'.format(read_raw_or_reconstruction))

    # ---

    # ----------------------------------------------------------------------------------------------------------------------
    # 2. DEFINE PROCESSING FUNCTIONS

    def fit_and_calculate_derivative_pid(dfpid, replace_col, fit_by_col, fit_to_col, new_cols_str, poly_deg, dt_exposure):
        # setup new column names
        poly_fit = new_cols_str + '_poly'
        poly_fit_error = new_cols_str + '_poly_error'
        dpoly_fit_dt = 'd{}_poly_dt'.format(new_cols_str)
        dpoly_fit_exposure = 'd{}_poly_exposure'.format(new_cols_str)
        d_exposure = 'd{}_exposure'.format(new_cols_str)

        # get data w/o NaNs
        dfpid_no_nans = dfpid[[replace_col, fit_by_col, fit_to_col]].dropna()

        # fit
        pf = Polynomial.fit(dfpid_no_nans[fit_by_col], dfpid_no_nans[fit_to_col], poly_deg)
        dpf = pf.deriv()

        # calculate columns
        dfpid_no_nans[poly_fit] = pf(dfpid_no_nans[fit_by_col])
        dfpid_no_nans[poly_fit_error] = dfpid_no_nans[fit_to_col] - dfpid_no_nans[poly_fit]
        dfpid_no_nans[dpoly_fit_dt] = dpf(dfpid_no_nans[fit_by_col])
        dfpid_no_nans[dpoly_fit_exposure] = dpf(dfpid_no_nans[fit_by_col]) * dt_exposure
        dfpid_no_nans[d_exposure] = dfpid[fit_to_col].diff() / dfpid[fit_by_col].diff() * dt_exposure

        # create mapping dict
        mapper_dict = dfpid_no_nans[
            [replace_col, poly_fit, poly_fit_error, dpoly_fit_dt, dpoly_fit_exposure, d_exposure]
        ].set_index(replace_col).to_dict()

        # create columns to map to
        dfpid[poly_fit] = dfpid[replace_col]
        dfpid[poly_fit_error] = dfpid[replace_col]
        dfpid[dpoly_fit_dt] = dfpid[replace_col]
        dfpid[dpoly_fit_exposure] = dfpid[replace_col]
        dfpid[d_exposure] = dfpid[replace_col]

        # apply mapping
        dfpid = dfpid.replace({poly_fit: mapper_dict[poly_fit],
                               poly_fit_error: mapper_dict[poly_fit_error],
                               dpoly_fit_dt: mapper_dict[dpoly_fit_dt],
                               dpoly_fit_exposure: mapper_dict[dpoly_fit_exposure],
                               d_exposure: mapper_dict[d_exposure],
                               })

        for fix_col in [poly_fit, poly_fit_error, dpoly_fit_dt, dpoly_fit_exposure, d_exposure]:
            dfpid[fix_col] = dfpid[fix_col].where(dfpid[fix_col] != dfpid[replace_col], np.nan)

        return dfpid


    # ----------------------------------------------------------------------------------------------------------------------
    # 2. PROCESSING

    # --- GLOBAL PROCESSING

    dfd = dfd[dfd['memb_id'].isin([1, 2])]
    dfd = dfd[~dfd['id'].isin(bad_pids)]
    dfd['z_true'] = dfd['d_dz'] - dfd['z_offset']
    dfd['error'] = dfd['z_corr'] - dfd['z_true']

    # -

    # --- SPECIFIC PROCESSING + EXPORT
    df = dfd.copy()
    df = df[df['t'] > start_time]
    pids = df.id.unique()

    dfpid_zs = []
    dfpid_rs = []
    for pid in pids:
        dfpid_ = df[df['id'] == pid]

        # STEP 1:   fit polynomial to z-positions
        #               COLUMNS ADDED: 'z_fit', 'z_fit_error', 'dz_exp'
        replace_col_ = 'frame'
        fit_by_col_ = 't'
        fit_to_col_ = 'z_corr'
        new_cols_str_ = 'z'
        poly_deg_ = 12
        dt_exposure_ = exposure_time
        dfpid_z = fit_and_calculate_derivative_pid(dfpid=dfpid_,
                                                  replace_col=replace_col_,
                                                  fit_by_col=fit_by_col_,
                                                  fit_to_col=fit_to_col_,
                                                  new_cols_str=new_cols_str_,
                                                  poly_deg=poly_deg_,
                                                  dt_exposure=dt_exposure_,
                                                  )

        # ---

        # STEP 2:   fit polynomial to r-positions
        #               COLUMNS ADDED: 'z_fit', 'z_fit_error', 'dz_exp'
        fit_to_col_ = 'rg'
        new_cols_str_ = 'r'
        dfpid_r = fit_and_calculate_derivative_pid(dfpid=dfpid_,
                                                  replace_col=replace_col_,
                                                  fit_by_col=fit_by_col_,
                                                  fit_to_col=fit_to_col_,
                                                  new_cols_str=new_cols_str_,
                                                  poly_deg=poly_deg_,
                                                  dt_exposure=dt_exposure_,
                                                  )

        dfpid_zs.append(dfpid_z)
        dfpid_rs.append(dfpid_r)

    # package
    dfpid_zs = pd.concat(dfpid_zs)
    dfpid_rs = pd.concat(dfpid_rs)

    # export
    dfpid_zs.to_excel(join(path_data, 'df_{}_local_displacement_dz-exposure.xlsx'.format(read_raw_or_reconstruction)))
    dfpid_rs.to_excel(join(path_data, 'df_{}_local_displacement_dr-exposure.xlsx'.format(read_raw_or_reconstruction)))

    # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------

# ---


# ----------------------------------------------------------------------------------------------------------------------
# 2. PROCESSING - Z-DISPLACEMENT

analyze_z = True
if analyze_z:

    # setup
    new_cols_str = 'z'

    # create file paths
    path_results_hist = join(path_results, new_cols_str, 'histograms')
    path_results_exposure_disp = join(path_results, new_cols_str, 'exposure-displacement')
    path_results_exposure_disp_per_pid = join(path_results_exposure_disp, 'per-pid')

    if not os.path.exists(path_results_hist):
        os.makedirs(path_results_hist)
    if not os.path.exists(path_results_exposure_disp):
        os.makedirs(path_results_exposure_disp)
    if not os.path.exists(path_results_exposure_disp_per_pid):
        os.makedirs(path_results_exposure_disp_per_pid)

    # ---

    # define variables
    fit_by_col = 't'
    fit_to_col = 'z_corr'
    poly_fit = new_cols_str + '_poly'
    poly_fit_error = new_cols_str + '_poly_error'
    dpoly_fit_dt = 'd{}_poly_dt'.format(new_cols_str)
    dpoly_fit_exposure = 'd{}_poly_exposure'.format(new_cols_str)
    d_exposure = 'd{}_exposure'.format(new_cols_str)

    # read dataframe
    dfz = pd.read_excel(join(path_data, 'df_reconstruction_local_displacement_dz-exposure.xlsx'))

    df = dfz.copy()
    pids = df['id'].unique()

    # ---

    # ----------------------------------------------------
    # 3. PLOT VELOCITY AND EXPOSURE DISPLACEMENT PER PARTICLE

    # per particle

    plot_velocity_and_exposure_displacement_per_pid = False
    if plot_velocity_and_exposure_displacement_per_pid:

        # plot z, dz/dt (velocity), and dz(dt_exposure)
        x = fit_by_col
        y1 = fit_to_col
        y12 = poly_fit
        y2 = dpoly_fit_dt
        y3 = d_exposure
        y32 = dpoly_fit_exposure

        # plot setup
        y1lbl = r'$z \: (\mu m)$'
        y2lbl = r'$\vec{u_{z}} \: (\mu m / s)$'
        y3lbl = r'$\Delta z_{exposure} \: (\mu m)$'
        xlbl = r'$t \: (s)$'
        ms = 1
        lw = 0.75

        # plot
        for pid in pids:
            dfpid = df[df['id'] == pid]

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.))

            ax1.plot(dfpid[x], dfpid[y1], 'o', ms=ms, label=pid)
            ax1.plot(dfpid[x], dfpid[y12], '-', linewidth=lw, color=sciblack)
            ax2.plot(dfpid[x], dfpid[y2], '-', color=sciblack)
            ax3.plot(dfpid[x], dfpid[y3], 'o', ms=ms)
            ax3.plot(dfpid[x], dfpid[y32], '-', linewidth=lw, color=sciblack)

            ax1.legend()
            ax1.set_ylabel(y1lbl)
            ax2.set_ylabel(y2lbl)
            ax3.set_ylabel(y3lbl)
            ax3.set_xlabel(xlbl)

            plt.tight_layout()
            plt.savefig(path_results_exposure_disp_per_pid +
                        '/pid{}_{}_position_velocity_exposure-displacement_by_{}.png'.format(pid, new_cols_str, x))
            plt.show()

    # ----------------------------------------------------
    # 3. PLOTS RELATING TO LOCALIZED Z-POSITION

    histogram_exposure_displacement_relative_z = True
    if histogram_exposure_displacement_relative_z:

        # histogram(x: z-position, y: z-velocity)
        x = fit_to_col
        y = dpoly_fit_dt

        df_ = df[[x, y]].dropna()
        xlbl = r'$z \: (\mu m)$'
        xticks = None
        ylbl = r'$\vec{u_{z}}^{fit} \: (\mu m / s)$'
        save_path = path_results_exposure_disp + '/{}_hist_{}_by_{}.png'.format(new_cols_str, x, y)

        plotting.scatter_and_kde_x_and_y(df_[x], df_[y],
                                         binwidth_x=2, binwidth_y=25,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=1, bandwidth_y=25,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

        # ---

        # histogram(x: z-position, y: exposure displacement)
        x = fit_to_col
        y = dpoly_fit_exposure

        df_ = df[[x, y]].dropna()
        xlbl = r'$z \: (\mu m)$'
        xticks = None
        ylbl = r'$\Delta z_{exposure}^{fit} \: (\mu m)$'
        save_path = path_results_exposure_disp + '/{}_hist_{}_by_{}.png'.format(new_cols_str, x, y)

        plotting.scatter_and_kde_x_and_y(df_[x], df_[y],
                                         binwidth_x=2, binwidth_y=0.5,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=1, bandwidth_y=1,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

        # ---

        # histogram(x: normalized r-position, y: measured z-position)
        x = 'r'
        y = fit_to_col

        df_ = df[[x, y, 'memb_radius']].dropna()
        xlbl = r'$r / a$'
        xticks = None
        ylbl = r'$z \: (\mu m)$'
        save_path = path_results_exposure_disp + '/{}_hist_{}_by_{}.png'.format(new_cols_str, x, y)

        plotting.scatter_and_kde_x_and_y(df_[x] / df_['memb_radius'], df_[y],
                                         binwidth_x=0.025, binwidth_y=5,
                                         kde_x=False, kde_y=False,
                                         bandwidth_x=1, bandwidth_y=5,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

        # ---

        # ONLY PARTICLES R-POSITION > 0.9 (r/a)
        # histogram(x: normalized r-position, y: measured z-position)
        x = 'r'
        y = fit_to_col

        df_ = df[df[x] / df['memb_radius'] > 0.875][[x, y, 'memb_radius']].dropna()
        xlbl = r'$r / a$'
        xticks = None
        ylbl = r'$z \: (\mu m)$'
        save_path = path_results_exposure_disp + '/{}_hist_boundary-particles_{}_by_{}.png'.format(new_cols_str, x, y)

        plotting.scatter_and_kde_x_and_y(df_[x] / df_['memb_radius'], df_[y],
                                         binwidth_x=0.01, binwidth_y=1,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=1, bandwidth_y=2,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

        # ---

        # histogram(x: normalized r-position, y: fitted z exposure displacement)
        x = 'r'
        y = dpoly_fit_exposure

        df_ = df[[x, y, 'memb_radius']].dropna()
        xlbl = r'$r / a$'
        xticks = None
        ylbl = r'$\Delta z_{exposure}^{fit} \: (\mu m)$'
        save_path = path_results_exposure_disp + '/{}_hist_{}_by_{}.png'.format(new_cols_str, x, y)

        plotting.scatter_and_kde_x_and_y(df_[x] / df_['memb_radius'], df_[y],
                                         binwidth_x=0.1, binwidth_y=0.5,
                                         kde_x=False, kde_y=False,
                                         bandwidth_x=1, bandwidth_y=1,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

        # ---

    # ---

    # ----------------------------------------------------
    # 3. PLOTS RELATING TO FITTED Z-POSITION

    histogram_exposure_displacement_relative_z_fit = False
    if histogram_exposure_displacement_relative_z_fit:
        # histogram(x: poly fit z-position, y: poly fit z-velocity)
        x = poly_fit
        y = dpoly_fit_dt
        c = 'nd_dr'
        cmap = 'viridis'

        df_ = df[[x, y, c]].dropna()
        xlbl = r'$z^{fit} \: (\mu m)$'
        xticks = None
        ylbl = r'$\vec{u_{z}}^{fit} \: (\mu m / s)$'
        save_path = path_results_exposure_disp + '/{}_hist_{}_by_{}.png'.format(new_cols_str, x, y)

        plotting.scatter_and_kde_x_and_y(df_[x], df_[y],
                                         binwidth_x=2, binwidth_y=25,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=1, bandwidth_y=25,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

        # ---

        # histogram(x: z-position, y: exposure displacement)
        x = poly_fit
        y = dpoly_fit_exposure

        df_ = df[[x, y, c]].dropna()
        xlbl = r'$z^{fit} \: (\mu m)$'
        xticks = None
        ylbl = r'$\Delta z_{exposure}^{fit} \: (\mu m)$'
        save_path = path_results_exposure_disp + '/{}_hist_{}_by_{}.png'.format(new_cols_str, x, y)

        plotting.scatter_and_kde_x_and_y(df_[x], df_[y],
                                         binwidth_x=2, binwidth_y=0.5,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=1, bandwidth_y=1,
                                         color=df_[c], colormap=cmap, scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

        # ---

    # ---

    # ----------------------------------------------------
    # 3. PLOTS RELATING TO Z-ERRORS

    histogram_exposure_displacement_relative_z_error = False
    if histogram_exposure_displacement_relative_z_error:

        # ---

        # histogram(x: poly fit z-position, y: poly fit z-velocity)
        x = dpoly_fit_exposure
        y = 'error'

        df_ = df[[x, y]].dropna()
        xlbl = r'$\Delta z_{exposure} \: (\mu m)$'
        xticks = None
        ylbl = r'$\epsilon_{z} \: (\mu m)$'
        save_path = path_results_exposure_disp + '/{}_hist_{}_by_{}.png'.format(new_cols_str, x, y)

        plotting.scatter_and_kde_x_and_y(df_[x], df_[y],
                                         binwidth_x=1, binwidth_y=0.5,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=1, bandwidth_y=1,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

        # ---

        # histogram(x: poly fit z-position, y: poly fit z-velocity)
        x = dpoly_fit_exposure
        y = poly_fit_error

        df_ = df[[x, y]].dropna()
        xlbl = r'$\Delta z_{exposure} \: (\mu m)$'
        xticks = None
        ylbl = r'$\epsilon_{z}^{fit} \: (\mu m)$'
        save_path = path_results_exposure_disp + '/{}_hist_{}_by_{}.png'.format(new_cols_str, x, y)

        plotting.scatter_and_kde_x_and_y(df_[x], df_[y],
                                         binwidth_x=1, binwidth_y=0.5,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=1, bandwidth_y=1,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

        # ---


        raise ValueError()

        # ---

        # histogram(x: poly fit z-position, y: poly fit z-velocity)

        plotting.scatter_and_kde_x_and_y(x, y,
                                         binwidth_x=1, binwidth_y=0.5,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=1, bandwidth_y=1,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

        # ---

        save_path = path_results_exposure_disp + '/{}_hist_z_by_axial-exposure-displacement.png'.format(new_cols_str)
        xlbl = r'$\Delta z_{exposure} \: (\mu m)$'  # r'$\vec{z} \: (\mu m / s)$'
        xticks = None
        ylbl = r'$z \: (\mu m)$'
        plotting.scatter_and_kde_x_and_y(dfpids['dz_exp'], dfpids['z_corr'],
                                         binwidth_x=1, binwidth_y=5,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=1, bandwidth_y=10,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

        save_path = path_results_exposure_disp + '/{}_hist_z-fit_by_axial-exposure-displacement.png'.format(new_cols_str)
        xlbl = r'$\Delta z_{exposure} \: (\mu m)$'  # r'$\vec{z} \: (\mu m / s)$'
        xticks = None
        ylbl = r'$z_{fit} \: (\mu m)$'
        plotting.scatter_and_kde_x_and_y(dfpids['dz_exp'], dfpids['z_fit'],
                                         binwidth_x=1, binwidth_y=5,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=1, bandwidth_y=10,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )


        ylbl = r'$\Delta z_{exposure} \: (\mu m)$'  # r'$\vec{z} \: (\mu m / s)$'
        xticks = None
        xlbl = r'$\Delta z \: (\mu m)$'
        save_path = path_results_exposure_disp + '/{}_hist_exposure-displacement_by_d{}.png'.format(new_cols_str,
                                                                                                    new_cols_str)

        plotting.scatter_and_kde_x_and_y(dfpids['d_dz_corr'], dfpids['dz_exp'],
                                         binwidth_x=5, binwidth_y=1,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=10, bandwidth_y=1,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

    # -----------------------------------------------------
    # 3. PLOTS RELATING ERROR

    histogram_error_by_zr = False
    if histogram_error_by_zr:

        # local processing
        df = dfd.copy()
        df = df.dropna(subset=['z_corr', 'error'])

        # ---

        # error by axial position

        x = df['z_corr']
        y = df['error']
        xlbl = r'$z \: (\mu m)$'
        ylbl = r'$\epsilon_{z} \: (\mu m)$'
        save_path = path_results_hist + '/hist_erorr_by_z.png'

        plotting.scatter_and_kde_x_and_y(x, y,
                                         binwidth_x=2.5, binwidth_y=1,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=5, bandwidth_y=1,
                                         color=None, colormap='coolwarm', scatter_size=0.25,
                                         xlbl=xlbl, ylbl=ylbl,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True)

        # ---

        # error by radial position

        x = df['nd_r']
        y = df['error']
        xlbl = r'$r/a$'
        xticks = [0, 0.5, 1]
        ylbl = r'$\epsilon_{z} \: (\mu m)$'
        save_path = path_results_hist + '/hist_erorr_by_r.png'

        plotting.scatter_and_kde_x_and_y(x, y,
                                         binwidth_x=0.1, binwidth_y=1,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=0.2, bandwidth_y=1,
                                         color=None, colormap='coolwarm', scatter_size=0.25,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

    # ----

    # ------------------------------------------------------
    # 3. PLOTS RELATING FIT

    histogram_fit_zr_by_zr = False
    if histogram_error_by_zr:

        # histogram: z by z_fit
        save_path = path_results_uncertainty + '/hist_z_by_z_fit.png'
        xlbl = r'$z_{fit} \: (\mu m)$'  # r'$\vec{z} \: (\mu m / s)$'
        xticks = None
        ylbl = r'$z \: (\mu m)$'
        plotting.scatter_and_kde_x_and_y(dfpids['z_fit'], dfpids['z_corr'],
                                         binwidth_x=5, binwidth_y=5,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=10, bandwidth_y=10,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

        # histogram: z by z_fit
        save_path = path_results_uncertainty + '/hist_z_fit_error_by_z_fit.png'
        xlbl = r'$z_{fit} \: (\mu m)$'  # r'$\vec{z} \: (\mu m / s)$'
        xticks = None
        ylbl = r'$\epsilon_{z_{f}} \: (\mu m)$'
        plotting.scatter_and_kde_x_and_y(dfpids['z_fit'], dfpids['z_fit_error'],
                                         binwidth_x=5, binwidth_y=1,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=10, bandwidth_y=2,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )




            # ---

# ---


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 2. PROCESSING - R-DISPLACEMENT

analyze_r = False
if analyze_r:

    # setup
    new_cols_str = 'r'

    # create file paths
    path_results_hist = join(path_results, new_cols_str, 'histograms')
    path_results_exposure_disp = join(path_results, new_cols_str, 'exposure-displacement')
    path_results_exposure_disp_per_pid = join(path_results_exposure_disp, 'per-pid')

    if not os.path.exists(path_results_hist):
        os.makedirs(path_results_hist)
    if not os.path.exists(path_results_exposure_disp):
        os.makedirs(path_results_exposure_disp)
    if not os.path.exists(path_results_exposure_disp_per_pid):
        os.makedirs(path_results_exposure_disp_per_pid)

    # ---

    # define variables
    fit_by_col = 't'
    dfit_to_col = 'drg'
    fit_to_col = 'rg'
    poly_fit = new_cols_str + '_poly'
    poly_fit_error = new_cols_str + '_poly_error'
    dpoly_fit_dt = 'd{}_poly_dt'.format(new_cols_str)
    dpoly_fit_exposure = 'd{}_poly_exposure'.format(new_cols_str)
    d_exposure = 'd{}_exposure'.format(new_cols_str)

    # read dataframe
    dfr = pd.read_excel(join(path_data, 'df_reconstruction_local_displacement_dr-exposure.xlsx'))

    df = dfr.copy()
    pids = df['id'].unique()

    # ---

    # ----------------------------------------------------
    # 3. PLOT VELOCITY AND EXPOSURE DISPLACEMENT

    plot_velocity_and_exposure_displacement_per_pid = False
    if plot_velocity_and_exposure_displacement_per_pid:

        # plot z, dz/dt (velocity), and dz(dt_exposure)
        x = fit_by_col
        y1 = fit_to_col
        y12 = poly_fit
        y2 = dpoly_fit_dt
        y3 = d_exposure
        y32 = dpoly_fit_exposure

        # plot setup
        y1lbl = r'$r \: (pix.)$'
        y2lbl = r'$\vec{u_{r}} \: (pix. / s)$'
        y3lbl = r'$\Delta r_{exposure} \: (pix.)$'
        xlbl = r'$t \: (s)$'
        ms = 1
        lw = 0.5

        # plot
        for pid in pids:
            dfpid = df[df['id'] == pid]

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.))

            ax1.plot(dfpid[x], dfpid[y1], 'o', ms=ms, label=pid)
            ax1.plot(dfpid[x], dfpid[y12], '-', linewidth=lw, color=sciblack)
            ax2.plot(dfpid[x], dfpid[y2], '-', color=sciblack)
            ax3.plot(dfpid[x], dfpid[y3], 'o', ms=ms)
            ax3.plot(dfpid[x], dfpid[y32], '-', linewidth=lw, color=sciblack)

            ax1.legend()
            ax1.set_ylabel(y1lbl)
            ax2.set_ylabel(y2lbl)
            ax3.set_ylabel(y3lbl)
            ax3.set_xlabel(xlbl)

            plt.tight_layout()
            plt.savefig(path_results_exposure_disp_per_pid +
                        '/pid{}_{}_position_velocity_exposure-displacement_by_{}.png'.format(pid, new_cols_str, x))
            plt.show()

    # ---

    # ----------------------------------------------------
    # 3. PLOTS RELATING TO LOCALIZED Z-POSITION

    histogram_exposure_displacement_relative_r = True
    if histogram_exposure_displacement_relative_r:
        # histogram(x: r-position, y: r-velocity)
        x = fit_to_col  # 'rg'
        y = dpoly_fit_dt

        df_ = df[[x, y, 'memb_radius']].dropna()
        xlbl = r'$r / a \: (pix.)$'
        xticks = [0, 0.5, 1]
        ylbl = r'$\vec{u_{r}}^{fit} \: (pix. / s)$'
        save_path = path_results_exposure_disp + '/{}_hist_{}_by_{}.png'.format(new_cols_str, x, y)

        plotting.scatter_and_kde_x_and_y(df_[x] / df_['memb_radius'], df_[y],
                                         binwidth_x=0.1, binwidth_y=0.25,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=0.1, bandwidth_y=0.5,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

        # ---

        # histogram(x: r-position, y: experimentally measured r-exposure displacement)
        x = fit_to_col  # 'rg'
        y = d_exposure

        df_ = df[[x, y, 'memb_radius']].dropna()
        xlbl = r'$r \: (pix.)$'
        xticks = None
        ylbl = r'$\Delta r_{exposure} \: (pix.)$'
        save_path = path_results_exposure_disp + '/{}_hist_{}_by_{}.png'.format(new_cols_str, x, y)

        plotting.scatter_and_kde_x_and_y(df_[x] / df_['memb_radius'], df_[y],
                                         binwidth_x=0.1, binwidth_y=0.025,
                                         kde_x=False, kde_y=True,
                                         bandwidth_x=0.1, bandwidth_y=0.025,
                                         color=None, colormap='coolwarm', scatter_size=0.5,
                                         xlbl=xlbl, ylbl=ylbl,
                                         xticks=xticks,
                                         figsize=(size_x_inches, size_y_inches),
                                         save_path=save_path, show_plot=True,
                                         )

        # ---

    # ---

# ---

print("Analysis completed without errors.")