# test post-processing script for zipper deflection

# imports
from os import listdir
from os.path import join
import os
import time
import re

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf

from skimage.io import imread

import matplotlib.pyplot as plt

from utils import functions
from utils.functions import fErrorFunction, fNonDimensionalNonlinearSphericalUniformLoad
from utils.plotting import lighten_color

# A note on SciencePlots colors
"""
Blue: #0C5DA5
Green: #00B945
Red: #FF2C00
Orange: #FF9500

Other Colors:
Light Blue: #7BC8F6
Paler Blue: #0343DF
Azure: #069AF3
Dark Green: #054907
"""

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)
del ax


# ---

# ----------------------------------------------------------------------------------------------------------------------
# 0. READ CONFIGURATION SETTINGS

def read_configuration_file(img_dir):
    """ dict_config = read_configuration_file(img_dir) """
    # read config and parse into dictionary
    path_config = join(img_dir, 'setup/config_test.xlsx')
    dict_config = pd.read_excel(path_config, index_col=0).to_dict(orient='dict')
    dict_config = dict_config['val']

    # optical setup
    mag_eff = dict_config['mag_eff']
    numerical_aperture = dict_config['numerical_aperture']
    wavelength = dict_config['wavelength']
    n0_medium = dict_config['n0_medium']
    hg_lamp_intensity_percent = dict_config['frame_rate']
    filter_cube = dict_config['frame_rate']
    pixel_size = dict_config['pixel_size']
    microns_per_pixel = dict_config['microns_per_pixel']

    # camera settings
    exposure_time = dict_config['frame_rate']
    frame_rate = dict_config['frame_rate']
    readout_bits = dict_config['frame_rate']
    pre_amp_gain = dict_config['frame_rate']
    electron_multiplying_gain = dict_config['frame_rate']

    # physical setup
    E_silpuran = dict_config['E_silpuran']
    poisson = dict_config['poisson']
    t_membrane = dict_config['t_membrane']

    # derived values
    depth_of_focus = functions.depth_of_field(mag_eff, numerical_aperture, wavelength, n0_medium,
                                              pixel_size=pixel_size * 1e-6) * 1e6

    dict_config.update({'depth_of_focus': depth_of_focus})

    return dict_config


def read_configuration_analysis_file(path_config_analysis):
    """ dict_config_analysis = read_configuration_analysis_file(path_config_analysis) """
    # read config_analysis and parse into dictionary
    dict_config_analysis = pd.read_excel(path_config_analysis, index_col=0).to_dict(orient='dict')
    dict_config_analysis = dict_config_analysis['val']

    return dict_config_analysis


def read_configuration_test_times(path_config_test_times):
    """ df_config_test_times = read_configuration_test_times(path_config_test_times) """

    if path_config_test_times is not None:
        df_config_test_times = pd.read_excel(path_config_test_times)
    else:
        df_config_test_times = None

    return df_config_test_times

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 1. SETUP DIRECTORIES


def setup_dirs(base_dir, fab_dir, wafer_id):
    img_dir = join(base_dir, 'images')
    reference_dir = join(img_dir, 'reference')
    results_dir = join(base_dir, 'results')
    trace_dir = join(results_dir, 'traces')
    save_dir = join(base_dir, 'analysis')
    fig_dir = join(save_dir, 'figures')
    validation_dir = join(save_dir, 'validation')
    stats_dir = join(save_dir, 'stats')
    animation_dir = join(save_dir, 'animations')
    animation_dir_rz = join(animation_dir, 'rz_by_frame')
    animation_dir_rz_figs = join(animation_dir, 'rz_by_frame', 'figs')
    animation_dir_fit_rz_figs = join(animation_dir, 'fit_rz_by_frame', 'figs')
    animation_dir_fit_rz_fill_between_figs = join(animation_dir, 'fit_rz_by_frame_fill_between', 'figs')
    fab_wafer_dir = join(fab_dir, 'Wafer{}'.format(wafer_id))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    if not os.path.exists(animation_dir):
        os.makedirs(animation_dir)

    if not os.path.exists(animation_dir_rz):
        os.makedirs(animation_dir_rz)

    if not os.path.exists(animation_dir_rz_figs):
        os.makedirs(animation_dir_rz_figs)

    if not os.path.exists(animation_dir_fit_rz_figs):
        os.makedirs(animation_dir_fit_rz_figs)

    if not os.path.exists(animation_dir_fit_rz_fill_between_figs):
        os.makedirs(animation_dir_fit_rz_fill_between_figs)

    if not os.path.exists(fab_wafer_dir):
        os.makedirs(fab_wafer_dir)

    dirs = {
        'img_dir': img_dir,
        'reference_dir': reference_dir,
        'results_dir': results_dir,
        'trace_dir': trace_dir,
        'save_dir': save_dir,
        'fig_dir': fig_dir,
        'validation_dir': validation_dir,
        'stats_dir': stats_dir,
        'animation_dir': animation_dir,
        'animation_dir_rz': animation_dir_rz,
        'animation_dir_rz_figs': animation_dir_rz_figs,
        'animation_dir_fit_rz_figs': animation_dir_fit_rz_figs,
        'animation_dir_fit_rz_fill_between_figs': animation_dir_fit_rz_fill_between_figs,
        'fab_wafer_dir': fab_wafer_dir,
    }

    return dirs


# ---

# ----------------------------------------------------------------------------------------------------------------------
# 1. READ BRIGHTFIELD IMAGE

def read_tiff(filepath):
    img = imread(filepath)
    if len(img.shape) > 2:
        avg_idx = np.argmin(img.shape)
        img_dtype = img.dtype
        img = np.mean(img, axis=avg_idx).astype(img_dtype)
    return img


def validate_img_rc(path_img_bf_rc, img_rc, path_save):
    img_bf_rc = read_tiff(path_img_bf_rc)

    fig, ax = plt.subplots()
    ax.imshow(img_bf_rc)
    ax.scatter(img_rc[0], img_rc[1], color='red', label='({}, {})'.format(img_rc[0], img_rc[1]))
    ax.legend()
    plt.tight_layout()
    plt.savefig(join(path_save, 'validate_img_rc.png'))
    plt.show()
    plt.close()


# ---

# ----------------------------------------------------------------------------------------------------------------------
# 3. READ TEST DIRECTORIES

def parse_step(test_dir_name):
    tdir_splits = re.split("'", test_dir_name)

    Vapp_str, n_test_str = tdir_splits[1], tdir_splits[3]

    Vapp = int(Vapp_str[:-1])
    n_test = int(n_test_str[-1])
    period = np.nan
    dwell_time = np.nan

    return Vapp, n_test, period, dwell_time


def parse_ramp(test_dir_name):
    tdir_splits = re.split("'", test_dir_name)[1]

    Vapp_str, period_str, n_test_str = re.split('_', tdir_splits)

    Vapp = int(Vapp_str[:-1])
    period = float(period_str[0])
    n_test = int(n_test_str[-1])
    dwell_time = np.nan

    return Vapp, n_test, period, dwell_time


def parse_triangle(test_dir_name):
    tdir_splits = re.split("'", test_dir_name)[1]

    Vapp_str, dwell_time_str, period_str = re.split('_', tdir_splits)

    Vapp = int(Vapp_str[:-1])
    dwell_time = float(dwell_time_str[:-2]) / 1e3
    period = float(period_str[:-2]) / 1e3
    n_test = 1

    return Vapp, n_test, period, dwell_time


def parse_test_dirs_by_type(test_type, test_dir_name):
    """ Vapp, n_test, period, dwell_time = parse_test_dirs_by_type(test_type, test_dir_name) """

    if test_type == 'step':
        Vapp, n_test, period, dwell_time = parse_step(test_dir_name)
    elif test_type == 'ramp':
        Vapp, n_test, period, dwell_time = parse_ramp(test_dir_name)
    elif test_type == 'triangle':
        Vapp, n_test, period, dwell_time = parse_triangle(test_dir_name)
    else:
        raise ValueError("Must be type type: 'step', 'ramp', 'triangle'. ")

    return Vapp, n_test, period, dwell_time


def parse_trace_endswith(test_type, Vapp, n_test, period, dwell_time):
    """ trace_endswith = parse_trace_endswith(test_type, Vapp, n_test, period, dwell_time) """
    if test_type == 'step':
        trace_endswith = '{}V_n={}.trc'.format(Vapp, n_test)
    elif test_type == 'ramp':
        trace_endswith = '{}V_{}sRamp_n={}.trc'.format(Vapp, int(period), n_test)
    elif test_type == 'triangle':
        trace_endswith = '{}V_{}ms_{}ms.trc'.format(Vapp, int(float(dwell_time) * 1e3), int(float(period) * 1e3))
    else:
        raise ValueError("Must be type type: 'step', 'ramp', 'triangle'. ")

    return trace_endswith


def get_test_time(df_test_times, Vapp, n_test):
    """ date_time = get_test_time(df_test_times, Vapp, n_test) """
    if df_test_times is not None:
        df_date_time = df_test_times[(df_test_times['V'] == np.abs(Vapp)) & (df_test_times['n'] == n_test)]
        if len(df_date_time) > 0:
            date_time = df_date_time['date_time'].iloc[0]
        else:
            date_time = np.nan
    else:
        date_time = np.nan
    return date_time


def get_trace(trace_dir, trace_endswith):
    """ fn_trace = get_trace(trace_dir, trace_endswith) """
    fn_trace = [f for f in os.listdir(trace_dir) if f.endswith(trace_endswith)]
    if len(fn_trace) == 1:
        fn_trace = fn_trace[0]
    elif len(fn_trace) > 1:
        raise ValueError("Multiple traces with identical ID's. ")
    else:
        fn_trace = ''

    return fn_trace


def get_test_type_dirs(results_dir):
    return [f for f in os.listdir(results_dir) if not f.startswith('.')]


def get_test_dirs(results_dir, test_dir_startswith):
    return [f for f in listdir(results_dir) if f.startswith(test_dir_startswith)]


def get_tests(results_dir, test_dir_startswith, trace_dir, df_test_times=None):
    test_type_dirs = get_test_type_dirs(results_dir)

    test_details = []
    for test_type_dir in test_type_dirs:
        test_dirs = get_test_dirs(join(results_dir, test_type_dir), test_dir_startswith)

        for test_dir_name in test_dirs:
            Vapp, n_test, period, dwell_time = parse_test_dirs_by_type(test_type_dir, test_dir_name)

            trace_endswith = parse_trace_endswith(test_type_dir, Vapp, n_test, period, dwell_time)
            fn_trace = get_trace(trace_dir, trace_endswith)

            date_time = get_test_time(df_test_times, Vapp, n_test)

            test_details.append([test_type_dir, test_dir_name, Vapp, n_test, period, dwell_time, fn_trace, date_time])

    test_details = np.array(test_details)

    return test_details


def parse_test_details(test_detail):
    """ uniq_id = parse_test_details(test_detail) """
    ttype, tdir, Vapp, n_test, period, dwell_time, fn_trace, date_time = test_detail

    uniq_id = '{}_{}V_n={}'.format(ttype, Vapp, n_test)

    # period
    if period != 'nan':
        uniq_id += '_Period{}s'.format(period)

    # dwell time
    if dwell_time != 'nan':
        uniq_id += '_Step{}ms'.format(int(float(dwell_time) * 1e3))

    return uniq_id


def get_tests_deprecated(results_dir, test_dir_startswith, df_test_times=None, sort_by_date=False):
    test_dirs = get_test_dirs(results_dir, test_dir_startswith)

    Vapps = []
    n_tests = []
    date_times = []
    for tdir in test_dirs:
        # get Vapp and n_test
        tdir_splits = re.split("'", tdir)

        Vapp_str, n_test_str = tdir_splits[1], tdir_splits[3]

        Vapp = int(Vapp_str[:-1])
        n_test = int(n_test_str[-1])

        # get date
        if df_test_times is not None:
            df_date_time = df_test_times[(df_test_times['V'] == np.abs(Vapp)) & (df_test_times['n'] == n_test)]
            if len(df_date_time) > 0:
                date_time = df_date_time['date_time'].iloc[0]
            else:
                date_time = np.nan
        else:
            date_time = np.nan

        Vapps.append(Vapp)
        n_tests.append(n_test)
        date_times.append(date_time)

    test_details = np.vstack([test_dirs, Vapps, n_tests, date_times]).T

    # sort
    test_details = test_details[test_details[:, 2].argsort()]
    test_details = test_details[test_details[:, 1].argsort(kind='mergesort')]

    if df_test_times is not None and sort_by_date is True:
        test_details = test_details[test_details[:, 3].argsort(kind='mergesort')]

    return test_details


def get_tests_images_creation_time(img_dir, img_filename='test_X1.tif'):
    """ frame, result = get_tests_images_datetime(img_dir, img_filename='test_X1.tif') """

    test_img_dir = join(img_dir, 'tests')
    Vapp_strs = [f for f in os.listdir(test_img_dir) if f[0] != '.']

    list_Vapps = []
    list_n_tests = []
    list_creation_times = []
    list_birth_times = []

    for Vapp_str in Vapp_strs:
        n_test_strs = [f for f in os.listdir(join(test_img_dir, Vapp_str)) if f[0] != '.']

        for n_test_str in n_test_strs:
            file = join(test_img_dir, Vapp_str, n_test_str, img_filename)

            file_stat = os.stat(file)
            birth_time = file_stat.st_birthtime

            creation_time = time.ctime(os.path.getctime(file))

            list_Vapps.append(Vapp_str[:-1])
            list_n_tests.append(n_test_str[-1])
            list_creation_times.append(creation_time)
            list_birth_times.append(birth_time)

    s_Vapps = pd.Series(list_Vapps)
    s_n_tests = pd.Series(list_n_tests)
    s_creation_times = pd.to_datetime(pd.Series(list_creation_times))
    s_birth_times = pd.to_datetime(pd.Series(list_creation_times))

    frame = {'Vapp': s_Vapps,
             'n_test': s_n_tests,
             'creation_time': s_creation_times,
             'birth_time': s_birth_times}

    result = pd.DataFrame(frame).sort_values('creation_time')

    return frame, result


# ---

# ----------------------------------------------------------------------------------------------------------------------
# 4. READ SURFACE PROFILE COORDS


def read_merged_process_profile_coords(path_profile_coords):
    merged_process_profile_coords = pd.read_excel(path_profile_coords)
    return merged_process_profile_coords


def get_profile(merged_process_profile_coords, fid, step):
    return merged_process_profile_coords[(merged_process_profile_coords['fid'] == fid) &
                                         (merged_process_profile_coords['step'] == step)]


def get_target_profile(target_radius, target_depth):
    x = np.linspace(-2, 2, 256)
    px = (x + 2) / 2
    py = erf(x) / 2 - 0.5
    dft = pd.DataFrame(np.vstack([px, py]).T, columns=['r', 'z'])
    dft['r'] = dft['r'] * target_radius / 2
    dft['z'] = dft['z'] * target_depth
    return dft


def get_profile_depth_and_roughness(dfs, dr_center, return_dfs_center=False):
    dfs_center = dfs[(dfs['r'] > -dr_center) & (dfs['r'] < dr_center)]
    z_max = dfs_center['z'].abs().mean()
    z_roughness = dfs_center['z'].std()

    if return_dfs_center:
        return z_max, z_roughness, dfs_center
    else:
        return z_max, z_roughness


def plot_profile_and_surface_roughness(dfs, dr_center, save_id, path_save):
    z_max, z_roughness, dfs_center = get_profile_depth_and_roughness(dfs, dr_center, return_dfs_center=True)

    legend_title = r'$z_{max} \pm z_{roughness} \: (\mu m)$'
    lbl = '{} '.format(np.round(z_max, 1)) + r'$\pm$' + '{}'.format(np.round(z_roughness, 1))

    fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches * 0.75))
    ax.plot(dfs.r * 1e-3, dfs.z, label=lbl)
    ax.set_xlabel(r'$r \: (mm)$')
    ax.set_ylabel(r'$z \: (\mu m)$')
    ax.grid(alpha=0.25)
    ax.legend(title=legend_title)
    plt.tight_layout()
    plt.savefig(join(path_save, '{}_surface_profile.png'.format(save_id)))
    plt.close()

    fig, ax = plt.subplots(figsize=(size_x_inches * 0.65, size_y_inches * 0.65))
    ax.plot(dfs_center.r, dfs_center.z, label=lbl)
    ax.set_xlabel(r'$r \: (\mu m)$')
    ax.set_ylabel(r'$z \: (\mu m)$')
    ax.grid(alpha=0.25)
    # ax.legend(title=legend_title)
    # ax.set_title('Note: smoothed profile', fontsize=8)
    plt.tight_layout()
    plt.savefig(join(path_save, '{}_surface_roughness.png'.format(save_id)))
    plt.close()


# ---

# ----------------------------------------------------------------------------------------------------------------------
# 5.a READ TEST COORDS


def read_coords(path_coords, cols_include, padding, cols_xy, img_rc, cols_r, frame_rate):
    coords = pd.read_excel(path_coords)
    coords = coords[cols_include]

    if padding:
        for col_xy in cols_xy:
            coords[col_xy] = coords[col_xy] - padding

    if img_rc:
        for i, col_r in enumerate(cols_r):
            coords[col_r] = functions.calculate_radius_at_xy(coords[cols_xy[i * 2]], coords[cols_xy[i * 2 + 1]],
                                                             xc=img_rc[0], yc=img_rc[1])

    if frame_rate:
        coords['t'] = coords['frame'] / frame_rate

    return coords


def validate_img_rc_and_pids(path_imgs, img_rc, xi, yi, coords, in_frame, save_id, path_save):
    if coords is not None:

        if in_frame is None:
            in_frame = coords['frame'].min()

        baseline_coords = coords[coords['frame'] == in_frame]
        baseline_xy = baseline_coords[['x', 'y']].to_numpy()

        xi = baseline_xy[:, 0]
        yi = baseline_xy[:, 1]

    fig, axs = plt.subplots(ncols=len(path_imgs), figsize=(9, 3.5))

    for ax, path_img in zip(axs, path_imgs):
        img = read_tiff(path_img)

        ax.imshow(img)
        ax.scatter(img_rc[0], img_rc[1], s=3, color='red', label='({}, {})'.format(img_rc[0], img_rc[1]))
        ax.scatter(xi, yi, s=5, color='blue', alpha=0.5)
        ax.legend(loc='upper right', labelcolor='white')

    plt.tight_layout()
    plt.savefig(join(path_save, '{}_validate_img_rc_and_pids_frame{}.png'.format(save_id, in_frame)))
    plt.close()


def validate_baseline_z0(baseline_coords, z0, px, raw_y, corr_y, save_id, path_save):
    z_mean_raw = baseline_coords[raw_y].mean()
    z_mean_corr = baseline_coords[corr_y].mean()

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    # 1. plot particles "raw" positions: z is arbitrary according to calibration stack
    ax1.scatter(baseline_coords[px], baseline_coords[raw_y], s=3, marker='o', color='blue')
    ax1.axhline(z_mean_raw, color='gray', linestyle='--', label='mean({})={}'.format(raw_y, np.round(z_mean_raw, 2)))

    if z0 is not None:
        ax1.axhline(z0, color='k', linestyle='--', label='z0')

    ax1.set_ylabel('{} '.format(raw_y) + r'$(\mu m)$')
    ax1.legend()

    # -

    # 2. plot particles "corrected" positions: z is defined by average z in baseline image
    ax2.scatter(baseline_coords[px], baseline_coords[corr_y], s=3, marker='o', color='blue')
    ax2.axhline(z_mean_corr, color='gray', linestyle='--', label='mean({})={}'.format(corr_y, np.round(z_mean_corr, 2)))

    if z0 is not None:
        ax2.axhline(z0, color='k', linestyle='--', label='z0')

    ax2.set_xlabel(r'$r \: (\mu m)$')
    ax2.set_ylabel('{} '.format(corr_y) + r'$(\mu m)$')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(join(path_save, '{}_validate_baseline_z0.png'.format(save_id)))
    plt.show()
    plt.close()


def transform_coords(coords, z0, flip_z, microns_per_pixel, z0_from_frame):
    if z0 is None and z0_from_frame is not None:

        if isinstance(z0_from_frame, int):
            z0 = coords[coords['frame'] == z0_from_frame]['z'].mean()
        else:
            z0 = coords[(coords['frame'] >= z0_from_frame[0]) & (coords['frame'] <= z0_from_frame[1])]['z'].mean()

    coords['z_corr'] = coords['z'] - z0
    coords['r_microns'] = coords['r'] * microns_per_pixel

    if flip_z:
        coords['z_corr'] = coords['z_corr'] * -1

    if z0_from_frame is not None:
        frame_coords = coords[coords['frame'] == z0_from_frame]
        return coords, frame_coords, z0
    else:
        return coords


def remove_focal_plane_bias_errors(coords, in_frames):
    coords = coords[(coords['frame'] >= in_frames[0]) & (coords['frame'] <= in_frames[1])]
    diff_frames = in_frames[1] - in_frames[0] + 1

    pids = coords['id'].unique()
    exclude_ids = []
    mean_diff_z_larges = []

    for pid in pids:
        dfpid = coords[coords['id'] == pid]

        diff_z = dfpid['z'].diff()
        diff_z_large = diff_z[diff_z > 4]
        mean_diff_z_large = diff_z_large.mean()
        count_diff_z_large = np.max([len(diff_z_large), 1])

        # square_diff_z = dfpid['z'].diff() ** 2

        # std_square_diff_z = square_diff_z.std()

        if count_diff_z_large > diff_frames / 5:
            exclude_ids.append(pid)
            mean_diff_z_larges.append(mean_diff_z_large)

            """fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False)

            ax1.plot(dfpid['frame'], dfpid['z'], '-o', label=pid)
            ax1.set_ylabel('z')
            ax1.legend(title='pID')
            ax1.set_title('std diff z = {}'.format(np.round(std_square_diff_z, 2)))

            ax2.plot(square_diff_z, '-o')
            ax2.set_ylabel('square diff z')
            ax2.set_xlabel('close to frame')

            plt.tight_layout()
            plt.show()
            plt.close()"""

    mean_mean_diff_z_larges = np.mean(mean_diff_z_larges)
    std_mean_diff_z_larges = np.std(mean_diff_z_larges)

    coords = coords[~coords['id'].isin(exclude_ids)]

    return coords, mean_mean_diff_z_larges, std_mean_diff_z_larges


# ---
# 5.b READ TRACE AND PROCESS DATA
def read_trace(fp_trace):
    """ df = read_trace(fp_trace) """
    df = pd.read_csv(fp_trace, sep='\t', lineterminator='\n', engine='c', keep_default_na=False)
    df = df.rename(columns={df.columns[-1]: df.columns[-1][:-1]})

    return df


def align_trace_and_coords(df_trace, coords, dfdt=0.205):
    """ df_trace = align_trace_and_coords(df_trace, coords=None, dfdt=0.205) """
    df_trace = df_trace[df_trace['Step number'].isin([1, 2, 3])]

    if coords is not None:
        # cross-correlate signals to align them?
        pass
    else:
        df_trace['t_rel'] = df_trace['time (s)'] - df_trace['time (s)'].iloc[0] + dfdt

    return df_trace

# ---


def plot_coords_and_trace_by_time(coords, df_trace, save_dir, save_id):
    # modify particle tracking
    dftg = coords.groupby('frame').mean()

    px = 't_rel'
    py = 'Ch. A Voltage (V)'

    px2 = 't'
    py2a = 'z_corr'
    py2b = 'z'

    fig, (ax, axr) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.1, size_y_inches * 1.0))

    ax.plot(df_trace[px], df_trace[py], '-o', ms=4, color='k', label='HVS')

    axr.plot(dftg[px2], -dftg[py2a], '-o', ms=4, color='tab:red', label='IDPT')

    axrr = axr.twinx()
    axrr.plot(dftg[px2], dftg[py2b], '-o', ms=4, color='tab:red', alpha=0.0)

    ax.set_ylabel(r'$V_{app} \: (V)$')
    ax.grid(alpha=0.35)
    ax.legend(loc='upper left')

    axr.set_xticks(np.arange(0, dftg[px2].max() + 0.35, 1))
    axr.set_xlabel(r'$t \: (s)$')
    axr.set_ylabel(r'$| \Delta z | \: (\mu m)$')
    axrr.set_ylabel(r'$z_{raw} \: (\mu m)$')
    axr.legend(loc='upper left')
    axr.grid(alpha=0.35)

    plt.tight_layout()
    plt.savefig(join(save_dir, save_id + '.png'))
    plt.show()


# ---

def analyze_single_frame(fcoords, df_surface_profile, frame_of_interest,
                         px, py, px_lims, py_lims, dz_min_max,
                         profile_radius, dr_center, t_membrane,
                         fit_error_function, plate_model, lva, save_id,
                         plot_per_frame_membrane_radial_profile, plot_per_frame_membrane_radial_profile_fill_between,
                         animation_dir_fit_rz_figs, animation_dir_fit_rz_fill_between_figs):
    """ fresults = [num_pids_dr_center, dz_mean_dr_center, dz_std_dr_center, dz_max_erf, A, rmse_erf, r_squared_erf] """

    # initialize some variables
    fit_erf_r, fit_erf_z, A, rmse_erf, r_squared_erf = None, None, None, None, None
    fit_d_r, fit_d_z = None, None

    # --- GET COORDS
    ftime = fcoords.iloc[0]['t']
    r = fcoords[px].to_numpy()
    z = fcoords[py].to_numpy()

    # --- MODEL SURFACE
    if fit_error_function:
        fERF = fErrorFunction(profile_radius=profile_radius)
        fit_erf_r, fit_erf_z, A, rmse_erf, r_squared_erf = fERF.fit(r, z)

    if plate_model == 'nonlinear':
        pass
        """
        # fit plate nonlinear plate theory
        a_metric_units = profile_radius * 1e-6
        fND_lr = fNonDimensionalNonlinearSphericalUniformLoad(r=a_metric_units,
                                                              h=t_membrane,
                                                              youngs_modulus=E_silpuran,
                                                              poisson=poisson)
        # assign fit function
        # fitfunc_lr = fND_lr

        # data to fit on
        r_metric_units = r * 1e-6
        z_metric_units = z * 1e-6
        nd_r = r_metric_units / a_metric_units
        nd_z = z_metric_units / t_membrane

        # data to evaluate rmse
        r_ev = r_metric_units.copy()
        z_ev = z_metric_units.copy()
        nd_r_ev = nd_r.copy()
        nd_z_ev = nd_z.copy()

        # guess and bounds
        if np.max(nd_z) > 2:
            nd_k_lower_bound = 0.01  # 1
            nd_k_guess = 5
            nd_p_guess = 100
        else:
            nd_k_lower_bound = 0.01
            nd_k_guess = 0.5
            nd_p_guess = -1

        fit_d_r, fit_d_z, d_p0, d_n0, rmse, r_squared = fND_lr.fit_nd_nonlinear(lva,
                                                                                nd_r,
                                                                                nd_z,
                                                                                nd_r_eval=nd_r_ev,
                                                                                nd_z_eval=nd_z_ev,
                                                                                guess=(
                                                                                    nd_p_guess,
                                                                                    nd_k_guess),
                                                                                bounds=(
                                                                                    [-1e9,
                                                                                     nd_k_lower_bound],
                                                                                    [1e9, 1e9])
                                                                                )

        # scale
        fit_d_r = fit_d_r * 1e6
        fit_d_z = fit_d_z * 1e6
        """

    # --- CALCULATE MAX DEFLECTION
    fcoords_dr_center = fcoords[fcoords[px] < dr_center]
    if len(fcoords_dr_center) > 0:
        num_pids_dr_center = len(fcoords_dr_center)
        dz_mean_dr_center = fcoords_dr_center[py].mean()

        if len(fcoords_dr_center) > 2:
            dz_std_dr_center = fcoords_dr_center[py].mean()
        else:
            dz_std_dr_center = np.nan
    else:
        num_pids_dr_center = 0
        dz_mean_dr_center = np.nan
        dz_std_dr_center = np.nan

    if fit_error_function:
        dz_max_erf = np.min(fit_erf_z)
    else:
        dz_max_erf = np.nan

    # --- PLOT

    if plot_per_frame_membrane_radial_profile:

        fig, ax = plt.subplots()

        # plot particles
        ax.scatter(r, z, s=10, c=z, marker='o', cmap='RdBu_r', vmin=-2.5, vmax=2.5, label='Particles')
        # ax.scatter(r, z, s=10, marker='o', color='blue', label='Particles')

        # plot membrane: upper and lower surfaces
        if fit_d_r is not None:
            ax.plot(fit_d_r, fit_d_z, '-', color='green', label='Fit: von Karman')

        # plot error function:
        if fit_erf_r is not None:
            ax.plot(fit_erf_r, fit_erf_z, '-', color='black', label='Fit: erf')

        # plot surface profile
        ax.plot(df_surface_profile.r, df_surface_profile.z, '-', color='gray', label='Surface Profile')

        ax.set_xlabel(r'$r \: (\mu m)$')
        ax.set_xlim([-25, profile_radius + 100])
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.set_ylim([py_lims[0], py_lims[1] + 2.5])
        ax.legend(loc='lower right')
        ax.set_title('Frame: {}, time: {} s'.format(frame_of_interest, np.round(ftime, 3)))
        plt.tight_layout()
        plt.savefig(join(animation_dir_fit_rz_figs, '{}_rz_by_frame{}.png'.format(save_id, frame_of_interest)), dpi=100)
        plt.close()

    if plot_per_frame_membrane_radial_profile_fill_between:

        fig, ax = plt.subplots()

        # plot particles
        ax.plot(r, z + t_membrane * 1e6, 'o', ms=5, color='k', label='Particles')

        # plot membrane: upper and lower surfaces
        if fit_d_r is not None:
            ax.fill_between(fit_d_r, fit_d_z + t_membrane * 1e6, fit_d_z,
                            color='green', alpha=0.25, label='Fit: von Karman')

        # plot error function:
        if fit_erf_r is not None:
            ax.fill_between(fit_erf_r, fit_erf_z + t_membrane * 1e6, fit_erf_z,
                            color='blue', alpha=0.25, label='Fit: erf')

        # plot surface profile
        ax.plot(df_surface_profile.r, df_surface_profile.z, '-', color='gray', label='Surface Profile')

        ax.set_xlabel(r'$r \: (\mu m)$')
        ax.set_xlim([-25, profile_radius + 50])
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.set_ylim([py_lims[0], py_lims[1] + t_membrane * 1e6])
        ax.legend(loc='lower right')
        ax.set_title('Frame: {}, time: {} s'.format(frame_of_interest, np.round(ftime, 3)))
        plt.tight_layout()
        plt.savefig(
            join(animation_dir_fit_rz_fill_between_figs, '{}_rz_by_frame{}.png'.format(save_id, frame_of_interest)),
            dpi=100)
        plt.close()

    # --- OUTPUT
    fresults = [num_pids_dr_center, dz_mean_dr_center, dz_std_dr_center, dz_max_erf, A, rmse_erf, r_squared_erf]

    return fresults


def identify_baseline_z0_in_tests(test_details, dict_test):
    results_dir = dict_test['results_dir']
    save_dir = dict_test['save_dir']
    stats_dir = dict_test['stats_dir']
    validation_dir = dict_test['validation_dir']
    animation_dir = dict_test['animation_dir']
    animation_dir_rz = dict_test['animation_dir_rz']
    animation_dir_rz_figs = dict_test['animation_dir_rz_figs']
    animation_dir_fit_rz_figs = dict_test['animation_dir_fit_rz_figs']
    animation_dir_fit_rz_fill_between_figs = dict_test['animation_dir_fit_rz_fill_between_figs']
    fn_test_coords = dict_test['fn_test_coords']
    cols_include = dict_test['cols_include']
    idpt_padding = dict_test['idpt_padding']
    cols_xy = dict_test['cols_xy']
    img_rc = dict_test['img_rc']
    cols_r = dict_test['cols_r']
    frame_rate = dict_test['frame_rate']
    microns_per_pixel = dict_test['microns_per_pixel']
    t_membrane = dict_test['t_membrane']
    profile_radius = dict_test['profile_radius']  # 2000 microns
    fr_step = dict_test['frame_step']
    baseline_frame = dict_test['baseline_frame']
    dr_center = dict_test['dr_center']

    analysis_save_id = dict_test['save_id']

    # define x and y cols: "raw" = rx, ry; "plotting" = px, py
    rx, ry = 'r', 'z'
    px, py = 'r_microns', 'z_corr'

    # --------------

    Vapps, z0s, dts = [], [], []
    for i, tdetail in enumerate(test_details):
        tdir, Vapp, n_test, date_time = tdetail[0], tdetail[1], tdetail[2], tdetail[3]
        Vapp = int(Vapp)
        n_test = int(n_test)

        # read test coords
        path_coords = join(results_dir, tdir, fn_test_coords)
        coords = read_coords(path_coords, cols_include, idpt_padding, cols_xy, img_rc, cols_r, frame_rate)

        # get frames to analyze
        frames_of_interest = coords['frame'].sort_values().unique()
        fr_i, fr_f = frames_of_interest[0], frames_of_interest[-1]

        coords_i = coords[coords['frame'] < 11]

        z0s.append(coords_i['z'].mean())
        Vapps.append(Vapp)
        dts.append(date_time)

    import matplotlib.dates as mdates
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False)
    ax1.scatter(dts, z0s)
    ax2.scatter(dts, Vapps)
    plt.tight_layout()
    plt.show()

    # --------------

    analysis_results = []
    z0s = []
    z0rs = []
    mean_mean_diff_z_largess = []
    Vapps = []
    for i, tdetail in enumerate(test_details):
        tdir, Vapp, n_test, date_time = tdetail[0], tdetail[1], tdetail[2], tdetail[3]
        Vapp = int(Vapp)
        n_test = int(n_test)
        save_id = analysis_save_id + '_{}V_n={}_'.format(Vapp, n_test)

        # read test coords
        path_coords = join(results_dir, tdir, fn_test_coords)
        coords = read_coords(path_coords, cols_include, idpt_padding, cols_xy, img_rc, cols_r, frame_rate)

        # get frames to analyze
        frames_of_interest = coords['frame'].sort_values().unique()
        fr_i, fr_f = frames_of_interest[0], frames_of_interest[-1]
        frames_of_interest = frames_of_interest[fr_i:fr_f:fr_step]

        coords_filter_fpb, mean_mean_diff_z_larges, std_mean_diff_z_larges = remove_focal_plane_bias_errors(coords,
                                                                                                            in_frames=(
                                                                                                            fr_i, 10))
        if not np.isnan(mean_mean_diff_z_larges):
            mean_mean_diff_z_largess.append(mean_mean_diff_z_larges)
            print("Mean, std diff z large = {} +/- {}".format(np.round(mean_mean_diff_z_larges, 2),
                                                              np.round(std_mean_diff_z_larges, 2)))
        else:
            mean_mean_diff_z_largess.append(np.nan)

        # get baseline frame
        if baseline_frame is None:
            baseline_frame = fr_i

        # transform coords to appropriate coordinate system: return baseline coords and z0
        coords_filter_fpb, baseline_coords, z0 = transform_coords(coords_filter_fpb, z0=None, flip_z=True,
                                                                  microns_per_pixel=microns_per_pixel,
                                                                  z0_from_frame=(fr_i, 10))

        # transform coords to appropriate coordinate system: return baseline coords and z0
        coords_filter_fpbr, baseline_coordsr, z0r = transform_coords(coords, z0=None, flip_z=True,
                                                                     microns_per_pixel=microns_per_pixel,
                                                                     z0_from_frame=baseline_frame)

        z0s.append(z0)
        z0rs.append(z0r)
        Vapps.append(Vapp)

    # plot z0s from each test to identify groupings
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(Vapps, z0s, 'o', ms=4, color='k')
    ax1.plot(Vapps, z0rs, 'o', ms=2, color='r')
    ax1.set_ylabel('z0')

    ax2.plot(Vapps, mean_mean_diff_z_largess, 'o', ms=4, color='k')
    ax2.set_xlabel('Vapp')
    ax2.set_ylabel('Mean diff z')

    plt.tight_layout()
    plt.show()


def iterate_through_tests(test_details, dict_test, df_surface_profile):
    results_dir = dict_test['results_dir']
    trace_dir = dict_test['trace_dir']
    save_dir = dict_test['save_dir']
    fig_dir = dict_test['fig_dir']
    stats_dir = dict_test['stats_dir']
    validation_dir = dict_test['validation_dir']
    animation_dir = dict_test['animation_dir']
    animation_dir_rz = dict_test['animation_dir_rz']
    animation_dir_rz_figs = dict_test['animation_dir_rz_figs']
    animation_dir_fit_rz_figs = dict_test['animation_dir_fit_rz_figs']
    animation_dir_fit_rz_fill_between_figs = dict_test['animation_dir_fit_rz_fill_between_figs']
    fn_test_coords = dict_test['fn_test_coords']
    cols_include = dict_test['cols_include']
    idpt_padding = dict_test['idpt_padding']
    cols_xy = dict_test['cols_xy']
    img_rc = dict_test['img_rc']
    cols_r = dict_test['cols_r']
    frame_rate = dict_test['frame_rate']
    microns_per_pixel = dict_test['microns_per_pixel']
    t_membrane = dict_test['t_membrane']
    profile_radius = dict_test['profile_radius']  # 2000 microns
    fr_step = dict_test['frame_step']
    baseline_frame = dict_test['baseline_frame']
    dr_center = dict_test['dr_center']

    analysis_save_id = dict_test['save_id']

    # define x and y cols: "raw" = rx, ry; "plotting" = px, py
    rx, ry = 'r', 'z'
    px, py = 'r_microns', 'z_corr'

    analysis_results = []
    z0s = []
    for i, tdetail in enumerate(test_details):
        ttype, tdir, Vapp, n_test, period, dwell_time, fn_trace, date_time = tdetail
        uniq_id = parse_test_details(tdetail)
        save_id = analysis_save_id + '_' + uniq_id

        Vapp = int(Vapp)
        n_test = int(n_test)
        period = int(float(period))
        dwell_time = float(dwell_time)

        # read test coords
        path_coords = join(results_dir, ttype, tdir, fn_test_coords)
        coords = read_coords(path_coords, cols_include, idpt_padding, cols_xy, img_rc, cols_r, frame_rate)

        # get frames to analyze
        frames_of_interest = coords['frame'].sort_values().unique()
        fr_i, fr_f = frames_of_interest[0], frames_of_interest[-1]
        frames_of_interest = frames_of_interest[fr_i:fr_f:fr_step]

        # -----

        # raise ValueError("Need to improve focal plane bias error removal function. ")
        coords_filter_fpb, mean_mean_diff_z_larges, std_mean_diff_z_larges = remove_focal_plane_bias_errors(coords,
                                                                                                            in_frames=(
                                                                                                            fr_i, 10))

        z0 = coords[(coords['frame'] >= fr_i) & (coords['frame'] <= 10)]['z'].mean()
        baseline_coords = coords_filter_fpb.groupby('id').mean().reset_index()

        # ------

        # get baseline frame
        if baseline_frame is None:
            baseline_frame = fr_i

        # validate particle coordinates on first test only
        if i == -1:
            validate_img_rc_and_pids(PATH_IMG_RCS, IMG_RC, None, None, coords, in_frame=baseline_frame,
                                     save_id=save_id, path_save=validation_dir)

        # transform coords to appropriate coordinate system: return baseline coords and z0
        coords = transform_coords(coords, z0=z0, flip_z=True,
                                  microns_per_pixel=microns_per_pixel,
                                  z0_from_frame=None)

        baseline_coords = transform_coords(baseline_coords, z0=z0, flip_z=True,
                                  microns_per_pixel=microns_per_pixel,
                                  z0_from_frame=None)

        # ---
        # incorporate TRACE data
        plot_traces = False
        if plot_traces and len(fn_trace) > 3:
            df_trace = read_trace(join(trace_dir, fn_trace))
            df_trace = align_trace_and_coords(df_trace, coords=None, dfdt=0.205)
            plot_coords_and_trace_by_time(coords, df_trace, save_dir=fig_dir, save_id=save_id)

        # ---


        # validate z0 for first test only or if there is a relatively large change in z0_raw
        if i == 0:
            pass
            # validate_baseline_z0(baseline_coords, z0, px, ry, py, save_id, path_save=validation_dir)
            # z0s.append(z0)
        elif np.abs(np.mean(z0s) - z0) > 4:
            pass
            # print(z0s)
            # print(z0)
            # validate_baseline_z0(baseline_coords, z0, px, ry, py, save_id, path_save=validation_dir)

        # ---

        # df_surface_profile
        px_min, px_max, px_spacing = 25, profile_radius, 50
        py_min = df_surface_profile['z'].min()
        py_max = np.max([coords[py].max(), df_surface_profile['z'].max()])
        py_spacing = 5
        dz_min_max = (coords[py].min(), coords[py].max())

        px_lims = [px_min - px_spacing, px_max + px_spacing]
        py_lims = [py_min - py_spacing, py_max + py_spacing]

        # ---
        fit_error_function = True
        plate_model, lva = False, 'no'  # 'nonlinear', 'no'
        if Vapp == 2500 and n_test == 2 and period == 17:  # i == 0:
            plot_per_frame_membrane_radial_profile = True
            plot_per_frame_membrane_radial_profile_fill_between = False
        else:
            plot_per_frame_membrane_radial_profile = False
            plot_per_frame_membrane_radial_profile_fill_between = False

        tresults = []
        for frame_of_interest in frames_of_interest:
            fcoords = coords[coords['frame'] == frame_of_interest][[px, py, 't']]

            if Vapp == 2500 and n_test == 2 and period == 7 and frame_of_interest % 5 == 0:  # i == 0:
                plot_per_frame_membrane_radial_profile = True
            else:
                plot_per_frame_membrane_radial_profile = False

            fresults = analyze_single_frame(fcoords, df_surface_profile, frame_of_interest,
                                            px, py, px_lims, py_lims, dz_min_max,
                                            profile_radius, dr_center, t_membrane,
                                            fit_error_function,
                                            plate_model, lva,
                                            save_id,
                                            plot_per_frame_membrane_radial_profile,
                                            plot_per_frame_membrane_radial_profile_fill_between,
                                            animation_dir_fit_rz_figs, animation_dir_fit_rz_fill_between_figs)
            tresults.append(fresults)

        res = pd.DataFrame(np.array(tresults),
                           columns=['num_pids_dr_center', 'dz_mean_dr_center', 'dz_std_dr_center',
                                    'dz_max_erf', 'A_erf', 'rmse_erf', 'r_squared_erf'])

        res.to_excel(join(stats_dir, '{}_results.xlsx'.format(save_id)))

        dz_quantile = 0.01
        percentile_dz_mean_dr_center = res['dz_mean_dr_center'].quantile(dz_quantile)
        percentile_dz_max_erf = res['dz_max_erf'].quantile(dz_quantile)
        tres = [int(Vapp), int(n_test), percentile_dz_mean_dr_center, percentile_dz_max_erf]
        analysis_results.append(tres)

    res = pd.DataFrame(np.array(analysis_results),
                       columns=['Vapp', 'n_test', 'percentile_dz_mean_dr_center', 'percentile_dz_max_erf'])

    res.to_excel(join(save_dir, '{}_results.xlsx'.format(analysis_save_id)))

    # ---

    ms = 3

    fig, ax = plt.subplots()
    ax.plot(res['Vapp'], res['percentile_dz_mean_dr_center'].abs(), 'o', ms=ms, label='Particles')
    ax.plot(res['Vapp'], res['percentile_dz_max_erf'].abs(), '^', ms=ms, label='Fit: erf')
    ax.set_xlabel(r'$V_{applied}$')
    ax.set_ylabel(r'$|\Delta z_{max}| \: (\mu m)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(join(save_dir, '{}_dz_by_Vapp.png'.format(analysis_save_id)))
    plt.show()

    fig, ax = plt.subplots()
    rp = res[res['Vapp'] >= 0]
    rn = res[res['Vapp'] < 0]

    ax.plot(rp['Vapp'], rp['percentile_dz_mean_dr_center'].abs(), 'o', ms=ms, color='red', label='FP(+V)')
    ax.plot(rp['Vapp'], rp['percentile_dz_max_erf'].abs(), '^', ms=ms, color='black', label='Fit(+V))')

    ax.plot(rn['Vapp'].abs(), rn['percentile_dz_mean_dr_center'].abs(), 'o', ms=ms, color='blue', label='FP(-V)')
    ax.plot(rn['Vapp'].abs(), rn['percentile_dz_max_erf'].abs(), '^', ms=ms, color='gray', label='Fit(-V)')

    ax.set_xlabel(r'$|V_{applied}|$')
    ax.set_ylabel(r'$|\Delta z_{max}| \: (\mu m)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(join(save_dir, '{}_dz_by_Vapp-abs.png'.format(analysis_save_id)))
    plt.show()

    # ---

    rm = res.groupby('Vapp').mean().reset_index()
    rstd = res.groupby('Vapp').std().reset_index()

    fig, ax = plt.subplots()
    ax.errorbar(rm['Vapp'], rm['percentile_dz_mean_dr_center'].abs(), yerr=rstd['percentile_dz_mean_dr_center'],
                fmt='o', elinewidth=1, capsize=2, ms=ms, label='Particles')
    ax.errorbar(rm['Vapp'], rm['percentile_dz_max_erf'].abs(), yerr=rstd['percentile_dz_mean_dr_center'],
                fmt='^', elinewidth=1, capsize=2, ms=ms, label='Fit: erf')
    ax.set_xlabel(r'$V_{applied}$')
    ax.set_ylabel(r'$|\Delta z_{max}| \: (\mu m)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(join(save_dir, '{}_dz_by_Vapp_errorbars.png'.format(analysis_save_id)))
    plt.show()

    fig, ax = plt.subplots()
    rpm = rm[rm['Vapp'] >= 0]
    rnm = rm[rm['Vapp'] < 0]
    rpstd = rstd[rstd['Vapp'] >= 0]
    rnstd = rstd[rstd['Vapp'] < 0]

    ax.errorbar(rpm['Vapp'], rpm['percentile_dz_mean_dr_center'].abs(), yerr=rpstd['percentile_dz_mean_dr_center'],
                fmt='o', elinewidth=1, capsize=2, ms=ms, color='red', label='FP(+V)')
    ax.errorbar(rpm['Vapp'], rpm['percentile_dz_max_erf'].abs(), yerr=rpstd['percentile_dz_mean_dr_center'],
                fmt='o', elinewidth=1, capsize=2, ms=ms, color='black', label='Fit(+V))')

    ax.errorbar(rnm['Vapp'].abs(), rnm['percentile_dz_mean_dr_center'].abs(),
                yerr=rnstd['percentile_dz_mean_dr_center'],
                fmt='^', elinewidth=1, capsize=2, ms=ms, color='blue', label='FP(-V)')
    ax.errorbar(rnm['Vapp'].abs(), rnm['percentile_dz_max_erf'].abs(), yerr=rnstd['percentile_dz_mean_dr_center'],
                fmt='^', elinewidth=1, capsize=2, ms=ms, color='gray', label='Fit(-V)')

    ax.set_xlabel(r'$|V_{applied}|$')
    ax.set_ylabel(r'$|\Delta z_{max}| \: (\mu m)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(join(save_dir, '{}_dz_by_Vapp-abs_errorbars.png'.format(analysis_save_id)))
    plt.show()

    # ---

    fig, ax = plt.subplots()

    ax.errorbar(rpm['Vapp'], rpm['percentile_dz_mean_dr_center'].abs(), yerr=rpstd['percentile_dz_mean_dr_center'],
                fmt='o', elinewidth=1, capsize=2, ms=ms + 1, color='red', label='+V')
    ax.errorbar(rnm['Vapp'].abs(), rnm['percentile_dz_mean_dr_center'].abs(),
                yerr=rnstd['percentile_dz_mean_dr_center'],
                fmt='^', elinewidth=1, capsize=2, ms=ms + 1, color='blue', label='-V')

    ax.set_xlabel(r'$|V_{applied}|$')
    ax.set_ylabel(r'$|\Delta z_{max}| \: (\mu m)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(join(save_dir, '{}_dz_by_Vapp-abs_errorbars_FPs-only.png'.format(analysis_save_id)))
    plt.show()


# ---

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # ---

    # INPUT PATH HERE AND RUN
    BASE_PATH = '/Users/mackenzie/Desktop/Zipper/Testing/Wafer18/w18_c1_test3'
    PATH_CONFIG_ANALYSIS = join(BASE_PATH, 'images/setup/config_analysis.xlsx')
    PATH_CONFIG_TEST_TIMES = None  # join(BASE_PATH, 'images/setup/config_test-times.xlsx')

    # ---

    # ---

    CONFIG_ANALYSIS = read_configuration_analysis_file(PATH_CONFIG_ANALYSIS)

    # 1. DEFINE TEST_ID AND DIRECTORIES
    WAFER_ID = CONFIG_ANALYSIS['wafer_id']
    FEATURE_ID = CONFIG_ANALYSIS['feature_id']
    FEATURE_LABEL = CONFIG_ANALYSIS['feature_label']
    TEST_ID = CONFIG_ANALYSIS['test_id']
    PROCESS_STEP = CONFIG_ANALYSIS['process_step']

    # 2. BRIGHTFIELD IMAGES THAT SHOW R=0
    IMG_XC = CONFIG_ANALYSIS['img_xc']
    IMG_YC = CONFIG_ANALYSIS['img_yc']  # Coordinates of center (r = 0) measured in FIJI
    fn_img_bf_rc = CONFIG_ANALYSIS['fn_img_bf_rc']  # image showing orifice (r = 0)
    fn_img_bf_focus_rc = CONFIG_ANALYSIS['fn_img_bf_focus_rc']  # image showing FP's in brightfield (OPTIONAL)
    fn_img_fluoro_rc = CONFIG_ANALYSIS['fn_img_fluoro_rc']  # image showing FP's

    # 3. SURFACE PROFILE
    PROFILE_RADIUS = CONFIG_ANALYSIS['profile_radius']  # microns
    DR_CENTER = CONFIG_ANALYSIS['dr_center']  # radial distance to compute surface roughness over

    # 4. IDPT SETTINGS AND TEST_COORDS
    IDPT_PADDING = CONFIG_ANALYSIS['idpt_padding']
    BASELINE_FRAME = CONFIG_ANALYSIS['baseline_frame']

    # ---

    # ----------------------------------------------------------------------------------------------------------------------
    # 0. SETUP DIRECTORIES

    TEST_ID = 'w{}_{}_test{}'.format(WAFER_ID, FEATURE_LABEL, TEST_ID)
    BASE_DIR = '/Users/mackenzie/Desktop/Zipper/Testing/Wafer{}/{}'.format(WAFER_ID, TEST_ID)
    FAB_DIR = '/Users/mackenzie/Desktop/Zipper/Fabrication'

    DIRS = setup_dirs(base_dir=BASE_DIR, fab_dir=FAB_DIR, wafer_id=WAFER_ID)
    IMG_DIR = DIRS['img_dir']
    REFERENCE_DIR = DIRS['reference_dir']
    RESULTS_DIR = DIRS['results_dir']
    TRACE_DIR = DIRS['trace_dir']
    SAVE_DIR = DIRS['save_dir']
    FIG_DIR = DIRS['fig_dir']
    VALIDATION_DIR = DIRS['validation_dir']
    STATS_DIR = DIRS['stats_dir']
    ANIMATION_DIR = DIRS['animation_dir']
    ANIMATION_DIR_RZ = DIRS['animation_dir_rz']
    ANIMATION_DIR_RZ_FIGS = DIRS['animation_dir_rz_figs']
    ANIMATION_DIR_FIT_RZ_FIGS = DIRS['animation_dir_fit_rz_figs']
    ANIMATION_DIR_FIT_RZ_FILL_BETWEEN_FIGS = DIRS['animation_dir_fit_rz_fill_between_figs']
    WAFER_FAB_DIR = DIRS['fab_wafer_dir']

    # ----------------------------------------------------------------------------------------------------------------------
    # 1. READ CONFIGURATION SETTINGS
    CONFIG = read_configuration_file(IMG_DIR)

    # optical setup
    M = CONFIG['mag_eff']
    NA = CONFIG['numerical_aperture']
    PIXEL_SIZE = CONFIG['pixel_size']
    MICRONS_PER_PIXEL = CONFIG['microns_per_pixel']
    DOF = CONFIG['depth_of_focus']

    # camera settings
    EXPOSURE_TIME = CONFIG['exposure_time']
    FRAME_RATE = CONFIG['frame_rate']

    # physical setup
    E_SILPURAN = CONFIG['E_silpuran']
    POISSON = CONFIG['poisson']
    T_MEMBRANE = CONFIG['t_membrane']

    # -

    # 1.B. READ TEST-TIMES
    DF_TEST_TIMES = read_configuration_test_times(PATH_CONFIG_TEST_TIMES)

    # ---

    # ----------------------------------------------------------------------------------------------------------------------
    # 2. READ BRIGHTFIELD IMAGE

    IMG_RC = (IMG_XC, IMG_YC)
    path_img_bf_rc = join(REFERENCE_DIR, fn_img_bf_rc)
    path_img_bf_focus_rc = join(REFERENCE_DIR, fn_img_bf_focus_rc)
    path_img_fluoro_rc = join(REFERENCE_DIR, fn_img_fluoro_rc)

    # ---

    # ----------------------------------------------------------------------------------------------------------------------
    # 3. READ TEST DIRECTORIES

    # DICT_, TESTS_TIMES = get_tests_images_creation_time(IMG_DIR, img_filename='test_X1.tif')
    print("NEED TO RESOLVE Z0 DISCREPANCY!")

    TEST_DIR_STARTSWITH = 'test-idpt_testset'
    TEST_DETAILS = get_tests(RESULTS_DIR, TEST_DIR_STARTSWITH, TRACE_DIR, df_test_times=DF_TEST_TIMES)

    # ---

    # ----------------------------------------------------------------------------------------------------------------------
    # 4. READ SURFACE PROFILE COORDS

    fn_profile_coords = 'w{}_merged_process_profiles.xlsx'.format(WAFER_ID)
    path_profile_coords = join(WAFER_FAB_DIR, 'results', fn_profile_coords)

    MERGED_PROCESS_PROFILE_COORDS = read_merged_process_profile_coords(path_profile_coords)

    DFS = get_profile(MERGED_PROCESS_PROFILE_COORDS, FEATURE_ID, PROCESS_STEP)
    Z_MAX, Z_ROUGHNESS = get_profile_depth_and_roughness(DFS, DR_CENTER)

    # plot_profile_and_surface_roughness(DFS, DR_CENTER, save_id=TEST_ID, path_save=VALIDATION_DIR)

    # ---

    # ----------------------------------------------------------------------------------------------------------------------
    # 5. READ TEST COORDS

    FN_TEST_COORDS = 'test_coords_stats.xlsx'
    COLS_INCLUDE = ['frame', 'id', 'z', 'x', 'y', 'cm', 'xg', 'yg']
    COLS_XY = ['x', 'y', 'xg', 'yg']
    COLS_R = ['r', 'rg']
    FRAME_STEP = 1

    PATH_IMG_RCS = [path_img_bf_rc, path_img_bf_focus_rc, path_img_fluoro_rc]

    dict_test = {
        'save_id': TEST_ID,
        'results_dir': RESULTS_DIR,
        'trace_dir': TRACE_DIR,
        'save_dir': SAVE_DIR,
        'fig_dir': FIG_DIR,
        'stats_dir': STATS_DIR,
        'validation_dir': VALIDATION_DIR,
        'animation_dir': ANIMATION_DIR,
        'animation_dir_rz': ANIMATION_DIR_RZ,
        'animation_dir_rz_figs': ANIMATION_DIR_RZ_FIGS,
        'animation_dir_fit_rz_figs': ANIMATION_DIR_FIT_RZ_FIGS,
        'animation_dir_fit_rz_fill_between_figs': ANIMATION_DIR_FIT_RZ_FILL_BETWEEN_FIGS,
        'fn_test_coords': FN_TEST_COORDS,
        'cols_include': COLS_INCLUDE,
        'idpt_padding': IDPT_PADDING,
        'cols_xy': COLS_XY,
        'img_rc': IMG_RC,
        'cols_r': COLS_R,
        'frame_rate': FRAME_RATE,
        'microns_per_pixel': MICRONS_PER_PIXEL,
        't_membrane': T_MEMBRANE,
        'profile_radius': PROFILE_RADIUS,
        'dr_center': DR_CENTER,
        'baseline_frame': BASELINE_FRAME,
        'frame_step': FRAME_STEP,

    }

    # --- THIS IS WHERE YOU ITERATE THROUGH EACH TEST
    # identify_baseline_z0_in_tests(TEST_DETAILS, dict_test)

    iterate_through_tests(TEST_DETAILS, dict_test, df_surface_profile=DFS)

    # --- END

    print("Analysis completed without errors")