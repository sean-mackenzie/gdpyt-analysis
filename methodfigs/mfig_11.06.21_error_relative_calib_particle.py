# test bin, analyze, and plot functions
from os.path import join
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import iqr, gaussian_kde
from sklearn.neighbors import KernelDensity

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from correction.correct import fit_in_focus_plane, correct_z_by_plane_tilt
from correction import correct
from utils import functions, bin, io, fit

# A note on SciencePlots colors
"""
Blue: #0C5DA5
Green: #00B945
Red: #FF9500
Orange: #FF2C00

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

plt.style.use(['science', 'ieee', 'std-colors'])  # 'ieee', 'std-colors'
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# TEST COORDS (FINAL)
"""
IDPT:
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/
results-07.29.22-idpt-tmg'

SPCT:
base_dir = ''
"""

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_error_relative_calib_particle'

method = 'idpt'

if method == 'spct':
    # test_dir = base_dir + '/tests/spct_soft-baseline_1'
    test_id = 1
    test_name = 'test_coords_particle_image_stats_spct-1_dzf-post-processed_raw'  # _corr-fc
    padding = 5
    padding_rel_true_x = 0
    padding_rel_true_y = 0
    calib_id_from_testset = 92
    calib_id_from_calibset = 46

    calib_baseline_frame = 12  # NOTE: baseline frame was 'calib_13.tif' but output coords always begin at frame = 0.

    xsub, ysub, rsub = 'gauss_xc', 'gauss_yc', 'gauss_rc'

elif method == 'idpt':
    # test_dir = base_dir + '/tests/tm16_cm19'
    test_id = 19
    test_name = 'test_coords_particle_image_stats_tm16_cm19_dzf-post-processed'
    padding = 5
    padding_rel_true_x = 0
    padding_rel_true_y = 0

    calib_id_from_testset = 42
    calib_id_from_calibset = 42

    xsub, ysub, rsub = 'xg', 'yg', 'rg'

else:
    raise ValueError()

path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

# ----------------------------------------------------------------------------------------------------------------------
# 0. SETUP PROCESS CONTROLS

# subject to change
true_num_particles_per_frame = 88
# num_dz_steps = 20
baseline_frame = 39
baseline_frames = [39, 40, 41]

# experimental
mag_eff = 10.01
NA_eff = 0.45
microns_per_pixel = 1.6
size_pixels = 16  # microns
depth_of_field = functions.depth_of_field(mag_eff, NA_eff, 600e-9, 1.0, size_pixels * 1e-6) * 1e6
print("Depth of field = {}".format(depth_of_field))
num_pixels = 512
area_pixels = num_pixels ** 2
img_xc, img_yc = num_pixels / 2 + padding, num_pixels / 2 + padding
area_microns = (num_pixels * microns_per_pixel) ** 2

# processing
z_range = [-50, 55]
measurement_depth = z_range[1] - z_range[0]
h = measurement_depth
num_frames_per_step = 3
true_num_particles_per_z = true_num_particles_per_frame * num_frames_per_step
filter_barnkob = measurement_depth / 10
filter_step_size = 10
min_cm = 0.5
min_percent_layers = 0.5
remove_ids = None

# initialize variables which modify later processing decisions
dict_fit_plane = None
dict_fit_plane_bspl_corrected = None
dict_flat_plane = None
bispl = None
bispl_raw = None

# dataset alignment
z_zero_from_calibration = 49.9  # 50.0
z_zero_of_calib_id_from_calibration = 49.6  # the in-focus position of calib particle in test set.

z_zero_from_test_img_center = 68.6  # 68.51
z_zero_of_calib_id_from_test = 68.1  # the in-focus position of calib particle in calib set.


# ---

# define some functions


def fit_line(x, a, b):
    return a * x + b


def fit_kde(y, bandwidth=None):
    """ kdex, kdey, bandwidth = fit_kde(y, bandwidth=None) """

    if bandwidth is None:
        """ Silverman's rule of thumb: https://en.wikipedia.org/wiki/Kernel_density_estimation """
        bandwidth = 0.9 * np.min([np.std(y), iqr(y) / 1.34]) * len(y) ** (-1 / 5)

    # get extents of range that KDE will evaluate over
    ymin, ymax = np.min(y), np.max(y)
    y_range = ymax - ymin

    # setup arrays
    y = y[:, np.newaxis]
    y_plot = np.linspace(ymin - y_range / 12.5, ymax + y_range / 12.5, 300)[:, np.newaxis]

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(y)
    log_dens_y = kde.score_samples(y_plot)

    kdex = y_plot[:, 0]
    kdey = np.exp(log_dens_y)

    return kdex, kdey, bandwidth


def kde_scipy(y, y_grid, bandwidth=0.2, **kwargs):
    """
    pdf, y_grid = kde_scipy(y, y_grid, bandwidth=0.2, **kwargs)

    Reference: https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    """

    if y_grid is None:
        ymin, ymax = np.min(y), np.max(y)
        y_range = ymax - ymin
        y_grid = np.linspace(ymin - y_range / 10, ymax + y_range / 10, 200)

    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(y, bw_method=bandwidth / y.std(ddof=1), **kwargs)
    return kde.evaluate(y_grid), y_grid


def plot_fitted_plane_and_points(df, dict_fit_plane, param_z, param_z_corr):
    rmse, r_squared = dict_fit_plane['rmse'], dict_fit_plane['r_squared']
    tilt_x, tilt_y = dict_fit_plane['tilt_x_degrees'], dict_fit_plane['tilt_y_degrees']
    px, py, pz = dict_fit_plane['px'], dict_fit_plane['py'], dict_fit_plane['pz']
    normal = dict_fit_plane['normal']
    d = dict_fit_plane['d']

    fig = plt.figure(figsize=(6.5, 5))

    for i, v in zip(np.arange(1, 5), [45, 0, 315, 270]):

        ax = fig.add_subplot(2, 2, i, projection='3d')
        sc = ax.scatter(df.x, df.y, df[param_z], c=df[param_z], s=3)
        ax.plot_surface(px, py, pz, alpha=0.4, color='red')
        ax.scatter(df.x, df.y, df[param_z_corr], color='k', s=1)
        ax.view_init(5, v)
        ax.patch.set_alpha(0.0)
        if i == 2:
            plt.colorbar(sc, shrink=0.5)
            ax.get_xaxis().set_ticks([])
            ax.set_ylabel(r'$y \: (pixels)$')
            ax.set_zlabel(r'$z \: (\mu m)$')
        elif i == 4:
            ax.get_yaxis().set_ticks([])
            ax.set_xlabel(r'$x \: (pixels)$')
            ax.set_zlabel(r'$z \: (\mu m)$')
        else:
            ax.set_xlabel(r'$x \: (pixels)$')
            ax.set_ylabel(r'$y \: (pixels)$')
            ax.get_zaxis().set_ticklabels([])

    # title
    plt.suptitle('RMSE: {}, '.format(np.round(rmse, 3)) +
                 r'$R^2$' + ': {}'.format(np.round(r_squared, 3)) + '\n' +
                 r'$(\theta_{x}, \theta_{y})=$' + ' ({}, {} deg.)'.format(np.round(tilt_x, 3), np.round(tilt_y, 3)))
    # deprecated title
    """plt.suptitle(r"$0 = n_x x + n_y y + n_z z - d$" + "= {}x + {}y + {}z - {} \n"
                                                      "(x, y: pixels; z: microns)".format(np.round(normal[0], 5),
                                                                                        np.round(normal[1], 5),
                                                                                        np.round(normal[2], 5),
                                                                                        np.round(d, 5)),
                 y=0.875)"""

    plt.subplots_adjust(hspace=-0.1, wspace=0.15)

    return fig


# ---

# ---
# set limits for all analyses
z_error_limit = 5  # microns
in_plane_distance_threshold = np.round(2 * microns_per_pixel, 1)  # microns
min_counts = 1
min_counts_bin_z = 20
min_counts_bin_r = 20
min_counts_bin_rz = 5

# -

# read test coords
analyze_post_processed_coords_groupby_id = False  # False True
if analyze_post_processed_coords_groupby_id:
    fp = join(base_dir, 'coords', test_name + '.xlsx')
    df = pd.read_excel(fp)

    dfc = df.groupby('id').count().reset_index()
    exclude_ids = dfc[dfc['z'] < min_counts]['id'].to_numpy()

    df = df[~df['id'].isin(exclude_ids)]
    # df = df[df['z_true'].abs() > 7.5]
    df = df[['frame', 'id', 'z', 'z_true', xsub, ysub, rsub, 'x', 'y', 'r']]

    # precision - iterate through each z_true position
    calc_precision = False
    if calc_precision:
        z_trues = df.z_true.unique()
        dfs = []
        for z_true in z_trues:
            dft = df[df['z_true'] == z_true]
            dfmr = dft.groupby('id').mean()['r'].to_numpy()
            dft = dft.groupby('id').std().reset_index()
            dft['r'] = dfmr
            dfs.append(dft)

        dfs = pd.concat(dfs)
        dfm = dfs.groupby('id').mean()
        dfmstd = dfs.groupby('id').mean()
        dfmstd['r'] = dfm['r'].to_numpy()
        dfmstd.to_excel(join(base_dir, 'results', test_name + '_precision-by-id' + '.xlsx'))

        fig, ax = plt.subplots()
        ax.scatter(dfm['r'], dfmstd['z'])
        ax.set_xlabel('r')
        ax.set_ylabel(r'$z-precision \: (\mu m)$')
        plt.tight_layout()
        plt.savefig(join(path_figs, '{}_z-precision_by_r.png'.format(method)))
        plt.show()
        plt.close()
        raise ValueError()

    # iterate through each z_true position
    frames = df.frame.unique()
    dfs = []
    for frame in frames:
        dft = df[df['frame'] == frame]

        z_true = dft[dft['id'] == calib_id_from_testset].z_true.iloc[0]
        z_calib = dft[dft['id'] == calib_id_from_testset].z.iloc[0]

        dft['error_rel_p_calib'] = dft['z'] - z_calib
        dfs.append(dft)

        """fig, ax = plt.subplots()
        ax.scatter(dft['r'], dft['error_rel_p_calib'])
        ax.set_xlabel('r')
        ax.set_ylabel(r'$\epsilon_{z}$')
        ax.set_title('frame: {}, z_true = {}'.format(frame, np.round(z_true, 1)))
    
        plt.tight_layout()
        plt.show()
        plt.close()"""

    # ---

    dfs = pd.concat(dfs)
    dfs['abs_error_rel_p_calib'] = dfs['error_rel_p_calib'].abs()

    dfg = dfs.groupby('id').mean()
    dfg.to_excel(join(base_dir, 'results', test_name + '_group-id' + '.xlsx'))

    fig, ax = plt.subplots()
    ax.scatter(dfg['r'], dfg['error_rel_p_calib'])
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
    plt.tight_layout()
    plt.savefig(join(path_figs, '{}_error_rel_p_calib_by_r.png'.format(method)))
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    ax.scatter(dfg['r'], dfg['abs_error_rel_p_calib'])
    ax.set_xlabel('r')
    ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
    plt.tight_layout()
    plt.savefig(join(path_figs, '{}_abs_error_rel_p_calib_by_r.png'.format(method)))
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    ax.scatter(dfg['r'], dfg['error_rel_p_calib'] * -1, label='mirror error')
    ax.scatter(dfg['r'], dfg['abs_error_rel_p_calib'], label='abs error')
    ax.set_xlabel('r')
    ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(join(path_figs, '{}_mirror_abs_error_rel_p_calib_by_r.png'.format(method)))
    plt.show()
    plt.close()

# ---

# compare errors relative to calibration particle
compare_errors_groupby_id = False  # False True
if compare_errors_groupby_id:

    fpi = 'test_coords_particle_image_stats_tm16_cm19_dzf-post-processed'
    fps = 'test_coords_particle_image_stats_spct-1_dzf-post-processed'  # _corr-fc_flipFalse
    fpsc = 'test_coords_particle_image_stats_spct-1_dzf-post-processed_corr-fc'  # _flipTrue
    fpsr = 'test_coords_particle_image_stats_spct-1_dzf-post-processed_raw'

    dfi = pd.read_excel(join(base_dir, 'results', fpi + '_group-id' + '.xlsx'))
    dfs = pd.read_excel(join(base_dir, 'results', fps + '_group-id' + '.xlsx'))
    dfsc = pd.read_excel(join(base_dir, 'results', fpsc + '_group-id' + '.xlsx'))
    dfsr = pd.read_excel(join(base_dir, 'results', fpsr + '_group-id' + '.xlsx'))

    px = 'r'
    pys = ['error_rel_p_calib', 'abs_error_rel_p_calib']
    py = pys[1]
    ms = 3

    fig, ax = plt.subplots()

    dfes = [dfi, dfs, dfsc, dfsr]
    for dfe, mtd in zip(dfes, ['IDPT', 'SPCT', 'SPCT f.c.', 'SPCT raw']):  # for py in pys:
        dfe = dfe.reset_index()
        dfe['error_squared'] = dfe[py] ** 2
        dfe = dfe.groupby('id').mean()
        dfe['rmse_z'] = np.sqrt(dfe['error_squared'])
        # dfe = dfe.sort_values('rmse_z')
        ax.scatter(dfe[px], dfe['rmse_z'], s=ms, label='{}: {}'.format(mtd, np.round(dfe['rmse_z'].mean(), 2)))

        popt, pcov = curve_fit(fit_line, dfe[px], dfe['rmse_z'])
        ax.plot(dfe[px], fit_line(dfe[px], *popt), label=np.round(popt[0], 4))

    """ax.scatter(dfi[px], dfi[py], s=ms, label='IDPT')
    ax.scatter(dfs[px], dfs[py], s=ms, label='SPCT f.c. flip=False')
    ax.scatter(dfsc[px], dfsc[py], s=ms, label='SPCT f.c. flip=True')"""

    ax.set_xlabel('r')
    ax.set_ylabel('rmse z')
    ax.set_ylim([-0.2, 2.6])
    ax.legend(title='Method: mean rmse-z')
    plt.tight_layout()
    plt.savefig(join(path_figs, 'compare_{}_by_r_fit-line_with-raw.png'.format('rmse_z')))
    plt.show()
    plt.close()

# ---

# compare errors relative to calibration particle
compare_precision_groupby_id = False  # False True
if compare_precision_groupby_id:

    fpi = 'test_coords_particle_image_stats_tm16_cm19_dzf-post-processed'
    fps = 'test_coords_particle_image_stats_spct-1_dzf-post-processed'  # _corr-fc_flipFalse

    dfi = pd.read_excel(join(base_dir, 'results', fpi + '_precision-by-id' + '.xlsx'))
    dfs = pd.read_excel(join(base_dir, 'results', fps + '_precision-by-id' + '.xlsx'))

    px = 'r'
    py = 'z'
    ms = 3

    fig, ax = plt.subplots()

    ax.scatter(dfi[px], dfi[py], s=ms, label='IDPT, {}'.format(np.round(dfi[py].mean(), 3)))
    ax.scatter(dfs[px], dfs[py], s=ms, label='SPCT, {}'.format(np.round(dfs[py].mean(), 3)))

    for df in [dfi, dfs]:
        popt, pcov = curve_fit(fit_line, df[px], df[py])
        ax.plot(df[px], fit_line(df[px], *popt), label=popt)

    ax.set_xlabel('r')
    ax.set_ylabel('precision z')
    # ax.set_ylim([])
    ax.legend(title='Method: mean', loc='upper left')
    plt.tight_layout()
    plt.savefig(join(path_figs, 'compare_precision_by_r.png'))
    plt.show()
    plt.close()

# ---

# for each true axial position, correct plane tilt and then calculate error relative to the calibration particle
analyze_error_relative_calib_post_tilt_corr = False  # False True
if analyze_error_relative_calib_post_tilt_corr:

    method = 'spct'  # idpt spct
    min_cm = 0.5  # 0.5 0.9
    correct_tilt = True
    plot_tilt_per_frame = True
    plot_3d_tilt_per_frame = True

    # save path
    path_results_tilt_corr = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                                  'zerrlim{}_cmin{}_mincountsallframes{}'.format(z_error_limit, min_cm, min_counts))
    path_scatter_zy_by_z = join(path_results_tilt_corr, '{}_scatter_zy_by_z'.format(method))

    if not os.path.exists(path_results_tilt_corr):
        os.makedirs(path_results_tilt_corr)
    if not os.path.exists(path_scatter_zy_by_z):
        os.makedirs(path_scatter_zy_by_z)

    # ---

    if method == 'spct':
        # test_dir = base_dir + '/tests/spct_soft-baseline_1'
        test_id = 1
        test_name = 'test_coords_particle_image_stats_spct-1_dzf-post-processed_raw'  # _corr-fc
        padding = 5
        padding_rel_true_x = 0
        padding_rel_true_y = 0
        calib_id_from_testset = 92
        calib_id_from_calibset = 46
        calib_baseline_frame = 12  # NOTE: baseline frame was 'calib_13.tif' but output coords always begin at frame = 0.
        xsub, ysub, rsub = 'gauss_xc', 'gauss_yc', 'gauss_rc'

    elif method == 'idpt':
        # test_dir = base_dir + '/tests/tm16_cm19'
        test_id = 19
        test_name = 'test_coords_particle_image_stats_tm16_cm19_dzf-post-processed'
        padding = 5
        padding_rel_true_x = 0
        padding_rel_true_y = 0
        calib_id_from_testset = 42
        calib_id_from_calibset = 42
        xsub, ysub, rsub = 'xg', 'yg', 'rg'

    else:
        raise ValueError()

    fp = join(base_dir, 'coords', test_name + '.xlsx')
    df = pd.read_excel(fp)

    # filter 1. filter by number of counts
    dfc = df.groupby('id').count().reset_index()
    exclude_ids = dfc[dfc['z'] < min_counts]['id'].to_numpy()
    df = df[~df['id'].isin(exclude_ids)]

    # filter 2. filter by Cm
    df = df[df['cm'] > min_cm]

    # filter by axial position
    # df = df[df['z_true'].abs() > 7.5]

    # -

    # get only necessary columns
    df = df[['frame', 'id', 'cm', 'z', 'z_true', xsub, ysub, rsub]]  # , 'x', 'y', 'r'
    df = df.rename(columns={xsub: 'x', ysub: 'y', rsub: 'r'})  # do this to fit plane to sub-positions

    # -

    # iterate through each z_true position
    z_trues = df.z_true.unique()
    dfs = []
    fit_plane_img_xyzc = []
    fit_plane_rmsez = []
    for z_true in z_trues:
        # clear z_calib
        z_calib = None

        dft = df[df['z_true'] == z_true]

        # --- correct tilt
        if correct_tilt:
            # step 0. filter dft such that it only includes particles that could reasonably be on the tilt surface
            reasonable_z_tilt_limit = 3.25
            reasonable_r_tilt_limit = int(np.round(250 / microns_per_pixel))  # convert units microns to pixels
            dft_within_tilt = dft[np.abs(dft['z'] - z_true) < reasonable_z_tilt_limit]

            # step 0.5. check if calibration particle is in this new group
            if not calib_id_from_testset in dft_within_tilt.id.unique():
                # z_calib = dft_within_tilt[dft_within_tilt['r'] < reasonable_r_tilt_limit].z.mean()
                z_calib = -3.6

            # step 1. fit plane to particle positions
            dict_fit_plane = fit_in_focus_plane(df=dft_within_tilt,  # note: x,y units are pixels at this point
                                                param_zf='z',
                                                microns_per_pixel=microns_per_pixel,
                                                img_xc=img_xc,
                                                img_yc=img_yc)

            fit_plane_img_xyzc.append(dict_fit_plane['z_f_fit_plane_image_center'])
            fit_plane_rmsez.append(dict_fit_plane['rmse'])

            # step 2. correct coordinates using fitted plane
            dft['z_plane'] = functions.calculate_z_of_3d_plane(dft.x, dft.y, popt=dict_fit_plane['popt_pixels'])
            dft['z_plane'] = dft['z_plane'] - dict_fit_plane['z_f_fit_plane_image_center']
            dft['z_corr'] = dft['z'] - dft['z_plane']

            # add column for tilt
            dft['tilt_x_degrees'] = dict_fit_plane['tilt_x_degrees']
            dft['tilt_y_degrees'] = dict_fit_plane['tilt_y_degrees']

            # rename
            dft = dft.rename(columns={'z': 'z_no_corr'})
            dft = dft.rename(columns={'z_corr': 'z'})

            if plot_tilt_per_frame:
                fig, (axx, ax) = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches * 1.5, size_y_inches))

                axx.scatter(dft['x'], dft['z'], s=2, label='tilt corr')
                axx.scatter(dft['x'], dft['z_no_corr'], s=2, label='raw')
                axx.scatter(dft_within_tilt['x'], dft_within_tilt['z'], s=1, marker='.', color='k', label='raw-fitted')
                axx.scatter(dft[dft['id'] == calib_id_from_testset]['x'], dft[dft['id'] == calib_id_from_testset]['z'],
                            marker='*', s=8, color='red', label=r'$p_{cal}$')

                ax.scatter(dft['y'], dft['z'], s=2, label='tilt corr')
                ax.scatter(dft['y'], dft['z_no_corr'], s=2, label='raw')
                ax.scatter(dft_within_tilt['y'], dft_within_tilt['z'], s=1, marker='.', color='k', label='raw-fitted')
                ax.scatter(dft[dft['id'] == calib_id_from_testset]['y'], dft[dft['id'] == calib_id_from_testset]['z'],
                           marker='*', s=8, color='red', label=r'$p_{cal}$')

                if z_calib is not None:
                    axx.plot([img_xc - reasonable_r_tilt_limit, img_xc + reasonable_r_tilt_limit],
                             [z_calib, z_calib],
                             color='k', label='avg(r<150um)={}'.format(np.round(z_calib, 1)))
                    ax.plot([img_yc - reasonable_r_tilt_limit, img_yc + reasonable_r_tilt_limit],
                            [z_calib, z_calib],
                            color='k', label='avg(r<150um)={}'.format(np.round(z_calib, 1)))

                # plot fitted plane
                plane_x = dict_fit_plane['px']
                plane_y = dict_fit_plane['py']
                plane_z = dict_fit_plane['pz']

                plot_plane_along_xix = [plane_x[0][0], plane_x[0][1]]
                plot_plane_along_xiz = [plane_z[0][0], plane_z[0][1]]
                plot_plane_along_xfx = [plane_x[1][0], plane_x[1][1]]
                plot_plane_along_xfz = [plane_z[1][0], plane_z[1][1]]
                axx.plot(plot_plane_along_xix, plot_plane_along_xiz, color='k', alpha=0.5)
                axx.plot(plot_plane_along_xfx, plot_plane_along_xfz, color='k', alpha=0.5)

                plot_plane_along_yiy = [plane_y[0][0], plane_y[1][0]]
                plot_plane_along_yiz = [plane_z[0][0], plane_z[1][0]]
                plot_plane_along_yfy = [plane_y[0][1], plane_y[1][1]]
                plot_plane_along_yfz = [plane_z[0][1], plane_z[1][1]]
                ax.plot(plot_plane_along_yiy, plot_plane_along_yiz, color='k', alpha=0.5)
                ax.plot(plot_plane_along_yfy, plot_plane_along_yfz, color='k', alpha=0.5)

                axx.set_ylabel('z')
                axx.set_xlabel('x')
                ax.set_xlabel('y')
                ax.legend(fontsize='small', frameon=True)
                plt.suptitle('z_true = {}, tilt(x={}, y={}) deg'.format(np.round(z_true, 1),
                                                                        np.round(dict_fit_plane['tilt_x_degrees'], 2),
                                                                        np.round(dict_fit_plane['tilt_y_degrees'], 2)
                                                                        ),
                             )
                plt.tight_layout()
                plt.savefig(join(path_scatter_zy_by_z, 'scatter-z-by-y_z-true={}.png'.format(np.round(z_true, 1))))
                plt.close()

                # ---

                # plotted fitted plane and points in 3D
                if plot_3d_tilt_per_frame:
                    fig = plot_fitted_plane_and_points(df=dft, dict_fit_plane=dict_fit_plane, param_z='z_no_corr',
                                                       param_z_corr='z')
                    plt.savefig(join(path_scatter_zy_by_z,
                                     'scatter-3D-with-plane-scatter_z-true={}.png'.format(np.round(z_true, 1))))
                    plt.close()

                # raise ValueError()
        # ---

        # get average position of calibration particle
        if z_calib is None:
            z_calib = dft[dft['id'] == calib_id_from_testset].z.mean()
        dft['z_calib'] = z_calib
        dft['error_rel_p_calib'] = dft['z'] - z_calib
        dfs.append(dft)

    # ---
    # analyze fit plane xyzc and rmse-z
    df_fit_plane = pd.DataFrame(data=np.vstack([z_trues, fit_plane_img_xyzc, fit_plane_rmsez]).T,
                                columns=['z_nominal', 'z_xyc', 'rmsez'])
    df_fit_plane['z_diff'] = df_fit_plane['z_xyc'] - df_fit_plane['z_nominal']
    df_fit_plane.to_excel(join(path_scatter_zy_by_z, 'fit-plane-xyzc-rmsez_by_z-true.xlsx'))

    # plot fit_plane_image_xyzc (the z-position at the center of the image) and rmse-z as a function of z_true
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(df_fit_plane['z_nominal'], df_fit_plane['z_diff'], '-o', label='xzyc')
    ax2.plot(df_fit_plane['z_nominal'], df_fit_plane['rmsez'], '-o', label='rmse')
    ax1.set_ylabel(r'$z_{nom} - z_{xyc} \: (\mu m)$')
    ax2.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    ax2.set_xlabel(r'$z_{nominal}$')
    plt.tight_layout()
    plt.savefig(join(path_scatter_zy_by_z, 'fit-plane-xyzc-rmsez_by_z-true.png'))
    plt.show()
    plt.close()

    # ---
    raise ValueError()

    dfs = pd.concat(dfs)
    dfs['abs_error_rel_p_calib'] = dfs['error_rel_p_calib'].abs()
    dfs = dfs[dfs['abs_error_rel_p_calib'] < z_error_limit]

    dfg = dfs
    # dfg = dfs.groupby('id').mean()
    dfg.to_excel(join(path_results_tilt_corr, '{}_error_relative_calib_particle_'.format(method) +
                      'zerrlim{}_cmin{}_mincountsallframes{}.xlsx'.format(z_error_limit, min_cm, min_counts)))

    plot_errors_rel_calib = True
    if plot_errors_rel_calib:
        fig, ax = plt.subplots()
        ax.scatter(dfg['r'], dfg['error_rel_p_calib'], s=2)
        ax.set_xlabel('r')
        ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
        plt.tight_layout()
        plt.savefig(join(path_results_tilt_corr, '{}_error_relative_calib_particle_'.format(method) +
                         'zerrlim{}_cmin{}_mincountsallframes{}_'.format(z_error_limit, min_cm, min_counts) +
                         'scatter-error_rel_p_calib.png'))
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.scatter(dfg['r'], dfg['abs_error_rel_p_calib'], s=2)
        ax.set_xlabel('r')
        ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
        plt.tight_layout()
        plt.savefig(join(path_results_tilt_corr, '{}_error_relative_calib_particle_'.format(method) +
                         'zerrlim{}_cmin{}_mincountsallframes{}_'.format(z_error_limit, min_cm, min_counts) +
                         'scatter-abs-error_rel_p_calib.png'))
        plt.show()
        plt.close()

    # ---

# ---

analyze_rmse_relative_calib_post_tilt_corr = False  # False True
if analyze_rmse_relative_calib_post_tilt_corr:

    plot_bin_z = True
    plot_bin_r = True
    plot_bin_r_z = True
    plot_bin_id = True
    plot_cmin_zero_nine = False

    correct_tilt_by_fit_idpt = True
    assign_z_true_to_fit_plane_xyzc = True

    # ---

    # read paths
    if not correct_tilt_by_fit_idpt:
        fpi = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                   'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                   'idpt_error_relative_calib_particle_' +
                   'zerrlim{}_cmin0.5_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))
        fps = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                   'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                   'spct_error_relative_calib_particle_' +
                   'zerrlim{}_cmin0.5_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))
        fpss = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                    'zerrlim{}_cmin0.9_mincountsallframes{}'.format(z_error_limit, min_counts),
                    'spct_error_relative_calib_particle_' +
                    'zerrlim{}_cmin0.9_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))

        # save paths
        path_save_z = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                           'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                           'bin-z_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit, min_counts))
        path_save_rz = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                            'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                            'bin-r-z_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit, min_counts))
        path_save_r = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                           'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                           'bin-r_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit, min_counts))
        path_save_id = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                            'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                            'bin-id_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit, min_counts))
    elif assign_z_true_to_fit_plane_xyzc:
        res_dir = 'relative-to-tilt-corr-calib-particle_08.06.23_raw-original'
        fpi = join(base_dir, 'results', res_dir,
                   'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                   'ztrue_is_fit-plane-xyzc', 'idpt_error_relative_calib_particle_' +
                   'zerrlim{}_cmin0.5_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))
        fps = join(base_dir, 'results', res_dir,
                   'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                   'ztrue_is_fit-plane-xyzc', 'spct_error_relative_calib_particle_' +
                   'zerrlim{}_cmin0.5_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))
        fpss = join(base_dir, 'results', res_dir,
                    'zerrlim{}_cmin0.9_mincountsallframes{}'.format(z_error_limit, min_counts),
                    'ztrue_is_fit-plane-xyzc', 'spct_error_relative_calib_particle_' +
                    'zerrlim{}_cmin0.9_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))

        # save paths
        path_save_z = join(base_dir, 'results', res_dir,
                           'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                           'ztrue_is_fit-plane-xyzc',
                           'bin-z_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit, min_counts))
        path_save_rz = join(base_dir, 'results', res_dir,
                            'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                            'ztrue_is_fit-plane-xyzc',
                            'bin-r-z_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit, min_counts))
        path_save_r = join(base_dir, 'results', res_dir,
                           'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                           'ztrue_is_fit-plane-xyzc',
                           'bin-r_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit, min_counts))
        path_save_id = join(base_dir, 'results', res_dir,
                            'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                            'ztrue_is_fit-plane-xyzc',
                            'bin-id_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit, min_counts))
    else:
        fpi = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                   'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                   'corr-tilt-by-fit-idpt', 'idpt_error_relative_calib_particle_' +
                   'zerrlim{}_cmin0.5_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))
        fps = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                   'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                   'corr-tilt-by-fit-idpt', 'spct_error_relative_calib_particle_' +
                   'zerrlim{}_cmin0.5_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))
        fpss = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                    'zerrlim{}_cmin0.9_mincountsallframes{}'.format(z_error_limit, min_counts),
                    'corr-tilt-by-fit-idpt', 'spct_error_relative_calib_particle_' +
                    'zerrlim{}_cmin0.9_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))

        # save paths
        path_save_z = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                           'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                           'corr-tilt-by-fit-idpt',
                           'bin-z_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit, min_counts))
        path_save_rz = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                            'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                            'corr-tilt-by-fit-idpt',
                            'bin-r-z_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit, min_counts))
        path_save_r = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                           'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                           'corr-tilt-by-fit-idpt',
                           'bin-r_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit, min_counts))
        path_save_id = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                            'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                            'corr-tilt-by-fit-idpt',
                            'bin-id_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit, min_counts))

    # ---

    # read coords
    dfi = pd.read_excel(fpi)
    dfs = pd.read_excel(fps)
    dfss = pd.read_excel(fpss)

    # ---

    # process data

    # number of z-positions
    num_z_positions = len(dfi.z_true.unique())
    true_total_num = true_num_particles_per_z * num_z_positions

    # scale to microns
    dfi['r_microns'] = dfi['r'] * microns_per_pixel
    dfs['r_microns'] = dfs['r'] * microns_per_pixel
    dfss['r_microns'] = dfss['r'] * microns_per_pixel

    # square all errors
    dfi['rmse_z'] = dfi['error_rel_p_calib'] ** 2
    dfs['rmse_z'] = dfs['error_rel_p_calib'] ** 2
    dfss['rmse_z'] = dfss['error_rel_p_calib'] ** 2

    # ---

    # -------------
    # bin by axial position

    # bin by z

    if plot_bin_z:

        if not os.path.exists(path_save_z):
            os.makedirs(path_save_z)

        # setup 2D binning
        z_trues = dfi.z_true.unique()

        column_to_bin = 'z_true'
        column_to_count = 'id'
        bins = z_trues
        round_to_decimal = 1
        return_groupby = True

        # compute 1D bin (z)
        dfim, dfistd = bin.bin_generic(dfi, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)
        dfsm, dfsstd = bin.bin_generic(dfs, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)
        dfssm, dfssstd = bin.bin_generic(dfss, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)

        # compute rmse-z
        dfim['rmse_z'] = np.sqrt(dfim['rmse_z'])
        dfsm['rmse_z'] = np.sqrt(dfsm['rmse_z'])
        dfssm['rmse_z'] = np.sqrt(dfssm['rmse_z'])


        # compute final stats and package prior to exporting
        def package_for_export(df_):
            """ df = package_for_export(df_=df) """
            df_['true_num_per_z'] = true_num_particles_per_z
            df_['percent_meas'] = df_['count_id'] / df_['true_num_per_z']
            df_ = df_.rename(columns=
                             {'z_true': 'z_nominal',
                              'z_calib': 'z_assert_true',
                              'error_rel_p_calib': 'error_rel_z_assert_true',
                              'abs_error_rel_p_calib': 'abs_error_rel_z_assert_true'}
                             )
            df_ = df_.drop(columns=['frame', 'id', 'z_no_corr', 'x', 'y', 'r', 'z_plane', 'r_microns'])
            return df_


        dfim = package_for_export(df_=dfim)
        dfsm = package_for_export(df_=dfsm)
        dfssm = package_for_export(df_=dfssm)

        # export
        dfim.to_excel(join(path_save_z, 'idpt_cm0.5_bin-z_rmse-z.xlsx'))
        dfsm.to_excel(join(path_save_z, 'spct_cm0.5_bin-z_rmse-z.xlsx'))
        dfssm.to_excel(join(path_save_z, 'spct_cm0.9_bin-z_rmse-z.xlsx'))

        # ---

        # plotting

        # filter before plotting
        dfim = dfim[dfim['count_id'] > min_counts_bin_z]
        dfsm = dfsm[dfsm['count_id'] > min_counts_bin_z]
        # dfssm = dfssm[dfssm['count_id'] > min_counts_bin_z]

        # plot: rmse_z by z_nominal (i.e., bin)
        fig, ax = plt.subplots(figsize=(size_x_inches * 1., size_y_inches * 1.))

        ax.plot(dfim.bin, dfim['rmse_z'], '-o', label='IDPT' + r'$(C_{m,min}=0.5)$')
        ax.plot(dfsm.bin, dfsm['rmse_z'], '-o', label='SPCT' + r'$(C_{m,min}=0.5)$')
        if plot_cmin_zero_nine:
            ax.plot(dfssm.bin, dfssm['rmse_z'], '-o', label='SPCT' + r'$(C_{m,min}=0.9)$')
            save_lbl = 'bin-z_rmse-z_by_z_all'
        else:
            save_lbl = 'bin-z_rmse-z_by_z'

        ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        ax.set_ylim([0, 3.25])
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_xticks([-50, -25, 0, 25, 50])
        ax.legend(loc='upper left')  # , borderpad=0.25, handletextpad=0.6, borderaxespad=0.3, markerscale=0.75)

        plt.tight_layout()
        plt.savefig(join(path_save_z, save_lbl + '.png'))
        plt.show()
        plt.close()

        # ---

        # plot: local (1) correlation coefficient, (2) percent measure, and (3) rmse_z

        # setup
        zorder_i, zorder_s, zorder_ss = 3.5, 3.3, 3.4
        ms = 4

        # plot
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches * 1.35, size_y_inches * 1.25))

        if plot_cmin_zero_nine:
            ax1.plot(dfim.bin, dfim['cm'], '-o', ms=ms, label='IDPT' + r'$(C_{m,min}=0.5)$', zorder=zorder_i)
            ax1.plot(dfsm.bin, dfsm['cm'], '-o', ms=ms, label='SPCT' + r'$(C_{m,min}=0.5)$', zorder=zorder_s)
            ax1.plot(dfssm.bin, dfssm['cm'], '-o', ms=ms, label='SPCT' + r'$(C_{m,min}=0.9)$', zorder=zorder_ss)

            ax2.plot(dfim.bin, dfim['percent_meas'], '-o', ms=ms, label='IDPT' + r'$(C_{m,min}=0.5)$', zorder=zorder_i)
            ax2.plot(dfsm.bin, dfsm['percent_meas'], '-o', ms=ms, label='SPCT' + r'$(C_{m,min}=0.5)$', zorder=zorder_s)
            ax2.plot(dfssm.bin, dfssm['percent_meas'], '-o', ms=ms, label='SPCT' + r'$(C_{m,min}=0.9)$',
                     zorder=zorder_ss)

            ax3.plot(dfim.bin, dfim['rmse_z'], '-o', ms=ms, label='IDPT' + r'$(C_{m,min}=0.5)$', zorder=zorder_i)
            ax3.plot(dfsm.bin, dfsm['rmse_z'], '-o', ms=ms, label='SPCT' + r'$(C_{m,min}=0.5)$', zorder=zorder_s)
            ax3.plot(dfssm.bin, dfssm['rmse_z'], '-o', ms=ms, label='SPCT' + r'$(C_{m,min}=0.9)$', zorder=zorder_ss)

            save_lbl = 'bin-z_local-cm-percent-meas-rmse-z_by_z_all'

        else:
            ax1.plot(dfim.bin, dfim['cm'], '-o', ms=ms, label='IDPT', zorder=zorder_i)
            ax1.plot(dfsm.bin, dfsm['cm'], '-o', ms=ms, label='SPCT', zorder=zorder_s)

            ax2.plot(dfim.bin, dfim['percent_meas'], '-o', ms=ms, label='IDPT', zorder=zorder_i)
            ax2.plot(dfsm.bin, dfsm['percent_meas'], '-o', ms=ms, label='SPCT', zorder=zorder_s)

            ax3.plot(dfim.bin, dfim['rmse_z'], '-o', ms=ms, label='IDPT', zorder=zorder_i)
            ax3.plot(dfsm.bin, dfsm['rmse_z'], '-o', ms=ms, label='SPCT', zorder=zorder_s)

            save_lbl = 'bin-z_local-cm-percent-meas-rmse-z_by_z'

        ax1.set_ylabel(r'$C_{m}^{\delta}$')
        ax1.legend(loc='upper left',
                   bbox_to_anchor=(1, 1))  # , borderpad=0.25, handletextpad=0.6, borderaxespad=0.3, markerscale=0.75)

        ax2.set_ylabel(r'$\phi_{z}^{\delta}$')

        ax3.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        # ax3.set_ylim([0, 2.8])
        # ax3.set_yticks([0, 1, 2])
        ax3.set_xlabel(r'$z \: (\mu m)$')
        ax3.set_xticks([-50, -25, 0, 25, 50])

        plt.tight_layout()
        plt.savefig(join(path_save_z, save_lbl + '.png'))
        plt.show()
        plt.close()

        # ---

        # compute mean rmse-z (using 1 bin)
        bin_h = 1

        dfim, _ = bin.bin_generic(dfi, column_to_bin, column_to_count, bin_h, round_to_decimal, return_groupby)
        dfsm, _ = bin.bin_generic(dfs, column_to_bin, column_to_count, bin_h, round_to_decimal, return_groupby)
        dfssm, _ = bin.bin_generic(dfss, column_to_bin, column_to_count, bin_h, round_to_decimal, return_groupby)

        dfim['rmse_z'] = np.sqrt(dfim['rmse_z'])
        dfsm['rmse_z'] = np.sqrt(dfsm['rmse_z'])
        dfssm['rmse_z'] = np.sqrt(dfssm['rmse_z'])


        # compute final stats and package prior to exporting
        def package_for_export(df_):
            """ df = package_for_export(df_=df) """
            df_['true_num'] = true_total_num
            df_['percent_meas'] = df_['count_id'] / df_['true_num']
            df_ = df_.rename(columns={'error_rel_p_calib': 'error_rel_z_assert_true',
                                      'abs_error_rel_p_calib': 'abs_error_rel_z_assert_true'})
            df_ = df_.drop(
                columns=['frame', 'id', 'z_no_corr', 'z_true', 'z_calib', 'x', 'y', 'r', 'z_plane', 'r_microns'])
            return df_


        dfim = package_for_export(df_=dfim)
        dfsm = package_for_export(df_=dfsm)
        dfssm = package_for_export(df_=dfssm)

        dfim.to_excel(join(path_save_z, 'idpt_cm0.5_mean_rmse-z_by_z.xlsx'))
        dfsm.to_excel(join(path_save_z, 'spct_cm0.5_mean_rmse-z_by_z.xlsx'))
        dfssm.to_excel(join(path_save_z, 'spct_cm0.9_mean_rmse-z_by_z.xlsx'))

    # -

    # -------------------

    # -

    # -------------------
    # bin by radial position

    # bin by r

    if plot_bin_r:

        if not os.path.exists(path_save_r):
            os.makedirs(path_save_r)

        # setup 2D binning
        r_bins = 4

        column_to_bin = 'r_microns'
        column_to_count = 'id'
        bins = r_bins
        round_to_decimal = 1
        return_groupby = True

        # compute 1D bin (z)
        dfim, dfistd = bin.bin_generic(dfi, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)
        dfsm, dfsstd = bin.bin_generic(dfs, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)
        dfssm, dfssstd = bin.bin_generic(dfss, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)

        # compute rmse-z
        dfim['rmse_z'] = np.sqrt(dfim['rmse_z'])
        dfsm['rmse_z'] = np.sqrt(dfsm['rmse_z'])
        dfssm['rmse_z'] = np.sqrt(dfssm['rmse_z'])


        # compute final stats and package prior to exporting
        def package_for_export(df_):
            """ df = package_for_export(df_=df) """
            df_ = df_.rename(columns=
                             {'r': 'r_pixels',
                              'error_rel_p_calib': 'error_rel_z_assert_true',
                              'abs_error_rel_p_calib': 'abs_error_rel_z_assert_true'}
                             )
            df_ = df_.drop(columns=['frame', 'id', 'z_no_corr', 'x', 'y', 'z_plane', 'z_true', 'z_calib',
                                    'tilt_x_degrees', 'tilt_y_degrees'])
            return df_


        dfim = package_for_export(df_=dfim)
        dfsm = package_for_export(df_=dfsm)
        dfssm = package_for_export(df_=dfssm)

        # export
        dfim.to_excel(join(path_save_r, 'idpt_cm0.5_bin-r_rmse-z.xlsx'))
        dfsm.to_excel(join(path_save_r, 'spct_cm0.5_bin-r_rmse-z.xlsx'))
        dfssm.to_excel(join(path_save_r, 'spct_cm0.9_bin-r_rmse-z.xlsx'))

        # ---

        # plotting

        # filter before plotting
        dfim = dfim[dfim['count_id'] > min_counts_bin_r]
        dfsm = dfsm[dfsm['count_id'] > min_counts_bin_r]
        dfssm = dfssm[dfssm['count_id'] > min_counts_bin_r]

        # plot
        fig, ax = plt.subplots(figsize=(size_x_inches * 1., size_y_inches * 1.))

        ax.plot(dfim.bin, dfim['rmse_z'], '-o', label='IDPT' + r'$(C_{m,min}=0.5)$')
        ax.plot(dfsm.bin, dfsm['rmse_z'], '-o', label='SPCT' + r'$(C_{m,min}=0.5)$')
        if plot_cmin_zero_nine:
            ax.plot(dfssm.bin, dfssm['rmse_z'], '-o', label='SPCT' + r'$(C_{m,min}=0.9)$')
            save_lbl = 'bin-r_rmse-z_by_r_all'
        else:
            save_lbl = 'bin-r_rmse-z_by_r'

        ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        ax.set_ylim([0, 2.4])
        ax.set_xlim([50, 500])
        ax.set_xticks([100, 200, 300, 400, 500])
        ax.set_xlabel(r'$r \: (\mu m)$')
        # ax.set_xticks([-50, -25, 0, 25, 50])
        ax.legend(loc='upper left')  # , borderpad=0.25, handletextpad=0.6, borderaxespad=0.3, markerscale=0.75)

        plt.tight_layout()
        plt.savefig(join(path_save_r, save_lbl + '.png'))
        plt.show()
        plt.close()

        # ---

        # compute mean rmse-z (using 1 bin)
        bin_h = 1

        dfim, _ = bin.bin_generic(dfi, column_to_bin, column_to_count, bin_h, round_to_decimal, return_groupby)
        dfsm, _ = bin.bin_generic(dfs, column_to_bin, column_to_count, bin_h, round_to_decimal, return_groupby)
        dfssm, _ = bin.bin_generic(dfss, column_to_bin, column_to_count, bin_h, round_to_decimal, return_groupby)

        dfim['rmse_z'] = np.sqrt(dfim['rmse_z'])
        dfsm['rmse_z'] = np.sqrt(dfsm['rmse_z'])
        dfssm['rmse_z'] = np.sqrt(dfssm['rmse_z'])


        # compute final stats and package prior to exporting
        def package_for_export(df_):
            """ df = package_for_export(df_=df) """
            df_ = df_.rename(columns=
                             {'error_rel_p_calib': 'error_rel_z_assert_true',
                              'abs_error_rel_p_calib': 'abs_error_rel_z_assert_true'}
                             )
            df_ = df_.drop(columns=['frame', 'id', 'z_no_corr', 'x', 'y', 'z_plane', 'z_true', 'z_calib',
                                    'tilt_x_degrees', 'tilt_y_degrees', 'r', 'r_microns'])
            return df_


        dfim = package_for_export(df_=dfim)
        dfsm = package_for_export(df_=dfsm)
        dfssm = package_for_export(df_=dfssm)

        dfim.to_excel(join(path_save_r, 'idpt_cm0.5_mean_rmse-z_by_r.xlsx'))
        dfsm.to_excel(join(path_save_r, 'spct_cm0.5_mean_rmse-z_by_r.xlsx'))
        dfssm.to_excel(join(path_save_r, 'spct_cm0.9_mean_rmse-z_by_r.xlsx'))

    # -------------------

    # -

    # -------------------
    # bin by radial and axial position

    # 2d-bin by r and z

    if plot_bin_r_z:

        if not os.path.exists(path_save_rz):
            os.makedirs(path_save_rz)

        # setup 2D binning
        z_trues = dfi.z_true.unique()
        r_bins = [150, 300, 450]

        columns_to_bin = ['r_microns', 'z_true']
        column_to_count = 'id'
        bins = [r_bins, z_trues]
        round_to_decimals = [1, 1]
        return_groupby = True
        plot_fit = False

        # compute 2D bin (r, z)
        dfim, dfistd = bin.bin_generic_2d(dfi, columns_to_bin, column_to_count, bins, round_to_decimals,
                                          min_counts_bin_rz, return_groupby)
        dfsm, dfsstd = bin.bin_generic_2d(dfs, columns_to_bin, column_to_count, bins, round_to_decimals,
                                          min_counts_bin_rz, return_groupby)
        dfssm, dfssstd = bin.bin_generic_2d(dfss, columns_to_bin, column_to_count, bins, round_to_decimals,
                                            min_counts_bin_rz, return_groupby)

        # compute rmse-z
        dfim['rmse_z'] = np.sqrt(dfim['rmse_z'])
        dfsm['rmse_z'] = np.sqrt(dfsm['rmse_z'])
        dfssm['rmse_z'] = np.sqrt(dfssm['rmse_z'])

        # resolve floating point bin selecting
        dfim = dfim.round({'bin_tl': 0, 'bin_ll': 1})
        dfistd = dfistd.round({'bin_tl': 0, 'bin_ll': 1})
        dfim = dfim.sort_values(['bin_tl', 'bin_ll'])
        dfistd = dfistd.sort_values(['bin_tl', 'bin_ll'])

        dfsm = dfsm.round({'bin_tl': 0, 'bin_ll': 1})
        dfsstd = dfsstd.round({'bin_tl': 0, 'bin_ll': 1})
        dfsm = dfsm.sort_values(['bin_tl', 'bin_ll'])
        dfsstd = dfsstd.sort_values(['bin_tl', 'bin_ll'])

        dfssm = dfssm.round({'bin_tl': 0, 'bin_ll': 1})
        dfssstd = dfssstd.round({'bin_tl': 0, 'bin_ll': 1})
        dfssm = dfssm.sort_values(['bin_tl', 'bin_ll'])
        dfssstd = dfssstd.sort_values(['bin_tl', 'bin_ll'])


        # compute final stats and package prior to exporting
        def package_for_export(df_):
            """ df = package_for_export(df_=df) """
            df_ = df_.rename(columns=
                             {'z_true': 'z_nominal',
                              'z_calib': 'z_assert_true',
                              'r': 'r_pixels',
                              'error_rel_p_calib': 'error_rel_z_assert_true',
                              'abs_error_rel_p_calib': 'abs_error_rel_z_assert_true'}
                             )
            df_ = df_.drop(columns=['frame', 'id', 'z_no_corr', 'x', 'y', 'z_plane',
                                    'tilt_x_degrees', 'tilt_y_degrees'])
            return df_


        dfim = package_for_export(df_=dfim)
        dfsm = package_for_export(df_=dfsm)
        dfssm = package_for_export(df_=dfssm)

        # export
        dfim.to_excel(join(path_save_rz, 'idpt_cm0.5_bin_r-z_rmse-z.xlsx'))
        dfsm.to_excel(join(path_save_rz, 'spct_cm0.5_bin_r-z_rmse-z.xlsx'))
        dfssm.to_excel(join(path_save_rz, 'spct_cm0.9_bin_r-z_rmse-z.xlsx'))

        # ---

        # plot
        clrs = ['black', 'blue', 'red']
        if plot_cmin_zero_nine:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches * 1, size_y_inches * 1.25))
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1, size_y_inches * 1))

        for i, bin_r in enumerate(dfim.bin_tl.unique()):
            dfibr = dfim[dfim['bin_tl'] == bin_r]
            ax1.plot(dfibr.bin_ll, dfibr['rmse_z'], '-o', ms=4, color=clrs[i], label=int(np.round(bin_r, 0)))

            dfsbr = dfsm[dfsm['bin_tl'] == bin_r]
            ax2.plot(dfsbr.bin_ll, dfsbr['rmse_z'], '-o', ms=4, color=clrs[i], label=int(np.round(bin_r, 0)))

            if plot_cmin_zero_nine:
                dfssbr = dfssm[dfssm['bin_tl'] == bin_r]
                ax3.plot(dfssbr.bin_ll, dfssbr['rmse_z'], '-o', ms=4, color=clrs[i], label=int(np.round(bin_r, 0)))

        ax1.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        ax1.set_ylim([0, 3.2])
        ax1.set_yticks([0, 1, 2, 3])
        ax1.legend(loc='upper center', ncol=3, title=r'$r^{\delta} \: (\mu m)$')  # ,  title=r'$r^{\delta}$')
        # , borderpad=0.25, handletextpad=0.6, borderaxespad=0.3, markerscale=0.75)

        ax2.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        ax2.set_ylim([0, 3.2])
        ax2.set_yticks([0, 1, 2, 3])

        if plot_cmin_zero_nine:
            ax3.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
            ax3.set_ylim([0, 3.2])
            ax3.set_yticks([0, 1, 2, 3])

            ax3.set_xlabel(r'$z \: (\mu m)$')
            ax3.set_xticks([-50, -25, 0, 25, 50])

            save_lbl = 'bin_r-z_rmse-z_by_r-z_all'

        else:
            ax2.set_xlabel(r'$z \: (\mu m)$')
            ax2.set_xticks([-50, -25, 0, 25, 50])

            save_lbl = 'bin_r-z_rmse-z_by_r-z'

        plt.tight_layout()
        plt.savefig(join(path_save_rz, save_lbl + '.png'))
        plt.show()
        plt.close()

        # ---

        # compute mean rmse-z per radial bin
        bins = [r_bins, 1]
        dfim, dfistd = bin.bin_generic_2d(dfi, columns_to_bin, column_to_count, bins, round_to_decimals,
                                          min_counts_bin_rz, return_groupby)
        dfsm, dfsstd = bin.bin_generic_2d(dfs, columns_to_bin, column_to_count, bins, round_to_decimals,
                                          min_counts_bin_rz, return_groupby)
        dfssm, dfssstd = bin.bin_generic_2d(dfss, columns_to_bin, column_to_count, bins, round_to_decimals,
                                            min_counts_bin_rz, return_groupby)

        dfim['rmse_z'] = np.sqrt(dfim['rmse_z'])
        dfsm['rmse_z'] = np.sqrt(dfsm['rmse_z'])
        dfssm['rmse_z'] = np.sqrt(dfssm['rmse_z'])


        # compute final stats and package prior to exporting
        def package_for_export(df_):
            """ df = package_for_export(df_=df) """
            df_ = df_.rename(columns=
                             {'r': 'r_pixels',
                              'error_rel_p_calib': 'error_rel_z_assert_true',
                              'abs_error_rel_p_calib': 'abs_error_rel_z_assert_true'}
                             )
            df_ = df_.drop(columns=['frame', 'id', 'z_no_corr', 'x', 'y', 'z_plane', 'z_true', 'z_calib',
                                    'tilt_x_degrees', 'tilt_y_degrees'])
            return df_


        dfim = package_for_export(df_=dfim)
        dfsm = package_for_export(df_=dfsm)
        dfssm = package_for_export(df_=dfssm)

        dfim.to_excel(join(path_save_rz, 'idpt_cm0.5_bin-rz_mean-rmse-z.xlsx'))
        dfsm.to_excel(join(path_save_rz, 'spct_cm0.5_bin-rz_mean-rmse-z.xlsx'))
        dfssm.to_excel(join(path_save_rz, 'spct_cm0.9_bin-rz_mean-rmse-z.xlsx'))

    # -------------------

    # -------------

    # per-particle rmse-z as a function of radial position

    if plot_bin_id:
        # save dir
        if not os.path.exists(path_save_id):
            os.makedirs(path_save_id)

        # setup
        px = 'r'
        pys = ['error_rel_p_calib', 'abs_error_rel_p_calib']
        py = pys[1]
        ms = 3

        # plot
        fig, ax = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches))

        dfes = [dfi, dfs, dfss]
        for dfe, mtd in zip(dfes, ['IDPT', 'SPCT_cmin0.5', 'SPCT_cmin0.9']):  # for py in pys:
            dfe = dfe.reset_index()
            dfe['error_squared'] = dfe[py] ** 2
            dfe = dfe.groupby('id').mean().reset_index()
            dfe['rmse_z'] = np.sqrt(dfe['error_squared'])
            dfe.to_excel(join(path_save_id, '{}_bin-id_rmse-z.xlsx'.format(mtd)))
            ax.scatter(dfe[px], dfe['rmse_z'], s=ms, label='{}: {}'.format(mtd, np.round(dfe['rmse_z'].mean(), 2)))

            popt, pcov = curve_fit(fit_line, dfe[px], dfe['rmse_z'])
            ax.plot(dfe[px], fit_line(dfe[px], *popt),
                    label=r'$\Delta \sigma_{z} / \Delta r=$' +
                          ' {} '.format(np.round(popt[0] * 1e3, 1)) +
                          r'$nm/\mu m$' + '\n' +
                          r'$\Delta \sigma_{z} / \Delta R=$' +
                          ' {} '.format(np.round(np.max(fit_line(dfe[px], *popt)) -
                                                 np.min(fit_line(dfe[px], *popt)), 1)) +
                          r'$\mu m$'
                    )

        ax.set_xlabel('r')
        ax.set_ylabel('rmse z')
        # ax.set_ylim([-0.2, 2.6])
        ax.legend(title='mean rmse-z', loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(join(path_save_id, 'compare_rmse_z_by_r_fit-line.png'))
        plt.show()
        plt.close()

    # ---

# ---

# ---

plot_custom_figs = False  # True False
if plot_custom_figs:

    plot_Cm_by_rz = False  # True False
    if plot_Cm_by_rz:
        path_save_rz = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_error_relative_calib_particle/' \
                       'results/relative-to-tilt-corr-calib-particle_08.06.23_raw-original/spct-is-raw/' \
                       'zerrlim5_cmin0.5_mincountsallframes1/ztrue_is_fit-plane-xyzc/' \
                       'Cmin=0.5/bin-r-z_zerrlim5_mincountsallframes1'
        fpi = 'idpt_cm0.5_bin_r-z_rmse-z.xlsx'
        fps = 'spct_cm0.5_bin_r-z_rmse-z.xlsx'

        dfim = pd.read_excel(join(path_save_rz, fpi))
        dfsm = pd.read_excel(join(path_save_rz, fps))

        plot_cmin_zero_nine = False

        # plot
        clrs = ['black', 'blue', 'red']
        if plot_cmin_zero_nine:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches * 1, size_y_inches * 1.25))
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1, size_y_inches * 1))

        for i, bin_r in enumerate(dfim.bin_tl.unique()):
            dfibr = dfim[dfim['bin_tl'] == bin_r]
            ax1.plot(dfibr.bin_ll, dfibr['cm'], '-o', ms=4, color=clrs[i], label=int(np.round(bin_r, 0)))

            dfsbr = dfsm[dfsm['bin_tl'] == bin_r]
            ax2.plot(dfsbr.bin_ll, dfsbr['cm'], '-o', ms=4, color=clrs[i], label=int(np.round(bin_r, 0)))

            if plot_cmin_zero_nine:
                dfssbr = dfssm[dfssm['bin_tl'] == bin_r]
                ax3.plot(dfssbr.bin_ll, dfssbr['cm'], '-o', ms=4, color=clrs[i], label=int(np.round(bin_r, 0)))

        ax1.set_ylabel(r'$C_{m}^{\delta}$')
        ax1.set_ylim([0.68, 1.02])
        ax1.set_yticks([0.8, 1.0])
        ax1.legend(loc='lower center', ncol=3, columnspacing=1.5,
                   title=r'$r^{\delta} \: (\mu m)$')  # ,  title=r'$r^{\delta}$')
        # , borderpad=0.25, handletextpad=0.6, borderaxespad=0.3, markerscale=0.75)

        ax2.set_ylabel(r'$C_{m}^{\delta}$')
        ax2.set_ylim([0.68, 1.02])
        ax2.set_yticks([0.8, 1.0])

        if plot_cmin_zero_nine:
            ax3.set_ylabel(r'$C_{m}^{\delta}$')
            # ax3.set_ylim([0, 3.2])
            # ax3.set_yticks([0, 1, 2, 3])

            ax3.set_xlabel(r'$z \: (\mu m)$')
            ax3.set_xticks([-50, -25, 0, 25, 50])

            save_lbl = 'bin_r-z_Cm_by_r-z_all'

        else:
            ax2.set_xlabel(r'$z \: (\mu m)$')
            ax2.set_xticks([-50, -25, 0, 25, 50])

            save_lbl = 'bin_r-z_Cm_by_r-z'

        plt.tight_layout()
        plt.savefig(join(path_save_rz, save_lbl + '.png'))
        plt.show()
        plt.close()

    # ---

    plot_spct_apparent_curvature_by_z = True  # True False
    if plot_spct_apparent_curvature_by_z:
        custom_path_coords = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_error_relative_calib_particle/' \
                             'results/relative-to-tilt-corr-calib-particle_08.06.23_raw-original/spct-is-raw/' \
                             'zerrlim5_cmin0.5_mincountsallframes1/ztrue_is_fit-plane-xyzc'
        fps = 'spct_error_relative_calib_particle_zerrlim5_cmin0.5_mincountsallframes1.xlsx'
        custom_path_save = join(custom_path_coords, 'apparent_surface_by_z')
        custom_path_save_z_by_r = join(custom_path_coords, 'apparent_surface_z_by_r')
        if not os.path.exists(custom_path_save):
            os.makedirs(custom_path_save)
        if not os.path.exists(custom_path_save_z_by_r):
            os.makedirs(custom_path_save_z_by_r)

        dfs = pd.read_excel(join(custom_path_coords, fps))

        eval_all_by_z_true = True  # evaluate particle positions from all three frames at each z true
        plot_z_by_r = False
        plot_z_by_r_with_fit = True
        if eval_all_by_z_true:
            zts = dfs['z_true'].unique()

            fit_As = []
            fit_rmses = []
            fit_r_squareds = []

            figg, axx = plt.subplots()
            clrs = iter(cm.Spectral_r(np.linspace(0, 1, len(zts))))
            norm = mpl.colors.Normalize(vmin=np.min(zts), vmax=np.max(zts))
            cmap = 'Spectral_r'

            for zt in zts:
                dfsz = dfs[dfs['z_true'] == zt]

                # fit spline to 'raw' data
                """bispl, rmse = fit.fit_3d_spline(x=dfsz['y'],
                                                y=dfsz['x'],
                                                z=dfsz['z'],
                                                kx=2,
                                                ky=2)
                print("fit bispl to raw, RMSE = {} microns".format(np.round(rmse, 3)))"""

                # plot z by r
                if plot_z_by_r:

                    fig, ax = plt.subplots()
                    ax.scatter(dfsz['r'] * microns_per_pixel, dfsz['z'], c=dfsz['id'])
                    ax.set_xlabel(r'$r \: (\mu m)$')
                    ax.set_ylabel(r'$z \: (\mu m)$')
                    plt.tight_layout()
                    plt.savefig(join(custom_path_save_z_by_r, 'z_by_r_znom={}.png'.format(np.round(zt, 1))))
                    plt.close()

                # plot z by r
                if plot_z_by_r_with_fit:
                    z_true_assigned = dfsz['z_calib'].mean()

                    def fit_parabola(x, a):
                        return a * x ** 2 + z_true_assigned

                    pr = dfsz['r'].to_numpy() * microns_per_pixel
                    pz = dfsz['z'].to_numpy()

                    popt, pcov = curve_fit(fit_parabola, pr, pz)
                    rmse, r_squared = fit.calculate_fit_error(fit_results=fit_parabola(pr, *popt), data_fit_to=pz)

                    fit_As.append(popt[0])
                    fit_rmses.append(rmse)
                    fit_r_squareds.append(r_squared)

                    fr = np.linspace(0, np.max(pr))
                    fz = fit_parabola(fr, *popt)

                    # plot
                    """fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches))
                    ax.scatter(dfsz['r'] * microns_per_pixel, dfsz['z'], color='k', s=5, label='SPCT: ' + r'$z_{i}$')
                    ax.plot(fr, fz, color='r', label='Fit: ' + r'$z_{surface}$')
                    ax.axhline(z_true_assigned, color='gray', linestyle='--', label='Substrate: ' + r'$z_{true}$')
                    ax.set_xlabel(r'$r \: (\mu m)$')
                    ax.set_ylabel(r'$z \: (\mu m)$')
                    ax.legend(loc='lower left')
                    plt.tight_layout()
                    plt.savefig(join(custom_path_save_z_by_r, 'z_by_r_rmse={}_znom={}.png'.format(np.round(rmse, 2),
                                                                                                  np.round(zt, 1))))
                    plt.close()"""

                    # add plot to outer figure
                    p1, = axx.plot(fr, fz - z_true_assigned, color=next(clrs),  # norm(zt), #
                                   label=int(np.round(zt, 0)))

            print("Mean rmse = {} microns".format(np.round(np.mean(fit_rmses), 2)))
            print("Mean R squared = {}".format(np.round(np.mean(fit_r_squareds), 2)))

            # outer figure
            axx.set_xlabel(r'$r \: (\mu m)$')
            axx.set_ylabel(r'$z_{surface} - z_{true} \: (\mu m)$')
            # axx.legend(loc='upper left', bbox_to_anchor=(1, 1))

            # color bar
            figg.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axx, label=r'$z \: (\mu m)$')

            plt.tight_layout()
            plt.savefig(join(custom_path_save, 'fit-curvature_by_z_cbar.png'))
            # plt.show()
            plt.close()

            # --
            fig, ax = plt.subplots()
            ax.plot(zts, fit_As, '-o', label='fit: ' + r'$ax^2+z_{true}$')
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_xticks([-50, -25, 0, 25, 50])
            ax.set_ylabel(r'$a$')
            ax.legend()
            plt.tight_layout()
            plt.savefig(join(custom_path_save, 'fit-parabola_by_z.png'))
            # plt.show()
            plt.close()

    # ---

# ---

# ---

plot_histogram_of_errors_relative_calib_particle = False  # True False
if plot_histogram_of_errors_relative_calib_particle:

    # ---

    spct_is = 'spct-is-raw'  # 'spct-is-raw' or 'spct-is-corr-fc'

    # read paths
    fpi = join(base_dir, 'results',
               'relative-to-tilt-corr-calib-particle_08.06.23_raw-original',
               spct_is,
               'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
               'ztrue_is_fit-plane-xyzc',
               'idpt_error_relative_calib_particle_' +
               'zerrlim{}_cmin0.5_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))
    fps = join(base_dir, 'results',
               'relative-to-tilt-corr-calib-particle_08.06.23_raw-original',
               spct_is,
               'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
               'ztrue_is_fit-plane-xyzc',
               'spct_error_relative_calib_particle_' +
               'zerrlim{}_cmin0.5_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))
    fpss = join(base_dir, 'results',
                'relative-to-tilt-corr-calib-particle_08.06.23_raw-original',
                spct_is,
                'zerrlim{}_cmin0.9_mincountsallframes{}'.format(z_error_limit, min_counts),
                'ztrue_is_fit-plane-xyzc',
                'spct_error_relative_calib_particle_' +
                'zerrlim{}_cmin0.9_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))

    # save paths
    path_save_hist_z = join(base_dir,
                            'results',
                            'relative-to-tilt-corr-calib-particle_08.06.23_raw-original',
                            spct_is,
                            'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                            'ztrue_is_fit-plane-xyzc',
                            'hist-zerr')

    if not os.path.exists(path_save_hist_z):
        os.makedirs(path_save_hist_z)

    # ---

    dfi = pd.read_excel(fpi)

    # histogram of z-errors
    error_col = 'error_rel_p_calib'
    binwidth_y = 0.1
    bandwidth_y = None  # 0.125
    xlim = 3
    ylim_top = 1000
    yticks = [0, 500, 1000]

    # iterate

    for estimate_kde in [False, True]:
        for df, mtd, mcm in zip([dfi], ['idpt'], [0.5]):
            y = df[error_col].to_numpy()

            # plot
            fig, ax = plt.subplots(figsize=(size_x_inches / 1.5, size_y_inches / 1.5))

            ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y  # + binwidth_y
            ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y - binwidth_y
            ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)
            ny, binsy, patchesy = ax.hist(y, bins=ybins, orientation='vertical', color='gray', zorder=2.5)

            ax.set_xlabel(r'$\epsilon_{z} \: (\mu m)$')
            ax.set_xlim([-xlim, xlim])
            ax.set_ylabel('Counts')
            ax.set_ylim([0, ylim_top])
            ax.set_yticks(yticks)

            if estimate_kde:
                kdex, kdey, bandwidth = fit_kde(y, bandwidth=bandwidth_y)
                # pdf, y_grid = kde_scipy(y, y_grid=None, bandwidth=bandwidth)

                axr = ax.twinx()

                axr.plot(kdex, kdey, linewidth=0.5, color='r', zorder=2.4)
                # axr.plot(y_grid, pdf, linewidth=0.5, linestyle='--', color='b', zorder=2.4)

                axr.set_ylabel('PDF')
                axr.set_ylim(bottom=0)
                # axr.set_yticks(yticks)
                save_id = 'idpt_cmin{}_histogram_z-errors_kde-bandwidth={}.png'.format(mcm, np.round(bandwidth, 4))
            else:
                save_id = 'idpt_cmin{}_histogram_z-errors.png'.format(mcm)

            plt.tight_layout()
            plt.savefig(join(path_save_hist_z, save_id))
            plt.show()
            plt.close()

    # ---

    # ---

    # SPCT

    # read coords
    dfs = pd.read_excel(fps)
    dfss = pd.read_excel(fpss)

    # ---

    # histogram of z-errors
    binwidth_y = 0.3
    bandwidth_y = None  # 0.25
    xlim = 5.5
    ylim_top = 300
    yticks = [0, 150, 300]

    # iterate

    for estimate_kde in [False, True]:
        for df, mtd, mcm in zip([dfs, dfss], ['spct', 'spct'], [0.5, 0.9]):
            y = df[error_col].to_numpy()

            # plot
            fig, ax = plt.subplots(figsize=(size_x_inches / 1.5, size_y_inches / 1.5))

            ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y  # + binwidth_y
            ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y - binwidth_y
            ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)
            ny, binsy, patchesy = ax.hist(y, bins=ybins, orientation='vertical', color='gray', zorder=2.5)

            ax.set_xlabel(r'$\epsilon_{z} \: (\mu m)$')
            ax.set_xlim([-xlim, xlim])
            ax.set_ylabel('Counts')
            ax.set_ylim([0, ylim_top])
            ax.set_yticks(yticks)

            if estimate_kde:
                kdex, kdey, bandwidth = fit_kde(y, bandwidth=bandwidth_y)
                # pdf, y_grid = kde_scipy(y, y_grid=None, bandwidth=bandwidth)

                axr = ax.twinx()

                axr.plot(kdex, kdey, linewidth=0.5, color='r', zorder=2.4)
                # axr.plot(y_grid, pdf, linewidth=0.5, linestyle='--', color='b', zorder=2.4)

                axr.set_ylabel('PDF')
                axr.set_ylim(bottom=0)
                # axr.set_yticks(yticks)
                save_id = 'spct_cmin{}_histogram_z-errors_kde-bandwidth={}.png'.format(mcm, np.round(bandwidth, 4))
            else:
                save_id = 'spct_cmin{}_histogram_z-errors.png'.format(mcm)

            plt.tight_layout()
            plt.savefig(join(path_save_hist_z, save_id))
            plt.show()
            plt.close()

    # ---

    # plot histogram of SPCT (Cm = 0.9) on top of (Cm = 0.5)

    # plot
    fig, ax = plt.subplots(figsize=(size_x_inches / 1.5, size_y_inches / 1.5))

    for df, mtd, mcm, clr, alpha in zip([dfs, dfss], ['spct', 'spct'], [0.5, 0.9], [scigreen, sciorange], [1, 0.8]):
        y = df[error_col].to_numpy()

        ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y  # + binwidth_y
        ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y - binwidth_y
        ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)
        ny, binsy, patchesy = ax.hist(y, bins=ybins, orientation='vertical', color=clr, alpha=alpha, label=mcm)

    ax.set_xlabel(r'$\epsilon_{z} \: (\mu m)$')
    ax.set_xlim([-xlim, xlim])
    ax.set_ylabel('Counts')
    ax.set_ylim([0, ylim_top])
    ax.set_yticks(yticks)
    ax.legend(title=r'$C_{m,min}$')
    plt.tight_layout()
    plt.savefig(join(path_save_hist_z, 'spct_cmin0.5-0.9_histogram_z-errors.png'))
    plt.show()
    plt.close()

# ---

# ---

plot_histogram_of_errors_relative_calib_particle_precision = False  # True False
if plot_histogram_of_errors_relative_calib_particle_precision:

    """
    "The corresponding position uncertainty was determined from the root-mean-square displacement of each nanoparticle
    between successive images divided by 2 , which is equivalent to a pooled standard deviation of successive image
    pairs (ISO Technical Advisory Group 4, 1995)."
    
    Ref: C. MCGRAY, C.R. COPELAND, S.M. STAVIS, & J. GEIST, Centroid precision and orientation precision of planar 
    localization microscopy, Journal of Microscopy, Vol. 263, Issue 3 2016, pp. 238–249
    """

    # save paths
    path_save_hist_z = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                            'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                            'hist-zprecision_by_rms-displacement')

    if not os.path.exists(path_save_hist_z):
        os.makedirs(path_save_hist_z)


    def flatten(l):
        return [item for sublist in l for item in sublist]


    # ---

    # IDPT
    hist_precision_idpt = False  # True False
    if hist_precision_idpt:

        fpi = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                   'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                   'idpt_error_relative_calib_particle_' +
                   'zerrlim{}_cmin0.5_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))
        dfi = pd.read_excel(fpi)

        # scale to microns
        dfi['x'] = dfi['x'] * microns_per_pixel
        dfi['y'] = dfi['y'] * microns_per_pixel

        # histogram of z-errors
        z_trues = dfi['z_true'].unique()
        spatial_dims = ['z', 'x', 'y']
        binwidth_ys = [0.0235, 0.01, 0.01]  # good for zooming in close: [0.0235, 0.005, 0.005]
        bandwidth_y = None
        xlim_lefts = [-0.05, -0.015, -0.015]  #
        xlim_rights = [1, 0.3, 0.3]  # [1, 0.3, 0.3]
        ylim_top = 900
        yticks = [0, 400, 800]

        # iterate
        estimate_kde = False  # True False

        for spatial_dim, binwidth_y, xlim_left, xlim_right in zip(spatial_dims, binwidth_ys, xlim_lefts, xlim_rights):

            for df, mtd, mcm in zip([dfi], ['idpt'], [0.5]):

                zpid_displacements = []
                zpid_stds = []
                for z_true in z_trues:
                    dfz = df[df['z_true'] == z_true]
                    z_pids = dfz['id'].unique()

                    for pid in z_pids:
                        dfzpid = dfz[dfz['id'] == pid]
                        if len(dfzpid) > 1:
                            # root mean squared displacement
                            zpid_displacement = dfzpid.diff()[spatial_dim].abs().tolist()
                            zpid_displacement = zpid_displacement[1:]
                            zpid_displacements.append(zpid_displacement)

                            # standard deviation of successive image pairs
                            zpid_stds.append(dfzpid[spatial_dim].std())

                y = np.array(flatten(zpid_displacements)) / np.sqrt(2)
                yy = np.array(zpid_stds)

                # plot
                fig, ax = plt.subplots(figsize=(size_x_inches / 1.35, size_y_inches / 1.35))

                if spatial_dim == 'z':
                    ybins = np.arange(np.min(y), xlim_right, binwidth_y)
                else:
                    ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y  # + binwidth_y
                    ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y - binwidth_y
                    ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)

                ny, binsy, patchesy = ax.hist(y, bins=ybins, orientation='vertical', color='gray', zorder=2.5,
                                              label='r.m.s.d. = {}'.format(np.round(np.mean(y), 3)))
                nyy, binsyy, patchesyy = ax.hist(yy, bins=ybins, orientation='vertical', color='blue',
                                                 alpha=0.5, zorder=2.7,
                                                 label='std = {}'.format(np.round(np.mean(yy), 3)))

                if spatial_dim == 'z':
                    xlbl = r'$r.m.s. \Delta z \: (\mu m)$'
                elif spatial_dim == 'x':
                    xlbl = r'$r.m.s. \Delta x \: (\mu m)$'
                elif spatial_dim == 'y':
                    xlbl = r'$r.m.s. \Delta y \: (\mu m)$'
                else:
                    raise ValueError()

                ax.set_xlabel(xlbl)
                # ax.set_xlim(left=xlim_left, right=xlim_right)
                ax.set_ylabel('Counts')
                # ax.set_ylim([0, ylim_top])
                # ax.set_yticks(yticks)
                ax.legend(loc='upper right', title='mean()')

                if estimate_kde:
                    kdex, kdey, bandwidth = fit_kde(yy, bandwidth=bandwidth_y)
                    # pdf, y_grid = kde_scipy(y, y_grid=None, bandwidth=bandwidth)

                    axr = ax.twinx()

                    axr.plot(kdex, kdey, linewidth=0.5, color='b', zorder=2.4, label=np.round(kdex[np.argmax(kdey)], 3))
                    # axr.plot(y_grid, pdf, linewidth=0.5, linestyle='--', color='b', zorder=2.4)

                    axr.set_ylabel('PDF')
                    axr.set_ylim(bottom=0)
                    # axr.set_yticks(yticks)
                    axr.legend(loc='lower right', title='argmax(std)')
                    save_id = 'idpt_cmin{}_histogram_{}-precision_by_rms-displacement_kde-bandwidth={}.png'.format(mcm,
                                                                                                                   spatial_dim,
                                                                                                                   np.round(
                                                                                                                       bandwidth,
                                                                                                                       4))
                else:
                    save_id = 'idpt_cmin{}_histogram_{}-precision_by_rms-displacement_full-scale.png'.format(mcm,
                                                                                                             spatial_dim)

                plt.tight_layout()
                plt.savefig(join(path_save_hist_z, save_id))
                plt.show()
                plt.close()

        # ---

    # ---

    # ---

    # SPCT
    hist_precision_spct = False  # True False
    if hist_precision_spct:

        fps = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                   'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                   'spct_error_relative_calib_particle_' +
                   'zerrlim{}_cmin0.5_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))
        fpss = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                    'zerrlim{}_cmin0.9_mincountsallframes{}'.format(z_error_limit, min_counts),
                    'spct_error_relative_calib_particle_' +
                    'zerrlim{}_cmin0.9_mincountsallframes{}.xlsx'.format(z_error_limit, min_counts))

        # read coords
        dfs = pd.read_excel(fps)
        dfss = pd.read_excel(fpss)

        # scale to microns
        dfs['x'] = dfs['x'] * microns_per_pixel
        dfs['y'] = dfs['y'] * microns_per_pixel
        dfss['x'] = dfss['x'] * microns_per_pixel
        dfss['y'] = dfss['y'] * microns_per_pixel

        # ---

        # histogram of z-errors
        spatial_dims = ['z', 'x', 'y']
        z_trues = dfs['z_true'].unique()
        binwidth_ys = [0.025, 0.01, 0.01]
        bandwidth_y = None
        xlims = [-0.05, 1]
        xlim_left = -0.05
        xlim_rights = [1, 0.25, 0.25]
        ylim_top = 400
        yticks = [0, 200, 400]

        binwidth_ys = [0.0235, 0.005, 0.005]
        bandwidth_y = None
        xlim_lefts = [-0.05, -0.02, -0.02]  # [-0.05, -0.02, -0.02]
        xlim_rights = [1, 0.3, 0.3]  # [1, 0.3, 0.3]

        # iterate
        estimate_kde = False

        for spatial_dim, binwidth_y, xlim_left, xlim_right in zip(spatial_dims, binwidth_ys, xlim_lefts, xlim_rights):

            for df, mtd, mcm in zip([dfs, dfss], ['spct', 'spct'], [0.5, 0.9]):

                zpid_displacements = []
                zpid_stds = []
                for z_true in z_trues:
                    dfz = df[df['z_true'] == z_true]
                    z_pids = dfz['id'].unique()

                    for pid in z_pids:
                        dfzpid = dfz[dfz['id'] == pid]
                        if len(dfzpid) > 1:
                            # root mean squared displacement
                            zpid_displacement = dfzpid.diff()[spatial_dim].abs().tolist()
                            zpid_displacement = np.array(zpid_displacement[1:])

                            # limit in-plane displacements to <3 microns
                            zpid_displacement = zpid_displacement[zpid_displacement < 3]
                            if len(zpid_displacement) > 0:
                                zpid_displacements.append(zpid_displacement)

                            # standard deviation of successive image pairs
                            zpid_stds.append(dfzpid[spatial_dim].std())

                y = np.array(flatten(zpid_displacements)) / np.sqrt(2)
                yy = np.array(zpid_stds)

                # filter --> only useful if trying to zoom in on the KDE Gaussian peak.
                # y = y[y < xlim_right]
                # yy = yy[yy < xlim_right]

                # plot
                fig, ax = plt.subplots(figsize=(size_x_inches / 1.5, size_y_inches / 1.5))

                if spatial_dim == 'z':
                    ybins = np.arange(np.min(y), xlim_right, binwidth_y)
                else:
                    ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y  # + binwidth_y
                    ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y - binwidth_y
                    ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)

                ny, binsy, patchesy = ax.hist(y, bins=ybins, orientation='vertical', color='gray', zorder=2.5,
                                              label='r.m.s.d. = {}'.format(np.round(np.mean(y), 3)))
                nyy, binsyy, patchesyy = ax.hist(yy, bins=ybins, orientation='vertical', color='blue', alpha=0.25,
                                                 zorder=2.7,
                                                 label='std = {}'.format(np.round(np.mean(yy), 3)))

                if spatial_dim == 'z':
                    xlbl = r'$r.m.s. \Delta z \: (\mu m)$'
                elif spatial_dim == 'x':
                    xlbl = r'$r.m.s. \Delta x \: (\mu m)$'
                elif spatial_dim == 'y':
                    xlbl = r'$r.m.s. \Delta y \: (\mu m)$'
                else:
                    raise ValueError()

                ax.set_xlabel(xlbl)
                # ax.set_xlim(left=xlim_left, right=xlim_right)
                ax.set_ylabel('Counts')
                # ax.set_ylim([0, ylim_top])
                # ax.set_yticks(yticks)
                ax.legend()

                if estimate_kde:
                    kdex, kdey, bandwidth = fit_kde(yy, bandwidth=bandwidth_y)
                    # pdf, y_grid = kde_scipy(y, y_grid=None, bandwidth=bandwidth)

                    axr = ax.twinx()

                    axr.plot(kdex, kdey, linewidth=0.5, color='b', zorder=2.4, label=np.round(kdex[np.argmax(kdey)], 3))
                    # axr.plot(y_grid, pdf, linewidth=0.5, linestyle='--', color='b', zorder=2.4)

                    axr.set_ylabel('PDF')
                    axr.set_ylim(bottom=0)
                    # axr.set_yticks(yticks)
                    axr.legend(loc='lower right', title='argmax(std)')
                    save_id = 'spct_cmin{}_histogram_{}-precision_by_rms-displacement_kde-bandwidth={}' \
                              '.png'.format(mcm, spatial_dim, np.round(bandwidth, 4))
                else:
                    save_id = 'spct_cmin{}_histogram_{}-precision_by_rms-displacement_full-scale.png'.format(mcm,
                                                                                                             spatial_dim)

                plt.tight_layout()
                plt.savefig(join(path_save_hist_z, save_id))
                plt.show()
                plt.close()

        # ---

    # ---

# ---

# ---

plot_compare_fit_plane_idpt_spct_by_z_nominal = False
if plot_compare_fit_plane_idpt_spct_by_z_nominal:

    # save path
    path_results_tilt_corr = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle',
                                  'zerrlim{}_cmin{}_mincountsallframes{}'.format(z_error_limit, min_cm, min_counts),
                                  'ztrue_is_fit-plane-xyzc')
    path_compare_scatter_zy_by_z = join(path_results_tilt_corr, 'compare_scatter_zy_by_z', 'both')
    if not os.path.exists(path_compare_scatter_zy_by_z):
        os.makedirs(path_compare_scatter_zy_by_z)

    # ---

    # plot fit_plane_image_xyzc (the z-position at the center of the image) and rmse-z as a function of z_true
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.25, size_y_inches))

    dfs = []
    for i, method in enumerate(['idpt', 'spct']):
        path_scatter_zy_by_z = join(path_results_tilt_corr, '{}_scatter_zy_by_z'.format(method))
        df_fit_plane = pd.read_excel(join(path_scatter_zy_by_z, 'fit-plane-xyzc-rmsez_by_z-true.xlsx'))
        # columns=['z_nominal', 'z_xyc', 'rmsez', 'z_diff'; which equals 'z_xyc' - 'z_nominal']
        df_fit_plane['mtd'] = i
        dfs.append(df_fit_plane)

        z_total_displacement = df_fit_plane['z_xyc'].iloc[-1] - df_fit_plane['z_xyc'].iloc[0]

        ax1.plot(df_fit_plane['z_nominal'], df_fit_plane['z_diff'], '-o', label=np.round(z_total_displacement, 2))
        ax2.plot(df_fit_plane['z_nominal'], df_fit_plane['rmsez'], '-o', label=method)

    ax1.set_ylabel(r'$z_{nom} - z_{xyc} \: (\mu m)$')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$\Delta z_{total} \: (\mu m)$')
    ax2.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    ax2.set_xlabel(r'$z_{nominal}$')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(join(path_compare_scatter_zy_by_z, 'compare_fit-plane-xyzc-rmsez_by_z-true.png'))
    plt.show()
    plt.close()

    # ---
    # export combined dataframe
    dfs = pd.concat(dfs)
    dfs.to_excel(join(path_compare_scatter_zy_by_z, 'compare_fit-plane-xyzc-rmsez_by_z-true.xlsx'))

    # ---

# ---

# ---

# for IDPT and SPCT - SIMULTANEOUSLY
# for each true axial position, correct plane tilt and then calculate error relative to the calibration particle
analyze_idpt_and_spct_error_relative_calib_post_tilt_corr = False  # False True
if analyze_idpt_and_spct_error_relative_calib_post_tilt_corr:

    min_cm = 0.5  # 0.5 0.9
    correct_tilt = True  # False True
    correct_spct_tilt_using_idpt_fit_plane = True
    assign_z_true_to_fit_plane_xyzc = True

    plot_tilt_per_frame_before_removing_large_z_errors = False
    plot_tilt_per_frame_after_removing_large_z_errors = False
    plot_tilt_per_frame = False  # False True  # I think this should always be False because it's deprecated
    plot_3d_tilt_per_frame = False  # False True

    # save path
    path_results_tilt_corr = join(base_dir, 'results', 'relative-to-tilt-corr-calib-particle_08.06.23_raw-original',
                                  'zerrlim{}_cmin{}_mincountsallframes{}'.format(z_error_limit, min_cm, min_counts))
    if assign_z_true_to_fit_plane_xyzc:
        path_results_tilt_corr = join(path_results_tilt_corr, 'ztrue_is_fit-plane-xyzc')
    elif correct_spct_tilt_using_idpt_fit_plane:
        path_results_tilt_corr = join(path_results_tilt_corr, 'corr-tilt-by-fit-idpt')
    path_scatter_zy_by_z = join(path_results_tilt_corr, 'compare_scatter_zy_by_z', 'both')

    if not os.path.exists(path_results_tilt_corr):
        os.makedirs(path_results_tilt_corr)
    if not os.path.exists(path_scatter_zy_by_z):
        os.makedirs(path_scatter_zy_by_z)
    """
    if not os.path.exists(join(path_results_tilt_corr, 'corr-tilt-by-fit-idpt')):
        os.makedirs(join(path_results_tilt_corr, 'corr-tilt-by-fit-idpt'))
    if not os.path.exists(join(path_results_tilt_corr, 'corr-tilt-by-fit-idpt', 'ztrue_is_fit-plane-xyzc')):
        os.makedirs(join(path_results_tilt_corr, 'corr-tilt-by-fit-idpt', 'ztrue_is_fit-plane-xyzc'))
    """

    # ---

    # shared
    padding = 5
    padding_rel_true_x = 0
    padding_rel_true_y = 0

    # spct
    s_test_id = 1
    s_test_name = 'test_coords_particle_image_stats_spct-1_aligned-corr-fc'  # dzf-post-processed_raw'  # _corr-fc
    s_calib_id_from_testset = 92
    s_calib_id_from_calibset = 46
    s_calib_baseline_frame = 12  # NOTE: baseline frame was 'calib_13.tif' but output coords always begin at frame = 0.
    s_xsub, s_ysub, s_rsub = 'gauss_xc', 'gauss_yc', 'gauss_rc'

    # idpt
    i_test_id = 19
    i_test_name = 'test_coords_particle_image_stats_tm16_cm19_aligned'  # _dzf-post-processed'
    i_calib_id_from_testset = 42
    i_calib_id_from_calibset = 42
    i_xsub, i_ysub, i_rsub = 'xg', 'yg', 'rg'

    # ---

    # read
    fpi = join(base_dir, 'coords', i_test_name + '.xlsx')
    fps = join(base_dir, 'coords', s_test_name + '.xlsx')

    dfi = pd.read_excel(fpi)
    dfs = pd.read_excel(fps)

    # filter 1. filter by number of counts
    dfic = dfi.groupby('id').count().reset_index()
    exclude_ids = dfic[dfic['z'] < min_counts]['id'].to_numpy()
    dfi = dfi[~dfi['id'].isin(exclude_ids)]

    dfsc = dfs.groupby('id').count().reset_index()
    exclude_ids = dfsc[dfsc['z'] < min_counts]['id'].to_numpy()
    dfs = dfs[~dfs['id'].isin(exclude_ids)]

    # filter 2. filter by Cm
    dfi = dfi[dfi['cm'] > min_cm]
    dfs = dfs[dfs['cm'] > min_cm]

    # filter by axial position
    # df = df[df['z_true'].abs() > 7.5]

    if i_rsub not in dfi.columns:
        dfi[i_rsub] = np.sqrt((img_xc - dfi[i_xsub]) ** 2 + (img_yc - dfi[i_ysub]) ** 2)

    if s_rsub not in dfs.columns:
        dfs[s_rsub] = np.sqrt((img_xc - dfs[s_xsub]) ** 2 + (img_yc - dfs[s_ysub]) ** 2)

    """
    Only if you're processing actually raw coordinates
    if dfi['z'].max() > 65:
        dfi['z'] = dfi['z'] - z_zero_from_calibration

    if dfs['z'].max() > 65:
        dfs['z'] = dfs['z'] - z_zero_from_calibration
    """

    # -

    # get only necessary columns
    dfi = dfi[['frame', 'id', 'cm', 'z', 'z_true', i_xsub, i_ysub, i_rsub]]  # , 'x', 'y', 'r'
    dfi = dfi.rename(columns={i_xsub: 'x', i_ysub: 'y', i_rsub: 'r'})  # do this to fit plane to sub-positions

    dfs = dfs[['frame', 'id', 'cm', 'z', 'z_true', s_xsub, s_ysub, s_rsub]]  # , 'x', 'y', 'r'
    dfs = dfs.rename(columns={s_xsub: 'x', s_ysub: 'y', s_rsub: 'r'})  # do this to fit plane to sub-positions

    # -

    # setup
    if min_cm == 0.9:
        z_trues = dfs.z_true.unique()
    else:
        z_trues = dfi.z_true.unique()

    # iterate through each z_true position
    dfis = []
    dfss = []

    i_fit_plane_img_xyzc = []
    s_fit_plane_img_xyzc = []

    i_fit_plane_rmsez = []
    s_fit_plane_rmsez = []

    """ NOTE: in order to exclude measurements with large xy-errors, we must import the SPCT coords processed and 
    outputted by mfig_11.06.21_calc_rigid_body_displacement.py (call them, dfs_filter_xy_errors) and, for each z_true, 
    discard particles in dfs (here) that are not in dfs_filter_xy_errors.
     
     NOTE: the programs are setup in a circular manner so I don't know which is supposed to be calculated first. I'm 
     going to continue without removing larger xy-errors. 
     """
    # fp_filter_xy_errors = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/' \
    #                      '11.06.21_error_relative_calib_particle/pre_08.06.23/coords/' \
    #                      'test_coords_particle_image_stats_spct-1_dzf-post-processed.xlsx'
    # dfs_filter_xy_errors = pd.read_excel()

    for z_true in z_trues:
        # clear z_calib
        z_calib = None
        i_z_calib = None
        s_z_calib = None

        dfit = dfi[dfi['z_true'] == z_true]
        dfst = dfs[dfs['z_true'] == z_true]

        # --- correct tilt
        if correct_tilt:
            # step 0. filter dft such that it only includes particles that could reasonably be on the tilt surface
            reasonable_z_tilt_limit = 3.25
            reasonable_r_tilt_limit = int(np.round(250 / microns_per_pixel))  # convert units microns to pixels
            dfit_within_tilt = dfit[np.abs(dfit['z'] - z_true) < reasonable_z_tilt_limit]
            dfst_within_tilt = dfst[np.abs(dfst['z'] - z_true) < reasonable_z_tilt_limit]

            # step 0.5. check if calibration particle is in this new group
            if not i_calib_id_from_testset in dfit_within_tilt.id.unique():
                # z_calib = dft_within_tilt[dft_within_tilt['r'] < reasonable_r_tilt_limit].z.mean()
                z_calib = -3.6

            # step 1. fit plane to particle positions
            i_dict_fit_plane = fit_in_focus_plane(df=dfit_within_tilt,  # note: x,y units are pixels at this point
                                                  param_zf='z',
                                                  microns_per_pixel=microns_per_pixel,
                                                  img_xc=img_xc,
                                                  img_yc=img_yc)
            s_dict_fit_plane = fit_in_focus_plane(df=dfst_within_tilt,  # note: x,y units are pixels at this point
                                                  param_zf='z',
                                                  microns_per_pixel=microns_per_pixel,
                                                  img_xc=img_xc,
                                                  img_yc=img_yc)

            i_fit_plane_img_xyzc.append(i_dict_fit_plane['z_f_fit_plane_image_center'])
            i_fit_plane_rmsez.append(i_dict_fit_plane['rmse'])
            s_fit_plane_img_xyzc.append(s_dict_fit_plane['z_f_fit_plane_image_center'])
            s_fit_plane_rmsez.append(s_dict_fit_plane['rmse'])

            if correct_spct_tilt_using_idpt_fit_plane:
                s_dict_fit_plane = i_dict_fit_plane

            # step 2. correct coordinates using fitted plane
            dfit['z_plane'] = functions.calculate_z_of_3d_plane(dfit.x, dfit.y, popt=i_dict_fit_plane['popt_pixels'])
            dfit['z_plane'] = dfit['z_plane'] - i_dict_fit_plane['z_f_fit_plane_image_center']
            dfit['z_corr'] = dfit['z'] - dfit['z_plane']

            dfst['z_plane'] = functions.calculate_z_of_3d_plane(dfst.x, dfst.y, popt=s_dict_fit_plane['popt_pixels'])
            dfst['z_plane'] = dfst['z_plane'] - s_dict_fit_plane['z_f_fit_plane_image_center']
            dfst['z_corr'] = dfst['z'] - dfst['z_plane']

            # add column for tilt
            dfit['tilt_x_degrees'] = i_dict_fit_plane['tilt_x_degrees']
            dfit['tilt_y_degrees'] = i_dict_fit_plane['tilt_y_degrees']
            dfst['tilt_x_degrees'] = s_dict_fit_plane['tilt_x_degrees']
            dfst['tilt_y_degrees'] = s_dict_fit_plane['tilt_y_degrees']

            # rename
            dfit = dfit.rename(columns={'z': 'z_no_corr'})
            dfit = dfit.rename(columns={'z_corr': 'z'})
            dfst = dfst.rename(columns={'z': 'z_no_corr'})
            dfst = dfst.rename(columns={'z_corr': 'z'})

            # plot before outlier removal
            if plot_tilt_per_frame_before_removing_large_z_errors:
                fig, (axx, ax) = plt.subplots(ncols=2, sharey=True,
                                              figsize=(size_x_inches * 2.75, size_y_inches * 1))

                axx.scatter(dfit['x'], dfit['z'], s=2, color='dodgerblue', label='i t.c.')
                axx.scatter(dfit['x'], dfit['z_no_corr'], s=2, color='blue', label='i raw')

                axx.scatter(dfst['x'], dfst['z'], s=2, color='lime', label='s t.c.')
                axx.scatter(dfst['x'], dfst['z_no_corr'], s=2, color='green', label='s raw')

                # axx.scatter(dfit_within_tilt['x'], dfit_within_tilt['z'], s=1, marker='.', color='k', label='raw-fitted')

                # ---
                # plot the calibration particle
                axx.scatter(dfit[dfit['id'] == i_calib_id_from_testset]['x'],
                            dfit[dfit['id'] == i_calib_id_from_testset]['z'],
                            marker='*', s=8, color='red', label=r'$p_{cal, IDPT}$')
                axx.scatter(dfst[dfst['id'] == s_calib_id_from_testset]['x'],
                            dfst[dfst['id'] == s_calib_id_from_testset]['z'],
                            marker='d', s=8, color='purple', label=r'$p_{cal, SPCT}$')

                # ---

                ax.scatter(dfit['y'], dfit['z'], s=2, color='dodgerblue', label='i t.c.')
                ax.scatter(dfit['y'], dfit['z_no_corr'], s=2, color='blue', label='i raw')

                ax.scatter(dfst['y'], dfst['z'], s=2, color='lime', label='s t.c.')
                ax.scatter(dfst['y'], dfst['z_no_corr'], s=2, color='green', label='s raw')

                # ax.scatter(dft_within_tilt['y'], dft_within_tilt['z'], s=1, marker='.', color='k', label='raw-fitted')

                # ---
                # plot the calibration particle
                ax.scatter(dfit[dfit['id'] == i_calib_id_from_testset]['y'],
                           dfit[dfit['id'] == i_calib_id_from_testset]['z'],
                           marker='*', s=8, color='red', label=r'$p_{cal, IDPT}$')
                ax.scatter(dfst[dfst['id'] == s_calib_id_from_testset]['y'],
                           dfst[dfst['id'] == s_calib_id_from_testset]['z'],
                           marker='d', s=8, color='purple', label=r'$p_{cal, SPCT}$')

                # ---

                if z_calib is not None:
                    axx.plot([img_xc - reasonable_r_tilt_limit, img_xc + reasonable_r_tilt_limit],
                             [z_calib, z_calib],
                             color='k', label='avg(r<150um)={}'.format(np.round(z_calib, 1)))
                    ax.plot([img_yc - reasonable_r_tilt_limit, img_yc + reasonable_r_tilt_limit],
                            [z_calib, z_calib],
                            color='k', label='avg(r<150um)={}'.format(np.round(z_calib, 1)))

                # plot fitted plane
                for dict_fit_plane, is_method, plane_clr in zip([i_dict_fit_plane, s_dict_fit_plane],
                                                                ['idpt', 'spct'],
                                                                ['navy', 'green']):
                    plane_x = dict_fit_plane['px']
                    plane_y = dict_fit_plane['py']
                    plane_z = dict_fit_plane['pz']

                    plot_plane_along_xix = [plane_x[0][0], plane_x[0][1]]
                    plot_plane_along_xiz = [plane_z[0][0], plane_z[0][1]]
                    plot_plane_along_xfx = [plane_x[1][0], plane_x[1][1]]
                    plot_plane_along_xfz = [plane_z[1][0], plane_z[1][1]]
                    axx.plot(plot_plane_along_xix, plot_plane_along_xiz, color=plane_clr, alpha=0.5, label=is_method)
                    axx.plot(plot_plane_along_xfx, plot_plane_along_xfz, color=plane_clr, alpha=0.5)

                    plot_plane_along_yiy = [plane_y[0][0], plane_y[1][0]]
                    plot_plane_along_yiz = [plane_z[0][0], plane_z[1][0]]
                    plot_plane_along_yfy = [plane_y[0][1], plane_y[1][1]]
                    plot_plane_along_yfz = [plane_z[0][1], plane_z[1][1]]
                    ax.plot(plot_plane_along_yiy, plot_plane_along_yiz, color=plane_clr, alpha=0.5, label=is_method)
                    ax.plot(plot_plane_along_yfy, plot_plane_along_yfz, color=plane_clr, alpha=0.5)

                axx.set_ylabel('z')
                axx.set_xlabel('x')
                # axx.legend(fontsize='small', frameon=True)
                axx.grid(alpha=0.125)
                ax.set_xlabel('y')
                ax.legend(fontsize='small', frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
                ax.grid(alpha=0.125)
                plt.suptitle(
                    'z_xyc(IDPT, SPCT) = ({}, {})'.format(np.round(i_dict_fit_plane['z_f_fit_plane_image_center'], 1),
                                                          np.round(s_dict_fit_plane['z_f_fit_plane_image_center'], 1),
                                                          ) + '\n' +
                    'rmse: ({}, {}); IDPT tilt: (x={}, y={}) deg'.format(np.round(i_dict_fit_plane['rmse'], 2),
                                                                         np.round(s_dict_fit_plane['rmse'], 2),
                                                                         np.round(i_dict_fit_plane['tilt_x_degrees'],
                                                                                  4),
                                                                         np.round(i_dict_fit_plane['tilt_y_degrees'],
                                                                                  4),
                                                                         ),
                )

                plt.tight_layout()
                plt.savefig(
                    join(path_scatter_zy_by_z, 'idpt-spct_scatter-z-by-y_z-true={}.png'.format(np.round(z_true, 1))))
                plt.close()

                # ---

                # plotted fitted plane and points in 3D
                if plot_3d_tilt_per_frame:
                    fig = plot_fitted_plane_and_points(df=dfit, dict_fit_plane=i_dict_fit_plane, param_z='z_no_corr',
                                                       param_z_corr='z')
                    plt.savefig(join(path_scatter_zy_by_z,
                                     'scatter-3D-with-plane-scatter_z-true={}.png'.format(np.round(z_true, 1))))
                    plt.close()

                # raise ValueError()
        # ---

        # get average position of calibration particle
        if assign_z_true_to_fit_plane_xyzc:
            i_z_calib = i_dict_fit_plane['z_f_fit_plane_image_center']
            s_z_calib = s_dict_fit_plane['z_f_fit_plane_image_center']
        elif z_calib is None:
            i_z_calib = dfit[dfit['id'] == i_calib_id_from_testset].z.mean()
            s_z_calib = dfst[dfst['id'] == s_calib_id_from_testset].z.mean()
        else:
            i_z_calib = z_calib
            s_z_calib = z_calib

        dfit['z_calib'] = i_z_calib
        dfit['error_rel_p_calib'] = dfit['z'] - i_z_calib

        dfst['z_calib'] = s_z_calib
        dfst['error_rel_p_calib'] = dfst['z'] - s_z_calib

        # ---
        # OUTLIER REMOVAL

        dfit = dfit[dfit['error_rel_p_calib'].abs() < z_error_limit]
        dfst = dfst[dfst['error_rel_p_calib'].abs() < z_error_limit]

        # plot after outlier removal
        if plot_tilt_per_frame_after_removing_large_z_errors:
            fig, (axx, ax) = plt.subplots(ncols=2, sharey=True,
                                          figsize=(size_x_inches * 2.75, size_y_inches * 1))

            axx.scatter(dfit['x'], dfit['z'], s=2, color='dodgerblue', label='i t.c.')
            axx.scatter(dfit['x'], dfit['z_no_corr'], s=2, color='blue', label='i raw')

            axx.scatter(dfst['x'], dfst['z'], s=2, color='lime', label='s t.c.')
            axx.scatter(dfst['x'], dfst['z_no_corr'], s=2, color='green', label='s raw')

            # axx.scatter(dfit_within_tilt['x'], dfit_within_tilt['z'], s=1, marker='.', color='k', label='raw-fitted')

            # ---
            # plot the calibration particle
            axx.scatter(dfit[dfit['id'] == i_calib_id_from_testset]['x'],
                        dfit[dfit['id'] == i_calib_id_from_testset]['z'],
                        marker='*', s=8, color='red', label=r'$p_{cal, IDPT}$')
            axx.scatter(dfst[dfst['id'] == s_calib_id_from_testset]['x'],
                        dfst[dfst['id'] == s_calib_id_from_testset]['z'],
                        marker='d', s=8, color='purple', label=r'$p_{cal, SPCT}$')

            # ---

            ax.scatter(dfit['y'], dfit['z'], s=2, color='dodgerblue', label='i t.c.')
            ax.scatter(dfit['y'], dfit['z_no_corr'], s=2, color='blue', label='i raw')

            ax.scatter(dfst['y'], dfst['z'], s=2, color='lime', label='s t.c.')
            ax.scatter(dfst['y'], dfst['z_no_corr'], s=2, color='green', label='s raw')

            # ax.scatter(dft_within_tilt['y'], dft_within_tilt['z'], s=1, marker='.', color='k', label='raw-fitted')

            # ---
            # plot the calibration particle
            ax.scatter(dfit[dfit['id'] == i_calib_id_from_testset]['y'],
                       dfit[dfit['id'] == i_calib_id_from_testset]['z'],
                       marker='*', s=8, color='red', label=r'$p_{cal, IDPT}$')
            ax.scatter(dfst[dfst['id'] == s_calib_id_from_testset]['y'],
                       dfst[dfst['id'] == s_calib_id_from_testset]['z'],
                       marker='d', s=8, color='purple', label=r'$p_{cal, SPCT}$')

            # ---

            if z_calib is not None:
                axx.plot([img_xc - reasonable_r_tilt_limit, img_xc + reasonable_r_tilt_limit],
                         [z_calib, z_calib],
                         color='k', label='avg(r<150um)={}'.format(np.round(z_calib, 1)))
                ax.plot([img_yc - reasonable_r_tilt_limit, img_yc + reasonable_r_tilt_limit],
                        [z_calib, z_calib],
                        color='k', label='avg(r<150um)={}'.format(np.round(z_calib, 1)))

            # plot fitted plane
            for dict_fit_plane, is_method, plane_clr in zip([i_dict_fit_plane, s_dict_fit_plane],
                                                            ['idpt', 'spct'],
                                                            ['navy', 'green']):
                plane_x = dict_fit_plane['px']
                plane_y = dict_fit_plane['py']
                plane_z = dict_fit_plane['pz']

                plot_plane_along_xix = [plane_x[0][0], plane_x[0][1]]
                plot_plane_along_xiz = [plane_z[0][0], plane_z[0][1]]
                plot_plane_along_xfx = [plane_x[1][0], plane_x[1][1]]
                plot_plane_along_xfz = [plane_z[1][0], plane_z[1][1]]
                axx.plot(plot_plane_along_xix, plot_plane_along_xiz, color=plane_clr, alpha=0.5, label=is_method)
                axx.plot(plot_plane_along_xfx, plot_plane_along_xfz, color=plane_clr, alpha=0.5)

                plot_plane_along_yiy = [plane_y[0][0], plane_y[1][0]]
                plot_plane_along_yiz = [plane_z[0][0], plane_z[1][0]]
                plot_plane_along_yfy = [plane_y[0][1], plane_y[1][1]]
                plot_plane_along_yfz = [plane_z[0][1], plane_z[1][1]]
                ax.plot(plot_plane_along_yiy, plot_plane_along_yiz, color=plane_clr, alpha=0.5, label=is_method)
                ax.plot(plot_plane_along_yfy, plot_plane_along_yfz, color=plane_clr, alpha=0.5)

            axx.set_ylabel('z')
            axx.set_xlabel('x')
            # axx.legend(fontsize='small', frameon=True)
            axx.grid(alpha=0.125)
            ax.set_xlabel('y')
            ax.legend(fontsize='small', frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(alpha=0.125)
            plt.suptitle(
                'z_xyc(IDPT, SPCT) = ({}, {})'.format(np.round(i_dict_fit_plane['z_f_fit_plane_image_center'], 1),
                                                      np.round(s_dict_fit_plane['z_f_fit_plane_image_center'], 1),
                                                      ) + '\n' +
                'rmse: ({}, {}); IDPT tilt: (x={}, y={}) deg'.format(np.round(i_dict_fit_plane['rmse'], 2),
                                                                     np.round(s_dict_fit_plane['rmse'], 2),
                                                                     np.round(i_dict_fit_plane['tilt_x_degrees'],
                                                                              4),
                                                                     np.round(i_dict_fit_plane['tilt_y_degrees'],
                                                                              4),
                                                                     ),
            )

            plt.tight_layout()
            plt.savefig(
                join(path_scatter_zy_by_z, 'idpt-spct_scatter-z-by-y_z-true={}.png'.format(np.round(z_true, 1))))
            plt.close()

            # ---

            # plotted fitted plane and points in 3D
            if plot_3d_tilt_per_frame:
                fig = plot_fitted_plane_and_points(df=dfit, dict_fit_plane=i_dict_fit_plane, param_z='z_no_corr',
                                                   param_z_corr='z')
                plt.savefig(join(path_scatter_zy_by_z,
                                 'scatter-3D-with-plane-scatter_z-true={}.png'.format(np.round(z_true, 1))))
                plt.close()

            # raise ValueError()
        # ---

        # ---

        dfis.append(dfit)
        dfss.append(dfst)

    # ---

    # ---
    # analyze fit plane xyzc and rmse-z
    df_fit_plane = pd.DataFrame(data=np.vstack([z_trues,
                                                i_fit_plane_img_xyzc, s_fit_plane_img_xyzc,
                                                i_fit_plane_rmsez, s_fit_plane_rmsez]).T,
                                columns=['z_nominal',
                                         'iz_xyc', 'sz_xyc',
                                         'irmsez', 'srmsez'])

    df_fit_plane['iz_diff'] = df_fit_plane['iz_xyc'] - df_fit_plane['z_nominal']
    df_fit_plane['sz_diff'] = df_fit_plane['sz_xyc'] - df_fit_plane['z_nominal']
    df_fit_plane.to_excel(join(path_scatter_zy_by_z, 'both_fit-respective-plane-xyzc-rmsez_by_z-true.xlsx'))

    # plot fit_plane_image_xyzc (the z-position at the center of the image) and rmse-z as a function of z_true
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(df_fit_plane['z_nominal'], df_fit_plane['iz_diff'], '-o', label='IDPT')
    ax1.plot(df_fit_plane['z_nominal'], df_fit_plane['sz_diff'], '-o', label='SPCT')
    ax2.plot(df_fit_plane['z_nominal'], df_fit_plane['irmsez'], '-o', label='IDPT')
    ax2.plot(df_fit_plane['z_nominal'], df_fit_plane['srmsez'], '-o', label='SPCT')
    ax1.set_ylabel(r'$z_{nom} - z_{xyc} \: (\mu m)$')
    ax2.set_ylabel('r.m.s.e.(z) fit plane ' + r'$(\mu m)$')
    ax2.set_xlabel(r'$z_{nominal}$')
    plt.tight_layout()
    plt.savefig(join(path_scatter_zy_by_z, 'both_fit-respective-plane-xyzc-rmsez_by_z-true.png'))
    plt.show()
    plt.close()

    # ---

    dfis = pd.concat(dfis)
    dfis['abs_error_rel_p_calib'] = dfis['error_rel_p_calib'].abs()
    dfis = dfis[dfis['abs_error_rel_p_calib'] < z_error_limit]

    dfss = pd.concat(dfss)
    dfss['abs_error_rel_p_calib'] = dfss['error_rel_p_calib'].abs()
    dfss = dfss[dfss['abs_error_rel_p_calib'] < z_error_limit]

    dfig = dfis
    dfsg = dfss

    # ---

    plot_errors_rel_calib = True
    if plot_errors_rel_calib:
        fig, ax = plt.subplots()
        ax.scatter(dfig['r'], dfig['error_rel_p_calib'], s=2, label='IDPT')
        ax.scatter(dfsg['r'], dfsg['error_rel_p_calib'], s=2, label='SPCT')
        ax.set_xlabel('r')
        ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
        ax.legend()
        plt.tight_layout()
        plt.savefig(join(path_results_tilt_corr, 'both_error_relative_calib_particle_' +
                         'zerrlim{}_cmin{}_mincountsallframes{}_'.format(z_error_limit, min_cm, min_counts) +
                         'scatter-error_rel_p_calib.png'))
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.scatter(dfig['r'], dfig['abs_error_rel_p_calib'], s=2, label='IDPT')
        ax.scatter(dfsg['r'], dfsg['abs_error_rel_p_calib'], s=2, label='SPCT')
        ax.set_xlabel('r')
        ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
        ax.legend()
        plt.tight_layout()
        plt.savefig(join(path_results_tilt_corr, 'both_error_relative_calib_particle_' +
                         'zerrlim{}_cmin{}_mincountsallframes{}_'.format(z_error_limit, min_cm, min_counts) +
                         'scatter-abs-error_rel_p_calib.png'))
        plt.show()
        plt.close()

    # ---

    # export
    dfig.to_excel(join(path_results_tilt_corr, 'idpt_error_relative_calib_particle_' +
                       'zerrlim{}_cmin{}_mincountsallframes{}.xlsx'.format(z_error_limit, min_cm, min_counts)))
    dfsg.to_excel(join(path_results_tilt_corr, 'spct_error_relative_calib_particle_' +
                       'zerrlim{}_cmin{}_mincountsallframes{}.xlsx'.format(z_error_limit, min_cm, min_counts)))

    # ---

    # plot tilt per frame
    dfig = dfig.groupby('z_true').mean().reset_index()
    xspan = 512 * microns_per_pixel
    ms = 4

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    ax1.plot(dfig.z_true, dfig.tilt_x_degrees, '-o', ms=ms, label='x', color='r')
    ax1.plot(dfig.z_true, dfig.tilt_y_degrees, '-s', ms=ms, label='y', color='k')

    ax2.plot(dfig.z_true, np.abs(xspan * np.tan(np.deg2rad(dfig.tilt_x_degrees))), '-o', ms=ms, label='x', color='r')
    ax2.plot(dfig.z_true, np.abs(xspan * np.tan(np.deg2rad(dfig.tilt_y_degrees))), '-s', ms=ms, label='y', color='k')

    ax1.set_ylabel('Tilt ' + r'$(deg.)$')
    ax1.legend()
    ax2.set_ylabel(r'$\Delta z_{FoV} \: (\mu m)$')
    ax2.set_xlabel(r'$z_{nominal} \: (\mu m)$')
    plt.tight_layout()
    plt.savefig(join(path_scatter_zy_by_z, 'sample-tilt-by-fit-IDPT_by_z-true.png'))
    plt.show()
    plt.close()

    # -

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.2))

    ax1.plot(dfig.z_true, dfig.tilt_x_degrees, '-o', ms=ms, label='x', color='r')
    ax1.plot(dfig.z_true, dfig.tilt_y_degrees, '-s', ms=ms, label='y', color='k')

    ax2.plot(dfig.z_true, np.abs(xspan * np.tan(np.deg2rad(dfig.tilt_x_degrees))), '-o', ms=ms, label='x', color='r')
    ax2.plot(dfig.z_true, np.abs(xspan * np.tan(np.deg2rad(dfig.tilt_y_degrees))), '-s', ms=ms, label='y', color='k')

    ax3.plot(df_fit_plane['z_nominal'], df_fit_plane['irmsez'], '-o', ms=ms, label='IDPT')
    # ax3.plot(df_fit_plane['z_nominal'], df_fit_plane['srmsez'], '-o', label='SPCT')

    ax1.set_ylabel('Tilt ' + r'$(deg.)$')
    ax1.legend()
    ax2.set_ylabel(r'$\Delta z_{FoV} \: (\mu m)$')
    ax3.set_ylabel(r'$\sigma_z^{fit} \: (\mu m)$')
    ax3.set_xlabel(r'$z_{nominal} \: (\mu m)$')
    ax3.legend()
    plt.tight_layout()
    plt.savefig(join(path_scatter_zy_by_z, 'sample-tilt-and-rmsez-by-fit-IDPT-and_by_z-true.png'))
    plt.show()
    plt.close()

    # -

    fig, ax = plt.subplots(figsize=(size_x_inches * 1.05, size_y_inches * 0.75))

    ax.plot(df_fit_plane['irmsez'], np.abs(dfig.tilt_x_degrees), 'o', ms=ms, label='x', color='r')
    ax.plot(df_fit_plane['irmsez'], np.abs(dfig.tilt_y_degrees), 's', ms=ms, label='y', color='k')

    ax.set_ylabel('Tilt ' + r'$(deg.)$')
    ax.set_xlabel(r'$\sigma_z^{fit} \: (\mu m)$')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(join(path_scatter_zy_by_z, 'abs-sample-tilt_by_rmsez-fit-IDPT.png'))
    plt.show()
    plt.close()

# ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# XXX. CORRECT STAGE TILT DURING CALIBRATION THAT AFFECTED IDPT TEST COORDS
correct_idpt_coords_to_account_for_tilt_during_calibration_image_acquisition = False  # False True
if correct_idpt_coords_to_account_for_tilt_during_calibration_image_acquisition:

    export_processed_coords = False
    plot_scatter_y_by_z_true = False


    def fit_plane_and_bispl(param_zf, path_figs=None):
        """
        dict_fit_plane, dict_fit_plane_bspl_corrected = fit_plane_and_bispl(param_zf='zf_from_nsv', path_figs=path_figs)

        NOTE: the calibration coords (hard coded here) come frome: /Users/mackenzie/Desktop/gdpyt-characterization/
        experiments/11.06.21_z-micrometer-v2/results/results-June-2023/calibration-spct_dzc-1
        """
        path_calib_coords = join(base_dir, 'coords', 'calib-coords')

        # file paths
        if path_figs is not None:
            path_calib_surface = path_results + '/calibration-surface'
            if not os.path.exists(path_calib_surface):
                os.makedirs(path_calib_surface)

        # read coords
        dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method='spct')
        dfc = dfcpid
        del dfcpid, dfcstats

        # print mean
        zf_methods = ['zf_from_peak_int', 'zf_from_nsv', 'zf_from_nsv_signal']
        for zfm in zf_methods:
            print("{}: {} +/- {}".format(zfm, np.round(dfc[zfm].mean(), 2), np.round(dfc[zfm].std(), 2)))

        # ---

        # fit plane
        dfc[param_zf] = dfc[param_zf]

        # ---

        # fit spline to 'raw' data
        bispl_raw, rmse = fit.fit_3d_spline(x=dfc.x,
                                            y=dfc.y,  # raw: dfc.y, mirror-y: img_yc * 2 - dfc.y
                                            z=dfc[param_zf],
                                            kx=2,
                                            ky=2)
        print("fit bispl to raw, RMSE = {} microns".format(np.round(rmse, 3)))

        # return 3 fits to actual data
        dict_fit_plane, dict_fit_plane_bspl_corrected, dfcal_field_curvature_and_tilt_corrected, bispl = \
            correct.fit_plane_correct_plane_fit_spline(dfc,
                                                       param_zf,
                                                       microns_per_pixel,
                                                       img_xc,
                                                       img_yc,
                                                       kx=2,
                                                       ky=2,
                                                       path_figs=path_figs)

        # return faux flat plane
        # faux_zf = 'faux_zf_flat'
        # dfc[faux_zf] = dfc['zf_from_nsv'].mean()
        # dict_flat_plane = correct.fit_in_focus_plane(dfc, faux_zf, microns_per_pixel, img_xc, img_yc)

        return dict_fit_plane, dict_fit_plane_bspl_corrected, bispl


    # -

    # RUN
    PARAM_ZF = 'zf_from_nsv'
    flip_correction = False

    # 1. fit "focus surface" to calibration coords
    # for PARAM_ZF in ['zf_from_peak_int', 'zf_from_nsv']:
    dict_fit_plane, dict_fit_plane_bspl_corrected, bispl_fc = fit_plane_and_bispl(param_zf=PARAM_ZF, path_figs=None)

    # 2. read test coords for both IDPT and SPCT
    path_idpt = join(base_dir, 'coords', 'raw-original_test-coords',
                     'test_coords_particle_image_stats_tm16_cm19_raw-original.xlsx')
    path_spct = join(base_dir, 'coords', 'raw-original_test-coords',
                     'test_coords_particle_image_stats_spct-1_raw-original.xlsx')

    dfi = pd.read_excel(path_idpt)
    dfs = pd.read_excel(path_spct)

    # shared
    padding = 5
    padding_rel_true_x = 0
    padding_rel_true_y = 0

    # dataset alignment
    z_zero_from_calibration = 49.9  # 50.0
    z_zero_of_calib_id_from_calibration = 49.49  # the in-focus position of calib particle in test set.
    z_zero_from_test_img_center = 68.6  # 68.51
    z_zero_of_calib_id_from_test = 68.1  # the in-focus position of calib particle in calib set.

    # spct
    s_calib_id_from_testset = 92
    s_calib_id_from_calibset = 46
    s_calib_baseline_frame = 12  # NOTE: baseline frame was 'calib_13.tif' but output coords always begin at frame = 0.
    s_xsub, s_ysub, s_rsub = 'gauss_xc', 'gauss_yc', 'gauss_rc'

    # idpt
    i_calib_id_from_testset = 42
    i_calib_id_from_calibset = 42
    i_xsub, i_ysub, i_rsub = 'xg', 'yg', 'rg'


    def correct_and_align_coords(dft):
        # CORRECTION #1: resolve z-position as a function of 'frame' discrepancy
        dft['z_true_corr'] = (dft['z_true'] - dft['z_true'] % 3) / 3 * 5 + 5

        # CORRECTION #2: shift 'z_true' according to z_f (test images)
        dft['z_true_minus_zf_from_test'] = dft['z_true_corr'] - z_zero_of_calib_id_from_test

        # CORRECTION #3: shift 'z' according to z_f (calibration images)
        dft['z_minus_zf_from_calib'] = dft['z'] - z_zero_of_calib_id_from_calibration

        # STEP #4: store "original" 'z' and 'z_true' coordinates
        dft['z_orig'] = dft['z']
        dft['z_true_orig'] = dft['z_true']

        # STEP #5: update 'z' and 'z_true' coordinates & add 'error' column
        dft['z'] = dft['z_minus_zf_from_calib']
        dft['z_true'] = dft['z_true_minus_zf_from_test']
        dft['error'] = dft['z'] - dft['z_true']

        # STEP #6: add 'z_no_corr' and 'error_no_corr' column
        dft['z_no_corr'] = dft['z']
        dft['error_no_corr'] = dft['error']

        return dft


    # columns to use: 'z', 'z_true'; reserved: 'z_no_corr'
    dfi = correct_and_align_coords(dfi)
    dfs = correct_and_align_coords(dfs)

    dfi = dfi[['frame', 'id', 'z_true', 'z', 'x', 'y', 'xg', 'yg', 'cm', 'z_no_corr']]
    dfs = dfs[['frame', 'id', 'z_true', 'z', 'x', 'y', 'gauss_xc', 'gauss_yc', 'cm', 'z_no_corr']]

    dfi = dfi[dfi['z_true'].abs() < 52]
    dfs = dfs[dfs['z_true'].abs() < 52]


    def test_calib_id_alignment(df_idpt, df_spct, zi, zs):
        dfiid = df_idpt[df_idpt['id'] == i_calib_id_from_testset]
        dfsid = df_spct[df_spct['id'] == s_calib_id_from_testset]

        dfiidg = dfiid.groupby('z_true').mean().reset_index()
        dfsidg = dfsid.groupby('z_true').mean().reset_index()

        diff_z = dfiidg[zi].to_numpy() - dfsidg[zs].to_numpy()
        diff_z_mean = np.mean(diff_z)
        diff_z_std = np.std(diff_z)

        return diff_z, diff_z_mean, diff_z_std


    dfiid = dfi[dfi['id'] == i_calib_id_from_testset]
    dfsid = dfs[dfs['id'] == s_calib_id_from_testset]
    pre_diff_z, pre_diff_z_mean, pre_diff_z_std = test_calib_id_alignment(dfi, dfs, zi='z', zs='z')

    #   2.a - calibration particle is "zero" position
    cxi = dfiid[dfiid['frame'] == 39].x.values[0]
    cyi = dfiid[dfiid['frame'] == 39].y.values[0]
    cx = dfsid[dfsid['frame'] == 39].x.values[0]
    cy = dfsid[dfsid['frame'] == 39].y.values[0]

    #   2.b - stage tilt
    dfi = correct.correct_z_by_plane_relative_to_calib_pid_xy(cx=cx, cy=cy,
                                                              df=dfi,
                                                              dict_fit_plane=dict_fit_plane_bspl_corrected,
                                                              param_z='z',
                                                              param_z_corr='z_corr_tilt',
                                                              param_z_surface='z_tilt',
                                                              flip_correction=flip_correction,
                                                              )

    dfiid_ = dfi[dfi['id'] == i_calib_id_from_testset]
    dfsid_ = dfs[dfs['id'] == s_calib_id_from_testset]
    post_diff_z, post_diff_z_mean, post_diff_z_std = test_calib_id_alignment(dfi, dfs, zi='z_corr_tilt', zs='z')

    dfi['z'] = dfi['z_corr_tilt']

    if export_processed_coords:
        # dfi.to_excel(join(base_dir, 'coords', 'test_coords_particle_image_stats_tm16_cm19_aligned.xlsx'))
        # dfs.to_excel(join(base_dir, 'coords', 'test_coords_particle_image_stats_spct-1_aligned.xlsx'))

        # correct SPCT coordinates using the tilt-corrected bivariate spline (field curvature).
        dfs_corr_fc = correct.correct_z_by_spline_relative_to_calib_pid_xy(cx, cy,
                                                                           dfs,
                                                                           bispl_fc,
                                                                           param_z='z',
                                                                           param_z_corr='z_corr_fc',
                                                                           param_z_surface='z_surf',
                                                                           flip_correction=True,
                                                                           )
        dfs_corr_fc['z'] = dfs_corr_fc['z_corr_fc']
        dfs_corr_fc.to_excel(join(base_dir, 'coords', 'test_coords_particle_image_stats_spct-1_aligned-corr-fc.xlsx'))

    # ---

    if plot_scatter_y_by_z_true:
        for zt in dfi.z_true.unique():
            dfiz = dfi[dfi['z_true'] == zt]
            dfsz = dfs[dfs['z_true'] == zt]

            fig, ax = plt.subplots()
            ax.scatter(dfiz['y'], dfiz['z_corr_tilt'], s=3)
            ax.scatter(dfsz['y'], dfsz['z'], s=5)
            ax.set_ylim([zt - 2.5, zt + 2.5])
            ax.set_title('z_nom={}, z_mean={}'.format(np.round(zt, 1), np.round(dfiz['z_corr_tilt'].mean(), 3)))
            plt.tight_layout()
            plt.savefig(join(path_figs, 'tilt-y_z-true={}.png'.format(np.round(zt, 1))))
            plt.close()

    # ---


# ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# YYY. SUPPLEMENTARY INFORMATION: PLOT SPCT - NUMBER OF PARTICLES IDENTIFIED PER FRAME (CALIBRATION IMAGES)
plot_SI_spct_number_particles_per_frame_in_calib = True  # False True
if plot_SI_spct_number_particles_per_frame_in_calib:

    # NOTE: I'm not 100% sure that this is the path to the actual calibration coords used to generate test coords
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_error_relative_calib_particle/' \
         'coords/calib-coords/calib_spct_stats_11.06.21_z-micrometer-v2_5umMS__spct-cal.xlsx'

    sp = '/Users/mackenzie/Desktop/gdpyt-characterization/pubfigs/2023_figs/SI'

    # read
    df = pd.read_excel(fp)

    # REMOVE PARTICLES NEAR BORDERS (which is applied to test images and used in publication)
    df = df[(df['x'] > 19 + padding_rel_true_x) & (df['x'] < 501 + padding_rel_true_x) &
            (df['y'] > 17 + padding_rel_true_x) & (df['y'] < 499 + padding_rel_true_x)]

    df = df[df['frame'] < 105]

    # groupby
    dfg = df.groupby('z_true').count().reset_index()

    # plot
    px = 'z_true'
    py = 'id'

    fig, ax = plt.subplots()

    ax.plot(dfg[px] - z_zero_from_calibration, dfg[py], 'o', ms=2, color='k', label=dfg[py].max())

    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_xlim([-55, 60])
    ax.set_xticks([-50, -25, 0, 25, 50])
    ax.set_ylabel(r"$N_{p}^{''}$")
    # ax.legend(title='max')

    plt.tight_layout()
    plt.savefig(join(sp, 'spct_calibration_images_num_identified_per_frame_remove-borders.png'))
    plt.show()


# ---

# ---

print("Analysis completed without errors.")