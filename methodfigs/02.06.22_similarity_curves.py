# 02.06.22 - local axial and radial displacements per membrane

# imports
import os
from os.path import join
import itertools

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, Akima1DInterpolator

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

import analyze
from utils import boundary, functions, io, bin, plot_collections
from utils.plotting import lighten_color
from utils.functions import fSphericalUniformLoad, fNonDimensionalNonlinearSphericalUniformLoad

# ---

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'
scipurple = '#845B97'
sciblack = '#474747'
scigray = '#9e9e9e'
sci_color_list = [sciblue, scigreen, scired, sciorange, scipurple, sciblack, scigray]

plt.style.use(['science', 'ieee'])  # , 'std-colors'
fig, ax = plt.subplots()
sci_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
sci_color_cycler = ax._get_lines.prop_cycler
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 0. Experimental Parameters

mag_eff = 5.0
numerical_aperture = 0.3
pixel_size = 16
# depth_of_focus = functions.depth_of_field(mag_eff, numerical_aperture, 600e-9, 1.0, pixel_size=pixel_size * 1e-6) * 1e6
microns_per_pixel = 3.2
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

# particle spacing
num_particles = 104
mean_dx = np.sqrt(image_length ** 2 / num_particles)
mean_min_dx = 25.5  # measured via spct_stats function

# axial positions
z_f_from_calib = 140
z_inital_focal_plane_bias_errors = 0
z_i_mean_allowance = 2.5

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
# 2. Setup directories

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/02.06.22_similarity_curves'
path_calib_coords = join(base_dir, 'data/coords/calib-coords')
path_similarity = join(base_dir, 'data/similarity')
path_test_similarity = join(base_dir, 'data/test-similarity-curves')
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

path_figs_test_sim = join(path_figs, 'test-similarity-curves')
if not os.path.exists(path_figs_test_sim):
    os.makedirs(path_figs_test_sim)

# ---

method = 'idpt'

df_pids_per_membrane = pd.read_excel(base_dir + '/data/results/df_pids_per_membrane.xlsx')
pids_lr = df_pids_per_membrane.iloc[0].dropna().values  # pids_lr
pids_ul = df_pids_per_membrane.iloc[1].dropna().values
pids_of_interest = np.hstack([pids_lr, pids_ul])

pids_saturated = [12, 13, 18, 34, 39, 49, 66, 78]
exclude_pids = [39, 61]

# ---

# ----------------------------------------------------------------------------------------------------------------------
# ANALYZE SPCT STATS
analyze_spct = False

if analyze_spct:

    # 1. READ CALIB COORDS, PLOT SPCT STATS, PLOT SIMILARITIES
    read_calib_coords = False
    if read_calib_coords:
        dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method=method)

    analyze_spct_stats = False
    if analyze_spct_stats:
        # read
        plot_collections.plot_spct_stats(base_dir + '/data', method=method)

    analyze_similarities = False
    if analyze_similarities:
        plot_collections.plot_similarity_analysis(base_dir + '/data', method=method, mean_min_dx=mean_min_dx)

# ---

# ----------------------------------------------------------------------------------------------------------------------
# ANALYZE TEST SIMILARITY CURVES
analyze_test_similarity_curves = True

if analyze_test_similarity_curves:

    # frames of interest
    frames_neg_to_pos_1 = np.arange(66, 74)
    frames_pos_to_neg_1 = np.arange(96, 102)
    frames_neg_to_pos_2 = np.arange(125, 130)
    frames_pos_to_neg_2 = np.arange(155, 160)
    frames_neg_to_pos_3 = np.arange(181, 186)

    frame_groups_of_interest = [frames_neg_to_pos_1, frames_pos_to_neg_1, frames_neg_to_pos_2, frames_pos_to_neg_2, frames_neg_to_pos_3]
    frames_of_interest = np.hstack(frame_groups_of_interest)

    # paths
    for sheet_number in [1, 2]:
        fn = 'particle_similarity_curves_sheet{}_dynamic_neg_first_11_1.xlsx'.format(sheet_number)

        # setup
        plot_peak_or_curve = 'both'  # 'peak' 'curve'

        # read
        for pid in pids_of_interest:
            if os.path.exists(join(path_test_similarity, 'per-pid', 'pid{}_z-corr_similarity-curves.xlsx'.format(pid))):
                dfpid = pd.read_excel(join(path_test_similarity, 'per-pid',
                                           'pid{}_z-corr_similarity-curves.xlsx'.format(pid)))
            else:
                df = pd.read_excel(join(path_test_similarity, fn))

                # filter
                df = df[df['id'].isin(pids_of_interest)]
                df = df[df['frame'].isin(frames_of_interest)]

                # correct z to z_corr (relative to z_f)
                df['z_cm'] = df['z_cm'] - z_f_from_calib
                df['z_est'] = df['z_est'] - z_f_from_calib

                # export
                df = df.sort_values('frame')
                df.to_excel(join(path_test_similarity, 'pids_z-corr_' + fn))

                # export per pid
                for pids in pids_of_interest:
                    dfpid = df[df['id'] == pids]
                    dfpid.to_excel(join(path_test_similarity, 'per-pid',
                                        'pid{}_z-corr_similarity-curves.xlsx'.format(pids)))

                # finally, get the pid you want to plot
                dfpid = df[df['id'] == pid]

            # ---

            if len(dfpid) < len(frames_of_interest):
                continue

            # plot @ frame
            for frames_of_interest_group in frame_groups_of_interest:

                fig, ax = plt.subplots(figsize=(size_x_inches * 1.2, size_y_inches * 0.725))
                clrs = iter(cm.viridis(np.linspace(0.05, 0.95, len(frames_of_interest_group))))

                for fr in frames_of_interest_group:

                    # get: this pid, this frame
                    dfi = dfpid[dfpid['frame'] == fr].sort_values('z_cm').reset_index()

                    # filter z-range
                    dfi = dfi[dfi['z_cm'].abs() < 40].reset_index()

                    # similarity curve
                    if plot_peak_or_curve == 'curve':
                        ax.plot(dfi.z_cm, dfi.cm, '-o', ms=2, linewidth=0.5, color=next(clrs),
                                label=np.round(fr / frame_rate, 2))
                    elif plot_peak_or_curve == 'peak':
                        ax.plot(dfi.iloc[dfi.cm.idxmax()].z_cm, dfi.cm.max(), 'o', ms=3, color=next(clrs),
                                label=np.round(fr / frame_rate, 2))
                    else:
                        p1, = ax.plot(dfi.z_cm, dfi.cm, '-o', ms=0.65, linewidth=0.5, color=next(clrs), alpha=0.65,
                                      label=np.round(fr / frame_rate, 2))
                        ax.scatter(dfi.iloc[dfi.cm.idxmax()].z_cm, dfi.cm.max(), s=40,
                                   marker='*', color=lighten_color(p1.get_color(), 1.1), zorder=3.5)

                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, title=r'$t \: (s)$')
                # ax.legend(loc='lower center', ncol=4, title='Frame')
                ax.set_ylabel(r'$S(I_{i}^{t}, I^{c}(z))$')
                ax.set_xlabel(r'$z \: (\mu m)$')
                # ax.set_xlim([-55, 55])
                # ax.set_xticks(ticks=[-40, 10], labels=['0', r'$h$'], minor=False)
                # ax.set_yticks(ticks=[0, 1], minor=False)

                plt.tight_layout()
                plt.minorticks_off()
                plt.savefig(path_figs_test_sim +
                            '/interp-sim-curves_pid{}_fr{}-{}.png'.format(pid, frames_of_interest_group[0],
                                                                          frames_of_interest_group[-1]))
                # plt.show()
                plt.close()

# ---

print("Analysis completed without errors.")