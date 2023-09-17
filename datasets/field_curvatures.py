# imports
from os.path import join
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import warnings

from utils import io, functions, bin, fit, plotting
from utils.plotting import lighten_color
from correction import correct

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

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
sci_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
sci_color_cycler = ax._get_lines.prop_cycler
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------

# ---

# ----------------------------------------------------------------------------------------------------------------------


def get_field_curvature(dataset, path_results):

    if dataset == '20X-0.5X':

        if path_results is not None:
            path_results = path_results + '/field-curvature_' + dataset
            if not os.path.exists(path_results):
                os.makedirs(path_results)

        # setup file paths
        method = 'spct'
        base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
                   'FINAL-04.25.22_SPCT_1um-calib_5um-test'
        path_calib_coords = join(base_dir, 'coords/calib-coords')
        path_calib_spct_pop = join(base_dir, 'coords/calib-coords/calib_spct_pop_defocus_stats.xlsx')

        # experimental
        microns_per_pixel = 1.6
        img_xc, img_yc = 256, 256

        # processing
        true_num_particles_per_frame = 92
        z_range = [-55, 55]
        measurement_depth = z_range[1] - z_range[0]

        # 1. read calib coords
        # mag_eff_c, zf_c, c1_c, c2_c = io.read_pop_gauss_diameter_properties(path_calib_spct_pop)
        dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method)

        param_zf = 'zf_from_peak_int'
        kx = 2
        ky = 2

        dict_fit_plane, dict_fit_plane_bspl_corrected, dfcal_field_curvature_corrected, bispl = \
            correct.fit_plane_correct_plane_fit_spline(dfcal=dfcpid,
                                                       param_zf=param_zf,
                                                       microns_per_pixel=microns_per_pixel,
                                                       img_xc=img_xc,
                                                       img_yc=img_yc,
                                                       kx=kx,
                                                       ky=ky,
                                                       path_figs=path_results)

        # step 1. correct coordinates using field curvature spline
        dfcstats_field_curvature_corrected = correct.correct_z_by_spline(dfcstats, bispl, param_z='z')

        # step 2. correct coordinates using fitted plane
        dfcstats_field_curvature_tilt_corrected = correct.correct_z_by_plane_tilt(dfcal=None,
                                                                                  dftest=dfcstats_field_curvature_corrected,
                                                                                  param_zf='none',
                                                                                  param_z='z_corr',
                                                                                  param_z_true='none',
                                                                                  popt_calib=None,
                                                                                  params_correct=None,
                                                                                  dict_fit_plane=dict_fit_plane_bspl_corrected,
                                                                                  )

        # export the corrected dfcstats
        if path_results is not None:
            dfcstats_field_curvature_tilt_corrected.to_excel(path_results +
                                                             '/calib_spct_stats_field-curvature-and-tilt-corrected.xlsx',
                                                             index=False)

        return bispl

    # ---

# ---