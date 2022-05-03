# test bin, analyze, and plot functions

from os.path import join
from os import listdir

import matplotlib.pyplot as plt
# imports
import numpy as np
import pandas as pd

import filter
import analyze
from correction import correct
from utils import plot_collections, plotting, io

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


"""
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/analyses/' \
           'results-04.17.22_10X-spct-meta-assessment'

method = 'spct'
min_cm = 0.5
min_percent_layers = 0.5
microns_per_pixel = 3.2
path_calib_spct_pop=None

if method == 'idpt' and path_calib_spct_pop is None:
    raise ValueError('Must specifiy path_calib_spct_pop for IDPT analyses.')

# filepaths
path_test_coords = join(base_dir, 'coords/test-coords')
path_calib_coords = join(base_dir, 'coords/calib-coords')
path_similarity = join(base_dir, 'similarity')
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

# --- --- SPCT ASSESSMENT

# calibration coords
dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method=method)

dfcg = dfc.groupby('id').mean()
dfcpid = dfcpid.set_index('id')

dfcpid = pd.concat([dfcpid, dfcg[['x', 'y']]], axis=1, join='inner').reset_index()
# dfcpidxy.to_excel(path_calib_coords + '/calib_idpt_pid_defocus_stats_xy.xlsx', index=False)

# inspect initial distribution of in-focus particle positions
fig, ax = plotting.scatter_z_by_xy(df=dfcpid, z_params=['zf_from_peak_int', 'zf_from_nsv'])
fig.savefig(path_figs + '/zf_scatter_xy_int-and-nsv.png')
dict_fit_plane, fig_xy, fig_xyz, fig_plane = correct.inspect_calibration_surface(df=dfcpid,
                                                                                 param_zf='zf_from_nsv',
                                                                                 microns_per_pixel=microns_per_pixel)
fig_xy.savefig(path_figs + '/zf_scatter_xy.png')
fig_xyz.savefig(path_figs + '/zf_scatter_xyz.png')
fig_plane.savefig(path_figs + '/zf_fit-3d-plane.png')

# read diameter paramaters
if path_calib_spct_pop is not None:
    mag_eff, zf, c1, c2 = io.read_pop_gauss_diameter_properties(path_calib_spct_pop)
else:
    mag_eff, zf, c1, c2 = io.read_pop_gauss_diameter_properties(dfcpop)


# --- CALIBRATION STACK SIMILARITY
# read
dfs, dfsf, dfsm, dfas, dfcs = io.read_similarity(path_similarity)

# plot
if dfsf is not None:
    fig, ax = plotting.plot_calib_stack_self_similarity(dfsf, min_percent_layers=min_percent_layers)
    ax.set_xlabel(r'$z_{calib.} \: (\mu m)$')
    ax.set_ylabel(r'$\overline{S}_{(i, i+1)}$')
    plt.tight_layout()
    plt.savefig(path_figs + '/calib_self-similarity-forward.png')
    plt.show()

if dfsm is not None:
    fig, ax = plotting.plot_calib_stack_self_similarity(dfsm, min_percent_layers=min_percent_layers)
    ax.set_xlabel(r'$z_{calib.} \: (\mu m)$')
    ax.set_ylabel(r'$\overline{S}_{(i-1, i, i+1)}$')
    plt.tight_layout()
    plt.savefig(path_figs + '/calib_self-similarity-middle.png')
    plt.show()

if dfas is not None:
    fig, ax = plotting.plot_particle_to_particle_similarity(dfcs, min_particles_per_frame=10)
    ax.set_xlabel(r'$z_{calib.} \: (\mu m)$')
    ax.set_ylabel(r'$\overline{S}_{i}(p_{i}, p_{N})$')
    plt.tight_layout()
    plt.savefig(path_figs + '/calib_per-frame_particle-to-particle-similarity.png')
    plt.show()

# --- --- INTRINSIC ABERRATIONS ASSESSMENT
# --- RAW
# evaluate
dict_ia = analyze.evaluate_intrinsic_aberrations(dfs,
                                                 z_f=zf,
                                                 min_cm=min_cm,
                                                 param_z_true='z_true',
                                                 param_z_cm='z_cm')

dict_ia = analyze.fit_intrinsic_aberrations(dict_ia)
io.export_dict_intrinsic_aberrations(dict_ia, path_results, unique_id='raw')

# plot
fig, ax = plotting.plot_intrinsic_aberrations(dict_ia, cubic=True, quartic=True)
ax.set_xlabel(r'$z_{raw} \: (\mu m)$')
ax.set_ylabel(r'$S_{max}(z_{l}) / S_{max}(z_{r})$')
ax.grid(alpha=0.125)
ax.legend(['Data', 'Cubic', 'Quartic'])
plt.tight_layout()
plt.savefig(path_figs + '/intrinsic-aberrations_raw.png')
plt.show()
"""



# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/analyses/' \
           'results-04.16.22_10X-spct-idpt-meta-assessment/idpt'

# plot_collections.plot_spct_stats(base_dir)



method = 'idpt'
path_calib_spct_pop = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/analyses/results-04.16.22_10X-spct-idpt-meta-assessment/spct/coords/calib-coords/calib_spct_pop_defocus_stats_02.06.22_membrane_characterization_calib_52.xlsx'
min_cm = 0.5
min_percent_layers = 0.5
microns_per_pixel = 3.2

plot_collections.plot_meta_assessment(base_dir, method, min_cm, min_percent_layers, microns_per_pixel, path_calib_spct_pop)

raise ValueError('ha')


j = 1