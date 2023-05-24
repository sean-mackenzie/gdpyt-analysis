# imports
from os.path import join
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from utils.plotting import lighten_color

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

mag_eff = 10.01
NA_eff = 0.45
microns_per_pixel = 1.6
size_pixels = 16  # microns

z_range = [-50, 55]
measurement_depth = z_range[1] - z_range[0]
h = measurement_depth
min_cm = 0.5

# processing
filter_step_size = 4

# file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_rigid_body_displacement'
path_read = join(base_dir, 'data')
path_save = join(base_dir, 'figs')
filetype = '.xlsx'

# rigid body transformation between dz steps
plot_icp_dz = True
if plot_icp_dz:
    # file names
    fni = 'dfdz_icp_z-error-{}_in-plane-dist-5'.format(filter_step_size)
    fpi = join(path_read, fni + '.xlsx')
    dfi = pd.read_excel(fpi)

    # plot
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, sharex=True, figsize=(size_x_inches * 0.9, size_y_inches * 1.45))

    ax0.plot(dfi.z, dfi.dx * microns_per_pixel, '-o', label='x')
    ax1.plot(dfi.z, dfi.dy * microns_per_pixel, '-o', label='y')
    ax2.plot(dfi.z, dfi.dz * -1, '-o')
    ax2.axhline(y=5, color='black', linewidth=0.5, linestyle='--', alpha=0.25)
    ax2.axhline(y=5 + 0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.125)
    ax2.axhline(y=5 - 0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.125)
    ax3.plot(dfi.z, dfi.rmse, '-o')

    ax0.set_ylabel(r'$\Delta_{x} \: (\mu m)$')
    ax0.set_ylim([-0.25, 0.25])
    ax0.set_yticks([-0.2, 0, 0.2])

    ax1.set_ylabel(r'$\Delta_{y} \: (\mu m)$')
    ax1.set_ylim([-0.25, 0.25])
    ax1.set_yticks([-0.2, 0, 0.2])

    ax2.set_ylim([6, 4])
    ax2.set_ylabel(r'$\Delta_{z} \: (\mu m)$')

    ax3.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    ax3.set_ylim([0, 2])
    ax3.set_yticks([0, 2])
    ax3.set_xlabel(r'$z \: (\mu m)$')

    plt.tight_layout()
    plt.savefig(path_save + '/dZ_rigid-body-transform_z-error-{}_sized.png'.format(filter_step_size))
    plt.show()

# ---

# rigid body transformation between dz steps
plot_icp_df = True
if plot_icp_df:
    # file names
    fni = 'df_icp_z-error-{}_in-plane-dist-5'.format(filter_step_size)
    fpi = join(path_read, fni + '.xlsx')
    dfi = pd.read_excel(fpi)

    # plot
    fig, (ax0, ax2, ax3) = plt.subplots(nrows=3, sharex=True,
                                        figsize=(size_x_inches * 0.9, size_y_inches * 1.45),
                                        gridspec_kw={'height_ratios': [1, 1, 1]})

    ax0.plot(dfi.z, dfi.std_dx * microns_per_pixel, '-o', label=r'$x$')
    ax0.plot(dfi.z, dfi.std_dy * microns_per_pixel, '-o', label=r'$y$')
    ax2.plot(dfi.z, dfi.std_dz, '-o')
    ax3.plot(dfi.z, dfi.rmse, '-o')

    ax0.legend(ncol=2)
    ax0.set_ylabel('stdev. ' + r'$(x, y) \: \: (\mu m)$')
    ax0.set_ylim([0, 0.12])
    ax0.set_yticks([0, 0.1])

    ax2.set_ylabel('stdev. ' + r'$(z) \: \: (\mu m)$')
    ax2.set_ylim([0, 0.45])
    ax2.set_yticks([0, 0.2, 0.4])

    ax3.set_ylabel('r.m.s. error ' + r'$(z) \: \: (\mu m)$')
    ax3.set_xlabel(r'$z \: (\mu m)$')
    ax3.set_ylim([0, 0.925])
    ax3.set_yticks([0, 0.5])

    plt.tight_layout()
    plt.savefig(path_save + '/dFrames_rigid-body-transform_z-error-{}_sized.png'.format(filter_step_size))
    plt.show()

# ---

print("Analysis completed without errors.")