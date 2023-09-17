# Plot Ellipses

from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF9500'
sciorange = '#FF2C00'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ---

# file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_particle_image_ellipses'
path_read = join(base_dir, 'data')
path_save = join(base_dir, 'figs')

# file name
fn = 'calib_spct_stats_11.06.21_z-micrometer-v2_1umSteps.xlsx'

# setup
microns_per_pixels = 1.6
image_pixels = 512
image_padding = 5
image_range = image_pixels + image_padding * 2

# ---

# read file
df = pd.read_excel(join(path_read, fn))
df = df.dropna(subset=['gauss_sigma_x_y'])

# modifiers
normalize = True
plot_rotated = True
plot_labels = True
plot_colorbar = True
filetype = '.svg'
dpi = 300

# plot variables
if plot_rotated:
    angle = -45
    px = 'gauss_sigma_x_r'
    py = 'gauss_sigma_y_r'
    pxy = 'gauss_sigma_x_y_r'
else:
    angle = 0
    px = 'gauss_sigma_x'
    py = 'gauss_sigma_y'
    pxy = 'gauss_sigma_x_y'

# ---

# processing
max_pid_baseline = 87
min_num_frames = 10
sigma_xy_lim = [0.5, 1.5]  # [0.125, 1.875]
plot_dz = 5

# filters
df = df[df['id'] < max_pid_baseline]

# pids with most counts
dfg = df.groupby('id').count().reset_index().sort_values(pxy, ascending=False)
pids_in_most_frames = dfg[dfg[pxy] > min_num_frames].id.values

# in order from greatest to least (the third row is tossed in because I want to plot those; not b/c they qualify)
pids_max_sigma_x_or_y_std = [71, 85, 46, 67, 73, 7, 9, 39, 54, 55, 84, 83, 11,
                             1, 15, 50, 0, 3, 2, 4, 22, 5, 16, 68, 69, 12, 10, 13, 14, 86,
                             26, 44]
pids_max_sigma_xy_std = [8, 6, 72, 0, 4, 3, 26, 10, 20, 70, 83, 24, 14, 17, 13,
                         7, 27, 9, 2, 16, 1, 11, 12, 21, 84, 5, 15, 45,
                         44, 85, 26, 50, 71, 73, 39, 67, 46, 54, 55, 69, 68, 83, 67]

# pids to plot
pids_to_plot = set(pids_in_most_frames) & set(pids_max_sigma_x_or_y_std) & set(pids_max_sigma_xy_std)
# pids_to_plot = np.arange(0, 87)
# pids_to_plot = [0, 2, 4, 7, 11, 22, 26, 39, 44, 46, 50, 51, 54, 55, 67, 69, 70, 71, 82, 83, 84, 85]

# plot sigma_xy as ellipses
dfp = df[df['id'].isin(pids_to_plot)]
# dfp = dfp[~dfp.id.isin([18, 19, 23, 28, 29, 32, 31, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 56, 57, 58, 59, 62, 63, 65, 72, 75])]

# ---

# split dataframe into 6 areas
x_range = [0, image_range / 3, image_range * 2 / 3, image_range]
y_range = [0, image_range / 3, image_range * 2 / 3, image_range]

fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches * 0.5))
# fig, ax = plt.subplots(nrows=9, figsize=(size_x_inches, size_y_inches * 3))
min_num_per_z = 2
ii = 0
for i in range(len(x_range) - 1):
    for j in range(len(y_range) - 1):

        xmin, xmax = x_range[i], x_range[i + 1]
        ymin, ymax = y_range[j], y_range[j + 1]

        dfqx = dfp[(dfp['x'] > xmin) & (dfp['x'] < xmax)]
        dfq = dfqx[(dfqx['y'] > ymin) & (dfqx['y'] < ymax)]

        # filter by number of particles per frame
        dfgc = dfq.groupby('z_true').count().reset_index()
        valid_z_range = dfgc[dfgc['id'] >= min_num_per_z]['z_true'].values
        dfq = dfq[dfq['z_true'].isin(valid_z_range)]

        dfgz = dfq.groupby('z_true').mean().reset_index()

        # plot
        if ii in [0, 2, 4, 6, 8]:
            ax.plot(dfgz['z_true'] - 50, dfgz[pxy], '-o', ms=1, label='{}, {}'.format(i, j))
        ii += 1


ax.set_xlabel(r'$z \: (\mu m)$')
ax.set_xticks([-50, 0, 50])
ax.set_ylabel(r'$w_{x}/w_{y}$')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1),
          title=r"$(m, n)$",
          handlelength=1, handletextpad=0.4, labelspacing=0.125, borderaxespad=0.3,
          )

plt.tight_layout()
plt.savefig(path_save + '/good-pids_sigma_xy_by_z-corr_sized' + filetype)
plt.show()
plt.close()

# ---

# plot the number of particles per frame
plot_num_by_z = False
if plot_num_by_z:
    fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches * 0.5))
    dfgzid = df.groupby(['z_true', 'id']).mean().reset_index()
    dfgz = dfgzid.groupby('z_true').count().reset_index()
    ax.plot(dfgz['z_true'] - 50, dfgz['id'], '-o', ms=1, label='Non-overlapping')

    ax.legend(loc='upper right', handlelength=1, handletextpad=0.4,)
    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_xticks([-50, 0, 50])
    ax.set_ylabel(r"$N_{p}^{''}$")
    ax.set_ylim(top=94.5)
    ax.set_yticks([40, 80])
    plt.minorticks_off()
    plt.tight_layout()
    # plt.savefig(path_save + '/num-pids_by_z-corr' + filetype)
    plt.show()
    plt.close()

# ---