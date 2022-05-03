import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import bin

"""
Blue: #0C5DA5
Green: #00B945
"""

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

sciblue = '#0C5DA5'
scigreen = '#00B945'

# ----------------------------------------------------------------------------------------------------------------------

"""
NOTE: There are two parts to this analysis:
    A. Calculate the mean rmse_z by grouping dataframes.
    B. Bin and plot rmse_z by dx.
"""

# ----------------------------------------------------------------------------------------------------------------------
# PART A.

# filepaths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/figure data/grid-overlap/'

path_read = base_dir + 'results/test-coords/'
path_figs = base_dir + 'figs/'
path_results = base_dir + 'results/average/'

fp1 = path_read + 'test_id1_coords_static_grid-overlap-random-z-nl1_percent_overlap.xlsx'
fp2 = path_read + 'test_id2_coords_static_grid-overlap-random-z-nl1_percent_overlap.xlsx'
fp3 = path_read + 'test_id11_coords_SPC_grid-overlap-random-z-nl1_percent_overlap.xlsx'
fp4 = path_read + 'test_id12_coords_SPC_grid-overlap-random-z-nl1_percent_overlap.xlsx'

df1 = pd.read_excel(fp1)
df2 = pd.read_excel(fp2)
df3 = pd.read_excel(fp3)
df4 = pd.read_excel(fp4)

# concat IDPT and SPCT dataframes
dfi = pd.concat([df1, df2], ignore_index=True)
dfs = pd.concat([df3, df4], ignore_index=True)

dfbi = bin.bin_local_rmse_z(dfi,
                            column_to_bin='z_true',
                            bins=1,
                            min_cm=0.5,
                            z_range=None,
                            round_to_decimal=4,
                            df_ground_truth=None,
                            dropna=True,
                            error_column='error')

dfbs = bin.bin_local_rmse_z(dfs,
                            column_to_bin='z_true',
                            bins=1,
                            min_cm=0.5,
                            z_range=None,
                            round_to_decimal=4,
                            df_ground_truth=None,
                            dropna=True,
                            error_column='error')

dfb_rmse = pd.concat([dfbi, dfbs], ignore_index=True)
dfb_rmse.to_excel(path_results + 'rmse_z_mean_1-bin.xlsx')


# ----------------------------------------------------------------------------------------------------------------------
# PART B.

# filepaths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/synthetic grid overlap random z nl1/'

path_read = base_dir + 'percent-overlap/results/bin by particle spacing/'
path_figs = base_dir + '/figs/'

fp1 = path_read + 'test_id1_coords_static_global_binned_rmsez_by_particle_spacing.xlsx'
fp2 = path_read + 'test_id2_coords_static_global_binned_rmsez_by_particle_spacing.xlsx'
fp3 = path_read + 'test_id11_coords_SPC_global_binned_rmsez_by_particle_spacing.xlsx'
fp4 = path_read + 'test_id12_coords_SPC_global_binned_rmsez_by_particle_spacing.xlsx'

df1 = pd.read_excel(fp1)
df2 = pd.read_excel(fp2)
df3 = pd.read_excel(fp3)
df4 = pd.read_excel(fp4)

# add column for true number of particles
true_num_13 = np.array([4, 4, 4, 4, 4, 4, 4, 2, 2, 2]) * 1000

# fix 'percent_meas' column
df1['true_num'] = true_num_13
df1['true_percent_meas'] = df1.num_meas / df1.true_num * 100
df2['true_percent_meas'] = df2.num_meas / 4824 * 100
df3['true_num'] = true_num_13
df3['true_percent_meas'] = df3.num_meas / df3.true_num * 100
df4['true_percent_meas'] = df4.num_meas / 4824 * 100

# stack dfs
dfi = pd.concat([df1, df2], ignore_index=True)
dfs = pd.concat([df3, df4], ignore_index=True)

# rename 'filename' to dx
dfi = dfi.rename(columns={'filename': 'dx'})
dfs = dfs.rename(columns={'filename': 'dx'})

# sort
dfi = dfi.sort_values('dx')
dfs = dfs.sort_values('dx')



# --- plots

# formatting
h = 80
ms = 3

# plot IDPT vs. SPCT: rmse_z(dx)
fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 2]})

ax1.plot(dfi.dx, dfi.true_percent_meas, '-o', ms=ms, color=sciblue)
ax1.plot(dfs.dx, dfs.true_percent_meas, '-o', ms=ms, color=scigreen)
ax1.plot(dfi.dx, dfi.true_percent_meas, '-o', ms=ms, color=sciblue)

ax1.set_ylabel(r'$\phi \: (\%)$')

ax2.plot(dfi.dx, dfi.rmse_z / h, '-o', ms=ms, color=sciblue, label='IDPT')
ax2.plot(dfs.dx, dfs.rmse_z / h, '-o', ms=ms, color=scigreen, label='SPCT')
ax2.plot(dfi.dx, dfi.rmse_z / h, '-o', ms=ms, color=sciblue)

ax2.set_xlabel(r'$\delta x \: (pix.)$')
ax2.set_ylabel(r'$\overline{\sigma_{z}}/h$')
ax2.set_yscale('log')
ax2.legend()

plt.tight_layout()
plt.savefig(path_figs + 'norm-rmse-z_true-percent-meas_by_dx.png')
plt.show()