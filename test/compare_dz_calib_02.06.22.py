# Compare dz step size for calibration stacks, 02.06.22

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/analyses'
path_read = base_dir + '/results-09.15.22_idpt-subpix/results'
path_figs = base_dir + '/shared-results/sweep_calib-step-size/figs'

dzs = [1, 2, 3, 4, 5, 6, 7, 8, 11, 15]

# ---

# read files
dfs = []
lr_rmses = []
lr_percent_meass = []
ul_rmses = []
ul_percent_meass = []
num_frames = []
frames = []

for dz in dzs:
    fn = '/dz{}/id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest_vertical'.format(dz, dz)
    df = pd.read_excel(path_read + fn + '.xlsx')
    df = df[df['frame'] > 0]
    dfs.append(df)

    # compute frames that were analyzed to accurately compare results
    frames.append(set(df.frame.unique()))

intersecting_frames = frames[0].intersection(frames[6])  # , frames[7], frames[8], frames[9])

for df in dfs:
    df = df[df['frame'].isin(intersecting_frames)]
    num_frames.append(len(df.frame.unique()))

    # compute (1) average fit rmse, (2) average fit percent measure
    lr_rmses.append(df[df['memb_id'] == 1].fit_rmse.mean())
    lr_percent_meass.append(df[df['memb_id'] == 1].fit_percent_meas.mean())
    ul_rmses.append(df[df['memb_id'] == 2].fit_rmse.mean())
    ul_percent_meass.append(df[df['memb_id'] == 2].fit_percent_meas.mean())

# package into dataframe
res = np.vstack([dzs, num_frames, lr_rmses, lr_percent_meass, ul_rmses, ul_percent_meass]).T
dfr = pd.DataFrame(res, columns=['dz', 'num_frames', 'lr_rmse', 'lr_percent_meas', 'ul_rmse', 'ul_percent_meas'])
dfr.to_excel(path_figs + '/rmse-z_percent_meas_compare-dzc_{}frame-sets.xlsx'.format(len(np.unique(num_frames))))
# ---

# plot

# plot individually
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

ax1.plot(dfr.dz, dfr.lr_percent_meas, '-o', label='lr')
ax1.plot(dfr.dz, dfr.ul_percent_meas, '-o', label='ul')
ax2.plot(dfr.dz, dfr.lr_rmse, '-o', label='lr')
ax2.plot(dfr.dz, dfr.ul_rmse, '-o', label='ul')

ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax1.set_ylabel(r'$\phi_{S.R.}$')
ax2.set_ylabel(r'$\sigma_{S.R.}$')
ax2.set_xlabel(r'$\Delta_{c} z \: (\mu m)$')
plt.tight_layout()
plt.savefig(path_figs +
            '/rmse-z_percent_meas_lr-ul_by_dzc_up-to-15_{}frame-sets.png'.format(len(np.unique(num_frames))))
plt.show()

# ---

# plot mean of both
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

ax1.plot(dfr.dz, (dfr.lr_percent_meas + dfr.ul_percent_meas) / 2, '-o')
ax2.plot(dfr.dz, (dfr.lr_rmse + dfr.ul_rmse) / 2, '-o', label='lr')

ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax1.set_ylabel(r'$\phi_{S.R.}$')
ax2.set_ylabel(r'$\sigma_{S.R.}$')
ax2.set_xlabel(r'$\Delta_{c} z \: (\mu m)$')
plt.tight_layout()
plt.savefig(path_figs +
            '/rmse-z_percent_meas_average_by_dzc_up-to-15_{}frame-sets.png'.format(len(np.unique(num_frames))))
plt.show()