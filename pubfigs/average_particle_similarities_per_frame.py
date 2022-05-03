# imports
from os.path import join
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.plotting import lighten_color

"""
Blue: #0C5DA5
Green: #00B945
"""

plt.style.use(['science', 'ieee', 'std-colors'])
sci_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

sciblue = '#0C5DA5'
scigreen = '#00B945'

# --- structure data

base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/figure data/particle-similarities-in-image'
path_read = join(base_dir, 'data')
path_save = join(base_dir, 'figs')

filetype = '.xlsx'

fp_synthetic = 'average_similarity_1id_synthetic_grid-no-overlap-nl1'
fp_exp_glass = 'average_similarity_4id_SPCT_20X_1Xmag_0.87umNR'
fp_exp_silp = 'average_similarity_5id_SPCT_11.06.21_z-micrometer-v2'
fp_exp_bpe = 'average_similarity_6id_10.07.21-BPE_Pressure_Deflection_avg-sim'
fp_exp_bpe_non_u_illum = 'average_similarity_7id_SPCT_11.02.21-BPE_Pressure_Deflection_20X_non-u-illlum'

dfs1 = pd.read_excel(join(path_read, fp_synthetic) + filetype)
dfe1 = pd.read_excel(join(path_read, fp_exp_glass) + filetype)
dfe2 = pd.read_excel(join(path_read, fp_exp_silp) + filetype)
dfe3 = pd.read_excel(join(path_read, fp_exp_bpe) + filetype)
dfe4 = pd.read_excel(join(path_read, fp_exp_bpe_non_u_illum) + filetype)

dfps = []
for df in [dfs1, dfe1, dfe2, dfe3, dfe4]:
    df['z_norm'] = (df['z_corr'] - df['z_corr'].min()) / (df['z_corr'].max() - df['z_corr'].min())
    dfps.append(df)

dfs1, dfe1, dfe2, dfe3, dfe4 = dfps[0], dfps[1], dfps[2], dfps[3], dfps[4]

# --- plot

# plot one by one
ylim = [0.499, 1.01]


# --- simple
param_z = 'z_corr'

fig, ax = plt.subplots()

ax.plot(dfs1[param_z], dfs1.sim, '-o', ms=1, label='Synthetic')
ax.plot(dfe1[param_z], dfe1.sim, '-o', ms=1, label='Glass')
ax.plot(dfe2[param_z], dfe2.sim, '-o', ms=1, label='Elastomer')
ax.plot(dfe3[param_z], dfe3.sim, '-o', ms=1, label='Device')
ax.plot(dfe4[param_z], dfe4.sim, '-o', ms=1, label='Device; N.U.I.')

ax.set_xlabel(r'$z/h$')
ax.set_ylabel(r'$\overline {S} (p_{i}, p_{N})$')
ax.set_ylim(ylim)
ax.legend()

plt.tight_layout()
plt.savefig(path_save + '/average-particle-image-similarity_{}.png'.format(param_z))
plt.show()



print('Analysis completed without errors.')