
# imports
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, Akima1DInterpolator
import random

from utils import io, plotting, fit, modify

import matplotlib.pyplot as plt
plt.style.use(['science', 'ieee', 'std-colors'])




# inspect calibration in-focus coords
"""
# file path
fp_in_focus = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/test_coords/calibration/calib_in-focus_coords_z-micrometer-v2.xlsx'

# read excel to disk
df = io.read_excel(path_name=fp_in_focus, filetype='.xlsx')



j = 1
"""

# inspect calibration correction coordinates
"""
# file path
fp_correction = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/test_coords/test/step/test_id1_coords_45micron_step_towards.xlsx'

# read excel to disk
df = io.read_excel(path_name=fp_correction, filetype='.xlsx')

# plot 3D scatter of all particle coordinates
plt.style.use(['science', 'ieee', 'scatter'])
fig, ax = plotting.plot_scatter_3d(df, fig=None, ax=None, elev=20, azim=-40, color='tab:blue', alpha=0.1)
fig, ax = plotting.plot_scatter_3d(df=[df.x, df.y, df.z_f], fig=fig, ax=ax, elev=20, azim=-40, alpha=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.tight_layout()
plt.show()

# make dictionary: dataframe for 5 particles
rand_pids = [pid for pid in random.sample(set(df.id.unique()), 5)]
dfpid = df[df.id.isin(rand_pids)]
dpidicts = modify.split_df_and_merge_dficts(dfpid, keys=rand_pids, column_to_split='id', splits=rand_pids, round_to_decimal=0)

# plot peak intensity profile
fig, ax = plotting.plot_scatter(dpidicts, xparameter='z', yparameter='peak_int', min_cm=None, z0=0, take_abs=False)
ax.set_xlabel(r'$z\: (\mu m)$')
ax.set_ylabel(r'$I_{peak}\: (A.U.)$')
ax.legend(rand_pids, title=r'$p_{ID}$')
plt.tight_layout()
plt.show()
"""

# 3D scatter plot of x, y, and in-focus z
"""
fig, ax = plotting.plot_scatter_3d([df.x, df.y, df.z_f], fig=None, ax=None, elev=5, azim=-40, color=None, alpha=0.75)
plt.show()

# Fit a 2D plane to the in-focus particles
points = np.stack((df.x, df.y, df.z_f)).T
px, py, pz = fit.fit_3d(points, fit_function='plane')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.x, df.y, df.z_f, c=df.z_f)
ax.plot_surface(px, py, pz, alpha=0.2, color='tab:red')
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('y', fontsize=18)
ax.set_zlabel('z', fontsize=18)
ax.view_init(5, -40)

ax.scatter(px[0][0], py[0][0], pz[0][0], color='red')
ax.scatter(px[0][1], py[0][1], pz[0][1], color='blue')
ax.scatter(px[1][0], py[1][0], pz[1][0], color='green')
ax.scatter(px[1][1], py[1][1], pz[1][1], color='purple')

plt.show()
"""