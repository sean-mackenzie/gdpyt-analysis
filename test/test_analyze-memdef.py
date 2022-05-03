
from os.path import join
from random import sample

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import griddata, CloughTocher2DInterpolator

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# setup figures
scale_fig_dim = [1, 1]
scale_fig_dim_legend_outside = [1.3, 1]
plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)


# parameters
frames_per_second = 34.376


# file paths
base_dir = '/Users/mackenzie/PythonProjects/jupyter/GDPyT/Analyze - pre 2022 experiments'
save_dir = join(base_dir, 'analysis')

# test details
test_name = 'loc11_test1_coords_cnn.csv'


# read files
fp = join(base_dir, test_name)
df = pd.read_csv(fp)

# filters
df = df.dropna()
df = df.drop(columns=['Unnamed: 0'])
df = df.rename(columns={'Frame': 'frame'})

# filter on num_frames
dfc = df.groupby('id').count()
dfc = dfc[dfc['z'] > 50]
pid_counts = dfc.index.to_numpy()
df = df[df['id'].isin(pid_counts)]

# filter on initial frame
dfi = df[df['frame'] == 0]
pid_init = dfi.id.to_numpy()
df = df[df['id'].isin(pid_init)]

# normalize all particles' z-coordinate by initial z-coordinate
df['dz'] = df['z']
for pid in pid_counts:
     z_norm = dfi[dfi['id'] == pid]['z'].mean()
     df['dz'] = df['dz'].where(df['id'] != pid, df['dz'] - z_norm, axis='index')


# plot initial distribution
n = 10
pid_list = sample(list(df.id.unique()), n)
"""fig, ax = plt.subplots()
for pid in pid_list:
     dfpid = df[df['id'] == pid]
     ax.plot(dfpid.frame, dfpid.dz, label=pid)

plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1, 1, 0))
plt.tight_layout()
plt.show()"""

pid_control = np.arange(25, 30, 15)

#fig, ax = plt.subplots()
for pid in pid_control:
     dfpid = df[df['id'] == pid]

     x = dfpid.frame.to_numpy()
     y = dfpid.dz.to_numpy()
     #ax.plot(x, y, label=pid)

     peaks, _ = find_peaks(y, distance=24, prominence=1)
     #ax.scatter(peaks, y[peaks], color='black')

#plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1, 1, 0))
#plt.show()

troughs = []
for i in range(len(peaks)-1):
    troughs.append(int(np.mean([peaks[i], peaks[i+1]])))

peaks_and_troughs = peaks
peaks_and_troughs = np.append(peaks, troughs)
peaks_and_troughs.sort()

"""fig, ax = plt.subplots()

for peak, trough in zip(peaks, troughs):
     dfi = df[df['frame'] == peak]
     dfi = dfi.round({'y': -2})
     dfi = dfi.groupby('y').mean()
     ax.plot(dfi.index, dfi.dz, color='blue')

     dfi = df[df['frame'] == trough]
     dfi = dfi.round({'y': -2})
     dfi = dfi.groupby('y').mean()
     ax.plot(dfi.index, dfi.dz, color='gray', linestyle='--')

plt.show()"""

use = 'CT'
for peak in peaks:
     dfp = df[df['frame'] == peak]

     if use == 'griddata':
          grid_x, grid_y = np.mgrid[0:512:50j, 0:512:50j]
          points = dfp[['x', 'y']].to_numpy()
          values = dfp['dz'].to_numpy()
          grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
          plt.subplots()
          plt.imshow(grid_z.T, extent=(0, 512, 0, 512), origin='lower')
          plt.title('Cubic')
          plt.show()

     elif use == 'CT':
          x = dfp.x.to_numpy()
          y = dfp.y.to_numpy()
          z = dfp.dz.to_numpy()

          X = np.linspace(min(x), max(x), 30)
          Y = np.linspace(min(y), max(y), 30)
          X, Y = np.meshgrid(X, Y)
          interp = CloughTocher2DInterpolator(list(zip(x, y)), z, fill_value=0)
          Z = interp(X, Y)
          plt.pcolormesh(X, Y, Z, shading='auto')
          plt.colorbar()
          plt.scatter(x, y, s=3, color='black')
          plt.axis("equal")
          plt.show()

          fig = plt.figure()
          ax = fig.add_subplot(projection='3d')
          ax.plot_wireframe(X, Y, Z, rstride=0, cstride=3)
          ax.view_init(15, 30)
          plt.show()

          fig, ax = plt.subplots()
          for i in np.arange(0, np.shape(Y)[0], 5):
               j = i
               ax.plot(Y[:, i], Z[:, i])
          plt.show()


          j=1




j = 1