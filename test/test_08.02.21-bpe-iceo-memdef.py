
from os.path import join
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# setup figures
scale_fig_dim = [1, 1]
scale_fig_dim_legend_outside = [1.3, 1]
plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# storage variables
"""
ON_30s:
fps = ['test_id0_0Vapp.xlsx', 'test_id1_150Vapp.xlsx', 'test_id2_300Vapp.xlsx', 'test_id3_450Vapp.xlsx', 'test_id4_600Vapp.xlsx']
vapps = [0, 150, 300, 450, 600]
dframes = [50, 10, 50, 25, 10]

ON_10s_OFF_15s:
fps = ['test_id14_-600Vapp.xlsx', 'test_id13_-450Vapp.xlsx', 'test_id12_-300Vapp.xlsx', 'test_id11_-150Vapp.xlsx',
     'test_id1_150Vapp.xlsx', 'test_id2_300Vapp.xlsx', 'test_id3_450Vapp.xlsx', 'test_id4_600Vapp.xlsx']
vapps = [-600, -450, -300, -150, 150, 300, 450, 600]
dframes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

SWITCH_5s:
fps = ['test_id1_150Vapp.xlsx', 'test_id2_300Vapp.xlsx', 'test_id3_450Vapp.xlsx', 'test_id4_600Vapp.xlsx']
vapps = [150, 300, 450, 600]
dframes = [10, 10, 10, 10]

"""

# parameters
frames_per_second = 34.376
z0 = 45

# file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/08.02.21 - bpe.g2 deflection'
test_dir = join(base_dir, 'test_coords/ON_10s_OFF_15s')
save_dir = join(base_dir, 'analysis')

# test details
fps = ['test_id14_-600Vapp.xlsx', 'test_id13_-450Vapp.xlsx', 'test_id12_-300Vapp.xlsx', 'test_id11_-150Vapp.xlsx']
     # 'test_id1_150Vapp.xlsx', 'test_id2_300Vapp.xlsx', 'test_id3_450Vapp.xlsx', 'test_id4_600Vapp.xlsx']
vapps = [-600, -450, -300, -150]  # , 150, 300, 450, 600]
dframes = [10, 10, 10, 10]  # , 10, 10, 10, 10, 10, 10]
dfids = [14, 13, 12, 11]

# read files
dfs = []
for fp, dframe in zip(fps[0:1], dframes):
     fp = join(test_dir, fp)
     df = pd.read_excel(fp)
     df = df[df['cm'] > 0.9]
     df['time'] = df['frame'] * dframe / frames_per_second
     dfs.append(df)


#

# plot all particle trajectories for each df
analyze_trajectory = True
plot_trajectory = True
plot_avg_trajectory = False
plot_dz_by_time = False
plot_dz_by_vapp = False
plot_rc_time_constant = False

if plot_trajectory or analyze_trajectory:

     if plot_dz_by_time:
          fig, ax = plt.subplots()

     dz_maxs = []
     for dfid, df, vapp in zip(dfids, dfs, vapps):

          # per-particle analysis
          if plot_trajectory:
               fig, ax = plt.subplots()

               pid_list = df.id.unique()
               pid_list.sort()

               dzs = []
               for id in pid_list:

                    pid = df[df['id'] == id]

                    if len(pid.time) > 5:

                         # pid = pid[pid['z'] > z0]

                         t = pid.time.to_numpy()
                         z = pid.z.to_numpy()

                         z -= z0 - 3.3

                         z = np.abs(z)

                         # z_f - z_i
                         #    dz = np.round(z[-1] - z[0], 1)
                         # z_max - z_i
                         #    dz = np.round(np.max(z) - z[0], 1)
                         # z_max - z_min
                         dz = np.round(np.max(z) - np.min(z), 1)
                         dzs.append(dz)

                         # plot
                         ax.scatter(t, z, label=id, s=5)

          # groupby analysis
          if plot_dz_by_time or plot_dz_by_vapp or plot_rc_time_constant or plot_avg_trajectory:
               dfc = df.groupby('id').count()
               dfc = dfc[dfc['frame'] > 15]
               pid_list = dfc.index.to_numpy()
               pid_list = [5]

               df = df[df['id'].isin(pid_list)]
               df['z'] = df['z'] - z0 + 3.3

               dfg = df.groupby('frame').mean()

               t = dfg.time.to_numpy()
               z = dfg.z.to_numpy()

               z = np.abs(z)

               # z_f - z_i
               #    dzs = np.round(z[-1] - z[0], 1)
               # z_max - z_i
               #dzs = np.round(np.max(z) - z[-1], 1)
               #dzs = np.round(np.max(z) - np.min(z), 1)
               dzs = np.round(np.min(z) - z[0], 1)
               dz_maxs.append(dzs)

               # plot avg trajectory
               if plot_avg_trajectory:
                    fig, ax = plt.subplots()
                    ax.scatter(t, z, s=5)

                    ax.set_ylim([0, 20])
                    ax.set_ylabel(r'$z \: (\mu m)$')
                    ax.set_xlim([0, 22.5])
                    ax.set_xlabel(r'$t \: (s)$')

                    ax.grid(alpha=0.125)

                    ax.set_title(
                         r'$\Delta z $' + ' ({} V) = '.format(vapp) + '{} microns'.format(np.round(np.mean(dzs), 2)))

                    # ax.legend(loc='upper left', bbox_to_anchor=(1.01, 0.99, 1, 0), markerscale=2, borderpad=0.1, handletextpad=0.05, borderaxespad=0.1)
                    # ax.legend()

                    plt.tight_layout()

                    plt.savefig(join(save_dir, 'id{}_avg_tracjectories.png'.format(dfid)))
                    plt.show()

               # plot dz by time
               if plot_dz_by_time:
                    z_norm = z - z[0] # - z[0]
                    if vapp == -450:
                         t += 0.25
                    ax.scatter(t, z_norm, s=2.5, label='{} V'.format(vapp))
                    ax.plot(t, z_norm, alpha=1)

          # plot
          if plot_trajectory:
               # ax.scatter(t, z, label=r'$z_{avg}$', s=5)

               ax.set_ylim([0, 20])
               ax.set_ylabel(r'$z \: (\mu m)$')
               ax.set_xlim([0, 22.5])
               ax.set_xlabel(r'$t \: (s)$')
               ax.grid(alpha=0.125)
               ax.set_title(r'$\Delta z_{pp}$' + ' ({} V) = '.format(vapp) + '{} microns'.format(np.round(np.mean(dzs), 2)))
               # ax.legend(loc='upper left', bbox_to_anchor=(1.01, 0.99, 1, 0), markerscale=2, borderpad=0.1, handletextpad=0.05, borderaxespad=0.1)
               #ax.legend()
               plt.tight_layout()
               #plt.savefig(join(save_dir, 'id{}_pid_tracjectories.png'.format(dfid)))
               plt.show()

     # plot dz by time
     if plot_dz_by_time:
          ax.set_ylim([-7.5, 7.5])
          ax.set_ylabel(r'$\Delta z \: (\mu m)$')
          ax.set_xlim([0, 22.5])
          ax.set_xlabel(r'$t \: (s)$')
          ax.grid(alpha=0.125)
          #ax.set_title(r'$\Delta z $' + ' ({} V) = '.format(vapp) + '{} microns'.format(np.round(np.mean(dzs), 2)))
          #ax.legend(loc='upper left', bbox_to_anchor=(1.01, 0.99, 1, 0), markerscale=2, borderpad=0.1, handletextpad=0.05, borderaxespad=0.1)
          ax.legend()
          plt.tight_layout()
          plt.savefig(join(save_dir, 'dz_by_time.png'))
          plt.show()


     # plot max deflection ~ V applied
     if plot_dz_by_vapp:
          fig, ax = plt.subplots()
          vapps = np.abs(vapps[1:])
          dz_maxs = np.abs(dz_maxs[1:])
          ax.scatter(vapps, dz_maxs, s=10)
          ax.plot(vapps, dz_maxs, alpha=0.5, label='data')  # r'$\Delta z_{avg}$')

          #ax.scatter(600, dz_maxs[-1] + .75, s=10, color='tab:red', label='est.', alpha=0.5)
          #ax.plot([450, 600], [dz_maxs[-2], dz_maxs[-1] + .75], color='tab:red', alpha=0.25)

          ax.set_ylim([2, 10])
          ax.set_ylabel(r'$| \Delta z_{pp} | \: (\mu m)$')
          ax.set_xlim([100, 500])
          ax.set_xlabel(r'$| V_{applied} | \: (V)$')

          ax.grid(alpha=0.125)
          ax.legend(loc='upper left')
          #ax.set_title(r'$\Delta z $' + ' ({} V) = '.format(vapp) + '{} microns'.format(np.round(np.mean(dzs), 2)))

          plt.tight_layout()

          plt.savefig(join(save_dir, 'dz_by_Vapp.png'))
          plt.show()


     # plot RC time constant
     """ Need to fit a line to estimate RC time constant. """
     if plot_rc_time_constant:
          fig, ax = plt.subplots()

          # use estimated dz_max for 600 Vapp
          dz_maxs = dz_maxs[:-1]
          dz_maxs.append(18)

          ax.scatter(vapps, dz_maxs, s=10)
          ax.plot(vapps, dz_maxs, alpha=0.5, label=r'$\Delta z_{avg}$')

          ax.scatter(600, 18, s=10, color='tab:red', label='est.', alpha=0.5)
          ax.plot([450, 600], [12.9, 18], color='tab:red', alpha=0.25)

          ax.set_ylim([0, 20])
          ax.set_ylabel(r'$z \: (\mu m)$')
          ax.set_xlim([0, 650])
          ax.set_xlabel(r'$V_{applied} \: (V)$')

          ax.grid(alpha=0.125)
          ax.legend()
          #ax.set_title(r'$\Delta z $' + ' ({} V) = '.format(vapp) + '{} microns'.format(np.round(np.mean(dzs), 2)))

          plt.tight_layout()

          plt.savefig(join(save_dir, 'dz_by_Vapp.png'))
          plt.show()


j = 1