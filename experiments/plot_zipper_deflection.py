from os.path import join
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

# setup figures
scale_fig_dim = [1, 1]
scale_fig_dim_legend_outside = [1.3, 1]
plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# SETUP I/O
# ----------------------------------------------------------------------------------------------------------------------

# parameters
frames_per_second = 25
z0 = 45

# file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/zipper'
test_dir = join(base_dir, 'test_coords/iter3')
save_dir = join(base_dir, 'analysis')

# test details
fps = ['test_id1_coords_50V.xlsx', 'test_id2_coords_75V.xlsx', 'test_id3_coords_100V.xlsx',
       'test_id4_coords_125V.xlsx', 'test_id5_coords_150V.xlsx', 'test_id6_coords_175V.xlsx']
vapps = [50, 75, 100, 125, 150, 175]  # , 150, 300, 450, 600]
dfids = [1, 2, 3, 4, 5, 6]

# read files
dfs = []
for fp in fps:
    fp = join(test_dir, fp)
    df = pd.read_excel(fp)

    df = df[df['cm'] > 0.9]

    df['time'] = df['frame'] / frames_per_second

    dfs.append(df)

# PER-TEST
plot_fig = False
for vapp, df in zip(vapps, dfs):

    # scatter: z of time
    if plot_fig:
        fig, ax = plt.subplots()
        sc = ax.scatter(df.time, df.z, c=df.id, s=1)
        cbar = plt.colorbar(sc)
        cbar.ax.set_title(r'$p_{ID}$')
        ax.set_xlabel(r'$t \: (s)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        # ax.set_title(r'$V_{applied} = $' + ' {} '.format(vapp) + r'$V$')
        ax.set_title(r'$ID_{test} = $' + ' {} '.format(vapp))
        plt.tight_layout()
        plt.savefig(join(save_dir, 'id{}_scatter-z-of-time.png'.format(vapp)))
        plt.show()

    # groupby: line: z of time
    if plot_fig:
        fig, ax = plt.subplots()
        dfg = df.groupby('frame').mean()
        ax.plot(dfg.time, dfg.z, '-o')
        ax.set_xlabel(r'$t \: (s)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.set_title(r'$V_{applied} = $' + ' {} '.format(vapp) + r'$V$')
        plt.savefig(join(save_dir, 'per-test/vapp{}_grouped-line-z-of-time.png'.format(vapp)))
        plt.show()


    def multi_scatter(dfg, save_id_tests):
        fig = plt.figure(figsize=(6.5, 5))
        for i, v in zip(np.arange(1, 5), [45, 0, 315, 270]):
            ax = fig.add_subplot(2, 2, i, projection='3d')
            sc = ax.scatter(dfg.x, dfg.y, dfg.z, c=dfg.z, s=4, vmin=90, vmax=155)
            ax.view_init(5, v)
            ax.patch.set_alpha(0.0)
            if i == 2:
                plt.colorbar(sc, shrink=0.5)
                ax.get_xaxis().set_ticks([])
                ax.set_ylabel(r'$y \: (\mu m)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            elif i == 4:
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(r'$x \: (\mu m)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            else:
                ax.set_xlabel(r'$x \: (\mu m)$')
                ax.set_ylabel(r'$y \: (\mu m)$')
                ax.get_zaxis().set_ticklabels([])
        plt.suptitle("Time = {}".format(save_id_tests), y=0.875)
        plt.subplots_adjust(hspace=-0.1, wspace=0.15)
        plt.show()


    # multi view scatter
    if plot_fig:
        for fr in [1, 100, 125, 150]:
            dff = df[df['frame'] == fr]
            multi_scatter(dff, fr)

    # deflection with radius
    if plot_fig:
        fig, ax = plt.subplots()
        for fr in [1, 80, 110, 160, 190]:
            dff = df[(df['frame'] == fr)]  # & (df['p_type'] == 1)]
            # dff = dff.sort_values('r')
            ax.plot(dff.x, dff.z, '-o', ms=3, label=np.round(fr / frames_per_second, 1))
        ax.legend(title=r'$t \: (s)$')
        ax.set_title(r'$ID_{test} = $' + ' {} '.format(vapp))
        ax.set_xlabel(r'$x \: (\mu m)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        plt.tight_layout()
        plt.savefig(join(save_dir, 'id{}_scatter-z-of-time-and-radius.png'.format(vapp)))
        plt.show()

    # groupby: scatter: z of time
    if plot_fig:
        pids = df.id.unique()
        for pid in pids:
            dfpid = df[df['id'] == pid]

            fig, ax = plt.subplots()
            ax.plot(dfpid.time, dfpid.z, '-o')
            ax.set_xlabel(r'$t \: (s)$')
            ax.set_ylabel(r'$z \: (\mu m)$')
            ax.set_title(r'$z(p_{ID}, V_{applied}) = $' + ' ({}, {})'.format(pid, vapp))
            # plt.savefig(join(save_dir, 'per-test/{}V/per-particle/pid{}-line-z-of-time.png'.format(vapp, pid)))
            plt.show()

    # 3D scatter: z of x, time
    if plot_fig:
        times = df.time.unique()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        for time in times:
            dft = df[df['time'] == time]

            sc = ax.scatter(dft.time, dft.x, dft.z, c=dft.z, s=3)

        ax.view_init(10, 135)
        ax.set_xlabel(r'$t \: (s)$')
        ax.set_ylabel(r'$x \: (pix)$')
        ax.set_zlabel(r'$z \: (\mu m)$')
        ax.set_title(r'$z(V_{applied} = $' + ' {}'.format(vapp) + r'$)$')
        plt.savefig(join(save_dir, 'per-test/{}V/scatter-3d-z-of-time-x.png'.format(vapp)))
        plt.close()

    # 3D scatter: z of x, time
    if plot_fig:

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        clrs = np.arange(0, len(gids))

        for gid, clr in zip(gids, clrs):
            dfgid = df[df['id'] == gid].sort_values('time')
            ax.plot(dfgid.time, dfgid.x, dfgid.z, c=cm.coolwarm(clr / len(gids)))

        ax.view_init(10, 135)
        ax.set_xlabel(r'$t \: (s)$')
        ax.set_ylabel(r'$x \: (pix)$')
        ax.set_zlabel(r'$z \: (\mu m)$')
        ax.set_zlim([18, 52])
        ax.set_title(r'$z(V_{applied} = $' + ' {}'.format(vapp) + r'$)$')
        plt.savefig(join(save_dir, 'per-test/{}V/line-3d-z-of-time-x.png'.format(vapp)))
        plt.show()

# PER-COLLECTION
"""for df in dfs:
    dfgg = df.groupby('id').mean()
    dfstd = df.groupby('id').std()
    dfcc = df.groupby('id').count()
    j = 1"""

# plot per-particle traces
gids = np.arange(100)
if plot_fig:
    for gid in gids:
        fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches))
        for vapp, df in zip(vapps, dfs):
            dfgid = df  # df[df['id'] == gid]
            ax.plot(dfgid.time, dfgid.z, '-o', label=vapp)
        ax.set_xlabel(r'$t \: (s)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.set_ylim([13, 52])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$V_{applied}$')
        ax.set_title(r'$z(p_{ID} \: = $' + ' {}'.format(gid) + r'$)$')
        plt.savefig(join(save_dir, 'per-collection/per-particle/pid{}-compare-z-of-time.png'.format(gid)))
        plt.close()

# plot test average deflection
if not plot_fig:


    for vapp, df in zip(vapps, dfs):
        fig, ax = plt.subplots()
        dfstd = df.groupby('id').std().reset_index()
        dfstd = dfstd.sort_values('z', ascending=False)
        gids = dfstd.iloc[0:20].id.to_numpy()
        # z_std_mean = dfstd.z.mean() * 1.5
        # gids = dfstd[dfstd['z'] > z_std_mean].id.to_numpy()
        dfgid = df[df.id.isin(gids)]
        # dfgid = dfgid[dfgid['z'] > 30]
        dfgid= dfgid.round({'frame': -1})
        dfgid = dfgid.groupby('frame').mean()
        if vapp == 175:
            ax.plot(dfgid.time, np.abs(dfgid.z - 25), '-o', ms=3, label=vapp)
        else:
            ax.plot(dfgid.time, np.abs(dfgid.z - 23), '-o', ms=3, label=vapp)

        ax.set_xlabel(r'$t \: (s)$')
        ax.set_ylabel(r'$z_{avg.}(p^{\sigma}_{ID}) \: (\mu m)$')
        # ax.set_ylim([32, 52])
        ax.legend(title=r'$V_{applied}$')
        plt.tight_layout()
        plt.savefig(join(save_dir, 'per-collection/vapp{}-z-of-time.png'.format(vapp)))
        plt.show()

j = 1