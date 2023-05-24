import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import bin
from utils.plotting import lighten_color

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
NOTE: :
    A. 
    B. 
"""

# ----------------------------------------------------------------------------------------------------------------------
# PART A.
plot_similarity = False
if plot_similarity:

    # --- READ FILES

    # filepaths
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/figure data/grid-overlap-similarity-due-to-overlap/'

    path_read = base_dir + 'data/'
    path_figs = base_dir + 'figs/'

    fp1 = path_read + 'idpt-single-particle-calib/mean_rmse-z_bin-dx.xlsx'
    fp2 = path_read + 'idpt-standard-calib/mean_rmse-z_bin-dx.xlsx'
    fp3 = path_read + 'spct/mean_rmse-z_bin-dx.xlsx'

    df1 = pd.read_excel(fp1)
    df2 = pd.read_excel(fp2)
    df3 = pd.read_excel(fp3)

    # ---

    # --- PROCESSING
    df1 = df1[df1['dx'] < 30]
    df2 = df2[df2['dx'] < 30]
    df3 = df3[df3['dx'] < 30]

    # --- PLOT BY DX
    plot_dx = False
    if plot_dx:
        # cm by dx
        x = 'dx'
        ys = ['cm', 'rmse_z']
        lbls = [r'$c_{m}^{\delta}$', r'$\sigma_{z}^{\delta} \: (\mu m)$']

        for y, lbl in zip(ys, lbls):
            fig, ax = plt.subplots(figsize=(size_x_inches * 1.125, size_y_inches))

            ax.plot(df1[x], df1[y], '-o', color=lighten_color(sciblue, 1.15), label='IDPT 1')
            ax.plot(df2[x], df2[y], '-o', color=sciblue, label='IDPT')
            ax.plot(df3[x], df3[y], '-o', color=scigreen, label='SPCT')

            ax.set_xlabel(r'$\delta x \: (pix.)$')
            ax.set_ylabel(lbl)
            ax.legend(title=r'$\delta x$', loc='upper left',
                      bbox_to_anchor=(1, 1))  # , labelspacing=0.1, handletextpad=0.4, columnspacing=1
            plt.tight_layout()
            plt.savefig(path_figs + '/compare_{}_by_dx.svg'.format(y))
            plt.show()
            plt.close()

        # ---

    # ---

    # --- PLOT DIFFERENCE BETWEEN IDPT AND IDPT SINGLE

    df21 = df2 - df1

    # cm by dx
    x = 'dx.1'
    ys = ['cm', 'rmse_z']
    lbls = [r'$c_{m}^{\delta}$', r'$\sigma_{z}^{\delta} \: (\mu m)$']

    for y, lbl in zip(ys, lbls):
        fig, ax = plt.subplots(figsize=(size_x_inches * 1.125, size_y_inches))

        ax.plot(df21[x], df21[y], '-o', color=sciblue, label='IDPT')

        ax.set_xlabel(r'$\delta x \: (pix.)$')
        ax.set_ylabel(lbl)
        ax.legend(title=r'$\delta x$', loc='upper left',
                  bbox_to_anchor=(1, 1))  # , labelspacing=0.1, handletextpad=0.4, columnspacing=1
        plt.tight_layout()
        plt.savefig(path_figs + '/difference_{}_by_dx.svg'.format(y))
        plt.show()
        plt.close()

    # ---


# ----------------------------------------------------------------------------------------------------------------------
# PART B.
plot_rmse = True
if plot_rmse:

    # --- READ FILES

    # filepaths
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/figure data/grid-overlap-similarity-due-to-overlap/'

    path_read = base_dir + 'data/'
    path_figs = base_dir + 'figs/'

    fp = '-no-dz-overlap_percent_overlap_contour_diameter'

    fp1 = 'idpt-single-particle-calib/idpt'
    fp2 = 'idpt-standard-calib/idpt'
    fp3 = 'spct/spct'
    dirs = [fp1, fp2, fp3]
    p_lbls = ['IDPT 1', 'IDPT', 'SPCT']

    # ---

    # --- PROCESSING
    bin_pdo = np.linspace(0.125, 5.125, 25)
    min_cm = 0.5

    # --- PLOT RMSE Z BY PERCENT DIAMETER OVERLAP
    compare_all = False
    if compare_all:

        # setup
        error_limit = 6.5
        columns_to_bin = ['percent_dx_diameter']
        column_labels = [r'$\tilde{\varphi} \: (\%)$']
        ylim_rmse = [-0.125, 2.795]

        for col, col_lbl in zip(columns_to_bin, column_labels):

            fig, (axr, ax) = plt.subplots(nrows=2, sharex=True)

            for dir, pl in zip(dirs, p_lbls):

                # read
                dfo = pd.read_excel(path_read + dir + fp + '.xlsx')

                # processing
                # dfo = dfo[dfo['error'] < error_limit]

                if pl == 'SPCT':
                    dfo = dfo[dfo['percent_dx_diameter'] < 1.25]

                dfob = bin.bin_local_rmse_z(df=dfo, column_to_bin=col, bins=bin_pdo, min_cm=min_cm,
                                            z_range=None, round_to_decimal=3, df_ground_truth=None)

                axr.plot(dfob.index, dfob.num_bind, '-d', ms=3, label=pl)
                ax.plot(dfob.index, dfob.rmse_z, '-o', ms=3)

            axr.set_ylabel(r'$N_{p} \: (\#)$')
            axr.set_ylim(bottom=0)
            axr.legend()

            ax.set_xlabel(col_lbl)
            ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
            ax.set_ylim(ylim_rmse)

            plt.tight_layout()
            plt.savefig(path_figs + '/compare_binned_rmsez_by_pdo.svg')
            plt.show()
            plt.close()

            # ---

        # ---

    # ---

    # --- PLOT RMSE Z DIFFERENCE (IDPT 1 - IDPT) BY PERCENT DIAMETER OVERLAP
    # read
    dfo1 = pd.read_excel(path_read + fp1 + fp + '.xlsx')
    dfo2 = pd.read_excel(path_read + fp2 + fp + '.xlsx')

    # processing
    col = 'percent_dx_diameter'
    col_lbl = r'$\tilde{\varphi} \: (\%)$'

    dfob1 = bin.bin_local_rmse_z(df=dfo1, column_to_bin=col, bins=bin_pdo, min_cm=min_cm,
                                z_range=None, round_to_decimal=3, df_ground_truth=None)
    dfob2 = bin.bin_local_rmse_z(df=dfo2, column_to_bin=col, bins=bin_pdo, min_cm=min_cm,
                                z_range=None, round_to_decimal=3, df_ground_truth=None)

    # plot
    x = dfob1.index.to_numpy()
    y = dfob1.rmse_z.to_numpy() - dfob2.rmse_z.to_numpy()

    fig, ax = plt.subplots()

    ax.plot(x, y, '-o', ms=3, label='IDPT 1 - IDPT')
    ax.axhline(y=0, linewidth=0.5, linestyle='--', color='black')

    ax.set_xlabel(col_lbl)
    ax.set_ylabel(r'$\Delta \sigma_{z} \: (\mu m)$')
    ax.legend()

    plt.tight_layout()
    plt.savefig(path_figs + '/difference_binned_rmsez_by_pdo.svg')
    plt.show()
    plt.close()

    # ---



print("Analysis completed without errors.")