import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import analyze
from utils import plot_collections, bin, modify, plotting

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

# --- --- SETUP

# --- files to read
dataset_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_overlap_noise-level1/'
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/synthetic grid overlap random z nl1/'
# test coords
base_testcoords = 'test_coords/'
idpt1 = 'dx-5-60-5/test_id1_coords_static_grid-overlap-random-z-nl1.xlsx'
spct1 = 'dx-5-60-5/test_id11_coords_SPC_grid-overlap-random-z-nl1.xlsx'
idpt2 = 'dx-7.5-57.5-5/test_id2_coords_static_grid-overlap-random-z-nl1.xlsx'
spct2 = 'dx-7.5-57.5-5/test_id12_coords_SPC_grid-overlap-random-z-nl1.xlsx'
# true coords
true1 = dataset_dir + 'grid-random-z/calibration_input/calib_-15.0.txt'
true2 = dataset_dir + 'grid-random-z/test_input_dx7.5/B0000.txt'
# diameter parameters
path_diameter_params = base_dir + 'test_coords/spct-defocus/calib_spct_pop_defocus_stats.xlsx'
# save ids
save_ids = ['test_id1_coords_static',
            'test_id11_coords_SPC',
            'test_id2_coords_static',
            'test_id12_coords_SPC',
            ]
modifiers = [True, True, False, False]


# --- --- PERCENT DIAMETER OVERLAP
export_percent_diameter_overlap = False
plot_percent_diameter_overlap = False

if plot_percent_diameter_overlap or export_percent_diameter_overlap:

    # --- read each percent diameter overlap dataframe (if available)
    calculate_percent_overlap = False

    for sid, modifier in zip(save_ids, modifiers):

        if calculate_percent_overlap:
            # --- For each test coords, calculate percent diameter overlap
            for test_coord, true_coord, filt, sid in zip([idpt1, spct1, idpt2, spct2],
                                                         [true1, true1, true2, true2],
                                                         modifiers,
                                                         save_ids):

                dfo = analyze.calculate_particle_to_particle_spacing(
                    test_coords_path=base_dir + base_testcoords + test_coord,
                    theoretical_diameter_params_path=path_diameter_params,
                    mag_eff=10,
                    z_param='z_true',
                    zf_at_zero=False,
                    zf_param='zf_from_dia',
                    max_n_neighbors=1,
                    true_coords_path=true_coord,
                    maximum_allowable_diameter=55)

                # filter dx 5 60 5 coords
                if filt:
                    dfo = dfo[(dfo['x'] < 850) | (dfo['x'] > 900)]

                # save to excel
                dfo.to_excel(base_dir + 'percent-overlap/{}_grid-overlap-random-z-nl1_percent_overlap.xlsx'.format(sid),
                             index=False)

        else:
            dfo = pd.read_excel(base_dir + 'percent-overlap/test_coords_percent_overlap/'
                                           '{}_grid-overlap-random-z-nl1_percent_overlap.xlsx'.format(sid))

        # --- --- --- ------ ------ --- EVALUATE RMSE Z

        # --- --- EVALUATE RMSE Z by percent diameter overlap

        # limit percent diameter overlap to -25% (not overlapping here)
        dfo['percent_dx_diameter'] = dfo['percent_dx_diameter'].where(dfo['percent_dx_diameter'] > -0.5, -0.5)

        # binning
        columns_to_bin = ['z_true', 'percent_dx_diameter']
        bin_z = [-27.5, -15, -2.5, 10, 22.5]
        bin_pdo = 8

        dfbicts = analyze.evaluate_2d_bin_local_rmse_z(df=dfo,
                                                       columns_to_bin=columns_to_bin,
                                                       bins=[bin_z, bin_pdo],
                                                       round_to_decimals=[3, 4],
                                                       min_cm=0.5,
                                                       equal_bins=[False, True])

        # --- --- PLOT RMSE Z

        if plot_percent_diameter_overlap:
            # Plot rmse z + number of particles binned as a function of percent diameter overlap for different z bins
            fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches*1.35, size_y_inches*1.5))
            for name, df in dfbicts.items():
                ax.plot(df.bin, df.rmse_z, '-o', label=name)
                ax2.plot(df.bin, df.num_bind, '-o')

            ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
            ax.set_yscale('log')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
            ax2.set_xlabel(r'$\gamma \: $(\%)')
            ax2.set_ylabel(r'$N_{p}$')
            plt.tight_layout()
            plt.savefig(base_dir + 'percent-overlap/{}_rmsez_num-binned_pdo.png'.format(sid))
            plt.show()

            # Average the above plot across z-bins and plot
            fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches*1.35, size_y_inches*1.5))

            dataframe_dfbicts = modify.stack_dficts_by_key(dfbicts, drop_filename=False)
            mean_bins = dataframe_dfbicts.groupby('bin').mean().index
            mean_rmse_z = dataframe_dfbicts.groupby('bin').mean().rmse_z
            sum_num_bind = dataframe_dfbicts.groupby('bin').sum().num_bind

            ax.plot(mean_bins, mean_rmse_z, '-o')
            ax2.plot(mean_bins, sum_num_bind, '-o')

            ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
            ax2.set_xlabel(r'$\gamma \: $(\%)')
            ax2.set_ylabel(r'$N_{p}$')
            plt.tight_layout()
            plt.savefig(base_dir + 'percent-overlap/{}_average_rmsez_num-binned_pdo.png'.format(sid))
            plt.show()

        # --- --- EXPORT RMSE Z TO EXCEL

        dfstack = modify.stack_dficts_by_key(dfbicts, drop_filename=False)
        dfstack.to_excel(base_dir + 'percent-overlap/{}_binned_rmsez_by_z_pdo.xlsx'.format(sid), index=False)

        # --- --- PLOT OTHER METRICS

        # --- calculate the local rmse_z uncertainty
        num_bins = 25
        bin_list = np.round(np.linspace(-1.25, 1, 10), 4)
        min_cm = 0.5
        z_range = [-40.1, 40.1]
        round_to_decimal = 4
        df_ground_truth = None

        # bin by percent diameter overlap
        if plot_percent_diameter_overlap:
            dfob = bin.bin_local_rmse_z(df=dfo, column_to_bin='percent_dx_diameter', bins=bin_list, min_cm=min_cm, z_range=z_range,
                                           round_to_decimal=round_to_decimal, df_ground_truth=df_ground_truth)

            fig, ax = plt.subplots()
            ax.plot(dfob.index, dfob.rmse_z, '-o')
            ax.set_xlabel(r'$\gamma \: $(\%)')
            ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
            plt.tight_layout()
            plt.savefig(base_dir + 'percent-overlap/{}_binned_rmsez_by_pdo.png'.format(sid))
            plt.show()

            # bin by z
            dfobz = bin.bin_local_rmse_z(df=dfo, column_to_bin='z_true', bins=num_bins, min_cm=min_cm, z_range=z_range,
                                           round_to_decimal=round_to_decimal, df_ground_truth=df_ground_truth)

            fig, ax = plt.subplots()
            ax.plot(dfobz.index, dfobz.rmse_z, '-o')
            ax.set_xlabel(r'$z_{true} \:$ ($\mu m$)')
            ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
            plt.tight_layout()
            plt.savefig(base_dir + 'percent-overlap/{}_binned_rmsez_by_z.png'.format(sid))
            plt.show()

        # --- --- CALCULATE RMSE BY PARTICLE TO PARTICLE SPACINGS

        # --- setup
        column_to_split = 'x'
        round_x_to_decimal = 0

        if modifier:
            splits = np.array([93.0, 189.0, 284.0, 380.0, 475.0, 571.0, 666.0, 762.0, 837.0, 930])  # 900.0
            keys = [57.5, 5, 10, 15, 20, 25, 30, 35, 40, 52.5]  # 47.5
        else:
            splits = np.array([79.0, 163.5, 254.0, 348.5, 447.0, 555.5, 665.0, 777.5, 900.0])
            keys = [7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5]

        # --- split df into dictionary
        dfsplicts_gdpyt = modify.split_df_and_merge_dficts(dfo, keys, column_to_split, splits, round_x_to_decimal)

        # --- rmse by z and particle to particle spacing
        column_to_bin = 'z_true'
        num_bins = 20
        min_cm = 0.5
        z_range = [-40.01, 40.01]
        round_to_decimal = 4
        df_ground_truth = None

        # plot
        plt.style.use(['science', 'ieee', 'muted'])
        fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches*1.15, size_y_inches*1.75))

        dfdxbs = []
        for name, dfdx in dfsplicts_gdpyt.items():
            dfdxb = bin.bin_local_rmse_z(dfdx,
                                         column_to_bin=column_to_bin,
                                         bins=num_bins,
                                         min_cm=min_cm,
                                         z_range=z_range,
                                         round_to_decimal=round_to_decimal,
                                         df_ground_truth=df_ground_truth)
            dfdxb = dfdxb.reset_index()
            dfdxb['test_id'] = name

            ax1.plot(dfdxb.bin, dfdxb.rmse_z, '-o', ms=4, label=name)
            ax2.plot(dfdxb.bin, dfdxb.percent_dx_diameter, '-o', ms=4)
            dfdxbs.append(dfdxb)

        ax1.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
        ax2.set_ylabel(r'$\gamma \: (\%)$')
        ax2.set_xlim([-25, -10])
        ax1.set_ylim([0.06, 0.2])
        ax2.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(base_dir + 'percent-overlap/{}_binned_rmsez_by_dx_and_z.png'.format(sid))
        plt.show()

        dfdxbs = pd.concat(dfdxbs, ignore_index=True)
        dfdxbs.to_excel(base_dir + 'percent-overlap/{}_binned_rmsez_by_dx_and_z.xlsx'.format(sid), index=False)

        # --- rmse z by binning x
        plt.style.use(['science', 'ieee', 'std-colors'])
        column_to_split = 'x'
        round_x_to_decimal = 0

        dfmbicts_gdpyt = analyze.calculate_bin_local_rmse_z(dfsplicts_gdpyt, column_to_split, splits, min_cm, z_range,
                                                            round_x_to_decimal, dficts_ground_truth=None)

        # --- plot global uncertainty - gdpyt
        if plot_percent_diameter_overlap:
            xlabel_for_keys = r'$\delta x (pix)$'
            h = 80
            scale_fig_dim = [1, 1]

            fig, ax, ax2 = plotting.plot_dfbicts_global(dfmbicts_gdpyt, parameters='rmse_z', xlabel=xlabel_for_keys, h=h,
                                                        scale=scale_fig_dim)

            plt.tight_layout()
            plt.savefig(base_dir + 'percent-overlap/{}_global_binned_rmsez_by_z.png'.format(sid))
            plt.show()

        # --- --- EXPORT GLOBAL RMSE Z TO EXCEL
        if export_percent_diameter_overlap:
            dfstack = modify.stack_dficts_by_key(dfmbicts_gdpyt, drop_filename=False)
            dfstack.to_excel(base_dir + 'percent-overlap/{}_global_binned_rmsez_by_particle_spacing.xlsx'.format(sid), index=False)


# --- --- COMBINED PARTICLE TO PARTICLE SPACING
plot_particle_spacing = False

if plot_particle_spacing:

    # read files
    read_dir = base_dir + 'percent-overlap/particle-to-particle-spacing/'

    fn1 = 'test_id1_coords_static_global_binned_rmsez_by_particle_spacing'
    fn2 = 'test_id2_coords_static_global_binned_rmsez_by_particle_spacing'
    fn11 = 'test_id11_coords_SPC_global_binned_rmsez_by_particle_spacing'
    fn12 = 'test_id12_coords_SPC_global_binned_rmsez_by_particle_spacing'

    df1 = pd.read_excel(read_dir + fn1 + '.xlsx')
    df2 = pd.read_excel(read_dir + fn2 + '.xlsx')
    df11 = pd.read_excel(read_dir + fn11 + '.xlsx')
    df12 = pd.read_excel(read_dir + fn12 + '.xlsx')

    # merge dataframes
    dfi = pd.concat([df1, df2], ignore_index=True)
    dfs = pd.concat([df11, df12], ignore_index=True)

    # rename 'filename' to dx
    dfi = dfi.rename(columns={'filename': 'dx'})
    dfs = dfs.rename(columns={'filename': 'dx'})

    # sort by dx
    dfi = dfi.sort_values('dx')
    dfs = dfs.sort_values('dx')

    # --- --- PLOT RMSE Z BY PARTICLE SPACING

    # setup
    h = 80
    s = 10

    fig, [ax, axr] = plt.subplots(nrows=2, sharex=True)  # , figsize=(size_x_inches*1.275, size_y_inches))
    axl = axr.twinx()

    # top figure
    ax.scatter(dfi.dx, dfi.rmse_z / h, s=s, label='IDPT')
    ax.plot(dfi.dx, dfi.rmse_z / h)

    ax.scatter(dfs.dx, dfs.rmse_z / h, s=s, label='SPCT')
    ax.plot(dfs.dx, dfs.rmse_z / h)

    ax.scatter(dfi.dx, dfi.rmse_z / h, s=s, color=sciblue)
    ax.plot(dfi.dx, dfi.rmse_z / h, color=sciblue)

    ax.set_ylabel(r'$\overline{\sigma_{z}} / h$')
    ax.set_yscale('log')
    # ax.set_ylim([0.0, 0.32])
    ax.legend(loc='upper right', handletextpad=0.4)  # , bbox_to_anchor=(1.2,1), title=r'$\overline{\sigma_{z}} / h$')

    # bottom figure
    # axl.scatter(dfi.dx, dfi.true_percent_meas, s=s, marker='s', color=sciblue, label=r'$\phi$')
    # axl.plot(dfi.dx, dfi.true_percent_meas, color=sciblue)
    axl.scatter(dfi.dx, dfi.percent_meas, s=s, marker='d', color=sciblue, label=r'$\phi_{ID}$')
    axl.plot(dfi.dx, dfi.percent_meas, color=sciblue, linestyle='--')

    # axr.scatter(dfs.dx, dfs.true_percent_meas, s=s, marker='s', color=scigreen, label=r'$\phi$')
    # axr.plot(dfs.dx, dfs.true_percent_meas, color=scigreen)
    axr.scatter(dfs.dx, dfs.percent_meas, s=s, marker='d', color=scigreen, label=r'$\phi_{ID}$')
    axr.plot(dfs.dx, dfs.percent_meas, color=scigreen, linestyle='--')

    # axl.scatter(dfi.dx, dfi.true_percent_meas, s=s, color=sciblue)
    # axl.plot(dfi.dx, dfi.true_percent_meas, color=sciblue)
    axl.scatter(dfi.dx, dfi.percent_meas, s=s, marker='d', color=sciblue)
    axl.plot(dfi.dx, dfi.percent_meas, color=sciblue, linestyle='--')

    axr.set_ylabel(r'$\phi$')
    axr.set_xlabel(r'$\delta x $ (pix)')
    axr.set_ylim([35, 105])

    axl.set_ylim([35, 105])
    axl.axis('off')

    axl.legend(loc='upper left', bbox_to_anchor=(0.625, 0.6), title='IDPT', handletextpad=0.4, labelspacing=0.25)
    axr.legend(loc='upper left', bbox_to_anchor=(0.785, 0.6), title='SPCT', handletextpad=0.4, labelspacing=0.25)

    plt.tight_layout()
    # plt.savefig(savedir + '/dx_plot_norm_rmse_z_and_percent_meas.png')
    plt.show()

    j = 1