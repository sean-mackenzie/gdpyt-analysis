# test bin, analyze, and plot functions
from os.path import join
import ast
import numpy as np
import pandas as pd
import analyze
from utils import io, bin, plotting, modify
import filter
from tracking import plotting as trackplot

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# ------------------------------------------------
"""
Notes on SciencePlots colors:
    std-colors: 7 colors
    muted: 10 colors
    high-vis: 7 colors + 7 linestyles
"""

# ------------------------------------------------
# formatting
plt.style.use(['science', 'ieee', 'std-colors'])
scale_fig_dim = [1, 1]
scale_fig_dim_outside_x_legend = [1.3, 1]
legend_loc = 'best'

fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

sciblue = '#0C5DA5'
scigreen = '#00B945'

# ------------------------------------------------
# read files
datasets = ['synthetic random density uniform z nl1']
save_ids = ['random uniform z']

test_id = 0
dataset = datasets[test_id]
save_id = save_ids[test_id]

# read .xlsx result files to dictionary
base_path = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/{}'.format(dataset)
read_path_name = join(base_path, 'test_coords')
path_name = join(base_path, 'figs')
save_path_name = join(base_path, 'results')
settings_sort_strings = ['settings_id', '_coords_']
test_sort_strings = ['test_id', '_coords_']
filetype = '.xlsx'

# ------------------------------------------------

# bin data for uncertainty assessment
column_to_bin_and_assess = 'z_true'
bins = 1
round_z_to_decimal = 6

# filters for binning
h = 80
z_range = [-40.001, 40.001]
min_cm = 0.5

# assess convergence
inspect_convergence_of_key = 11.0
convergence_increments = 3

# identify data
keys = [1, 2.5, 5, 7.5, 10]  # densities: [1e-4, 2.5e-4, 5e-4, 7.5e-4, 10e-4]
keys_labels = ['1e-3', '2.5e-3', '5e-3', '7.5e-3', '10e-3']

# split dict by key
inspect_gdpyt_by_key = 1.0
inspect_spc_by_key = 11.0
filter_keys = 0  # Note: all values greater than this will be collected
split_keys = 6  # split static (less than) and SPC (greater than)

# -----------------------
# i/o
#save_id = save_id + ' cm={}'.format(min_cm)
results_drop_columns = ['frame', 'id', 'stack_id', 'z_true', 'z', 'x', 'y', 'x_true', 'y_true']

# -----------------------
# formatting figures
save_plots = False
show_plots = True
save_id = save_ids[test_id] + ' cmin={}'.format(min_cm)

# compare static and spc
labels_compare = ['GDPyT', 'GDPT']
colors_compare = None
# compare all
ylim_compare_all = [-0.0005, 0.25]
ylim_percent_true_measured_compare_all = [0, 105]
ylim_percent_measured_compare_all = [0, 105]
ylim_cm_compare_all = [min_cm, 1.01]
# compare filtered
ylim_compare = [-0.0005, 0.15]
ylim_percent_true_measured_compare = [0, 105]
ylim_percent_measured_compare = [0, 105]
ylim_cm_compare = [min_cm, 1.01]

# local
labels_local = [lbl for lbl in keys if lbl > filter_keys]
labels_local.sort()
colors_local = None
linestyles = ['-', '--']
ylim_gdpyt = [-0.0005, 0.085]
ylim_spc = [-0.005, 0.5]
ylim_percent_true_measured_gdpyt = [0, 105]
ylim_percent_true_measured_spc = [0, 105]


# end setup
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# read files

dficts = io.read_files('df', read_path_name, test_sort_strings, filetype, startswith=test_sort_strings[0])
dfsettings = io.read_files('dict', read_path_name, settings_sort_strings, filetype, startswith=settings_sort_strings[0],
                           columns=['parameter', 'settings'], dtype=str)
dficts_ground_truth = io.read_ground_truth_files(dfsettings)

# ---------------------------------------------------------------

# filter out particle ID's > 170 (because likely due to image segmentation problems)
#dficts = filter.dficts_filter(dfictss=dficts, keys=['z_true'], values=[-25], operations=['greaterthan'], copy=True)
#dficts_ground_truth = filter.dficts_filter(dfictss=dficts_ground_truth, keys=['z'], values=[-25], operations=['greaterthan'], copy=True)

# ----------------------------------------------------------------------------------------------------------------------

# start analysis

# ---------------------------------------------------------------

# calculate results by cm_sweep for a single density but different z_true's
analyze_cm_sweep = False

if analyze_cm_sweep:
    # ------------------------
    # setup

    # cm sweep
    cm_i = 0.5
    cm_f = 0.995
    cm_steps = 10
    """bin_z_steps = cm_steps
    
    df = dficts[5.0]
    dfb = bin.bin_by_column(df, column_to_bin='z_true', number_of_bins=bin_z_steps, round_to_decimal=5)
    z_bins = dfb.bin.unique()
    
    keys_splits = z_bins # [-30.0, -10.0, 10.0, 30.0]
    """


    # legend labels
    labels_cm_gdpyt = ['-40:-20', '-20:0', '0:20', '20:40']
    # axes labels
    xlabel_for_cm_sweep_keys = r'$c_{m}$'
    dft_number_ylabel = r'$\frac{\phi_{ID}^2}{\phi}$'
    # axes sizing
    scale_fig_dim_nrows = [1, 1.25]
    # axes limits
    ylim_cm_gdpyt = [0.004, 0.026]
    ylim_global_cm_sweep = [-0.0005, 0.1505]
    ylim_global_nd_tracking = [-0.0005, 1.5]
    ylim_per_measure_bin_z = [99.2, 100.1]
    ylim_global_nd_tracking = [0.95, 1.42]
    # modifiers
    smooth_plots = True
    export_grouped_bin_z_data = False
    export_bin_z_data = False
    save_bin_z_plots = False
    save_dft_plots = False
    # 3d plots
    cmap = 'coolwarm'
    ylbl = r'$z$'
    xlbl = r'$c_{m}$'
    zlbl = r'$\phi_{ID}$'  # r'$\sigma_{z} / h$'
    elev = 90  # 35
    azi = 90  # 100
    vmax = 100
    vmin = 0.0

    if cm_steps < 100:
        contour_stride = 100
    else:
        contour_stride = cm_steps

    # ------------------------


    # ------------------------
    # data processing

    # get dataframe
    """df_gdpyt = dficts[list(dficts.keys())[0]]
    df_gdpyt_ground_truth = dficts_ground_truth[list(dficts.keys())[0]]
    # bin-and-split dataframe by z_true into dictionary
    zicts = modify.split_df_and_merge_dficts(df_gdpyt, keys=keys_splits, column_to_split='z_true', splits=keys_splits, round_to_decimal=round_z_to_decimal)
    zicts_ground_truth = modify.split_df_and_merge_dficts(df_gdpyt_ground_truth, keys=keys_splits, column_to_split='z', splits=keys_splits, round_to_decimal=round_z_to_decimal)"""
    # calculate z-uncertainty per c_m
    dfz_gdpyt = analyze.calculate_results_by_cm_sweep(dficts, dficts_ground_truth, cm_steps, z_range, cm_i=cm_i)
    keys_splits = [1.0, 2.0, 3.0, 4.0, 5.0]
    cmicts = modify.split_df_and_merge_dficts(dfz_gdpyt, keys=keys_splits, column_to_split='id', splits=keys_splits)


    # concat all bin_z dicts:
    """
    # get dataframe
    df_gdpyt = dficts[list(dficts.keys())[0]]
    df_gdpyt_ground_truth = dficts_ground_truth[list(dficts.keys())[0]]
    # bin-and-split dataframe by z_true into dictionary
    zicts = modify.split_df_and_merge_dficts(df_gdpyt, keys=keys_splits, column_to_split='z_true', splits=keys_splits, round_to_decimal=round_z_to_decimal)
    zicts_ground_truth = modify.split_df_and_merge_dficts(df_gdpyt_ground_truth, keys=keys_splits, column_to_split='z', splits=keys_splits, round_to_decimal=round_z_to_decimal)
    # calculate z-uncertainty per c_m
    dfz_gdpyt = analyze.calculate_results_by_cm_sweep(zicts, zicts_ground_truth, cm_steps, z_range, cm_i=cm_i)
    cmicts = modify.split_df_and_merge_dficts(dfz_gdpyt, keys=keys_splits, column_to_split='id', splits=keys_splits)
    
    dfz_grouped = modify.stack_dficts_by_key(cmicts, drop_filename=False)
    
    # ------------------------
    # data export
    
    if export_grouped_bin_z_data:
        io.export_df_to_excel(dfz_grouped,
                              path_name=join(save_path_name, save_id + '_gdpyt_grouped-z-bin_cm_sweep_measurement_results'),
                              include_index=True, index_label='cm_threshold', filetype='.xlsx',
                              drop_columns=None)
    
    if export_bin_z_data:
        for name, df in cmicts.items():
            io.export_df_to_excel(df,
                                  path_name=join(save_path_name, save_id + '_gdpyt_z-bin{}_cm_sweep_measurement_results'.format(name)),
                                  include_index=True, index_label='cm_threshold', filetype='.xlsx',
                                  drop_columns=None)
    """

    # ------------------------
    # data visualization

    # plot measurement uncertainty surface: z-unc. ~ function(c_m, z)
    """
    y = dfz_grouped.index.to_numpy()
    x = dfz_grouped.filename.to_numpy()
    z = np.round(dfz_grouped.percent_meas.to_numpy(), 1)  # z = dfz_grouped.rmse_z.to_numpy() / h
    
    X = np.reshape(x, (cm_steps, bin_z_steps))
    Y = np.reshape(y, (cm_steps, bin_z_steps))
    Z = np.reshape(z, (cm_steps, bin_z_steps))
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, contour_stride, cmap='binary')
    ax.set_ylabel(xlbl)
    ax.set_xlabel(ylbl)
    ax.set_zlabel(zlbl)
    ax.view_init(elev, azi)  # (60, 35) elevation, azimuthal: elevation is above x-y plane and azimuthal is rotation CC about z-axis.
    plt.tight_layout()
    plt.savefig(join(path_name, save_id + '_percent_meas_contours_cm_z.png'))  # '_uncertainty_contours_cm_z.png'
    
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, contour_stride, cmap=cmap)
    ax.set_ylabel(xlbl)
    ax.set_xlabel(ylbl)
    ax.set_zlabel(zlbl)
    ax.view_init(elev, azi)
    plt.tight_layout()
    plt.savefig(join(path_name, save_id + '_percent_meas_contours_cm_z_colored.png'))
    
    # mask z
    zz = np.ma.masked_where(np.isnan(z), z)
    ZZ = np.reshape(zz, (cm_steps, bin_z_steps))
    
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, ZZ, rstride=1, cstride=1, linewidth=1, antialiased=False,
                    cmap=cmap, edgecolor='none', vmin=vmin, vmax=vmax)
    ax.set_ylabel(xlbl)
    ax.set_xlabel(ylbl)
    ax.set_zlabel(zlbl)
    ax.view_init(elev, azi)
    plt.tight_layout()
    plt.savefig(join(path_name, save_id + '_percent_meas_surface_cm_z_colored.png'))"""

    save_bin_z_plots = True
    if save_bin_z_plots:
        """parameter = ['rmse_z', 'percent_meas']
        fig, ax, ax2 = plotting.plot_dfbicts_local(cmicts, parameter, h, scale=scale_fig_dim_nrows,
                                                   xlabel=xlabel_for_cm_sweep_keys, scatter_on=False, nrows=2)
        ax.set_ylim(ylim_cm_gdpyt)
        ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$')
        ylim_per_measure_bin_z = [98.9, 100.1]
        ax2.set_ylim(ylim_per_measure_bin_z)
        labels_cm_gdpyt = [1e-4, 2.5e-4, 5e-4, 7.5e-4, 10e-4]
        ax2.legend(labels_cm_gdpyt, loc='lower left')
        plt.tight_layout()
        plt.savefig(join(path_name, save_id + '_cm_sweep_density_rmse_z_and_percent_meas.png'))
        plt.show()"""
        xy_bin_length = 20
        particles_per_xy_bin = 1024 ** 2 / xy_bin_length ** 2
        cmicts = modify.dficts_scale(cmicts, columns=['num_meas'], multipliers=[1/particles_per_xy_bin])
        cmicts = modify.dficts_new_column(cmicts, new_columns=['rmse_z_norm'], columns=['rmse_z'], multipliers=['num_meas'])

        parameter = ['rmse_z', 'rmse_z_norm']
        fig, ax, ax2 = plotting.plot_dfbicts_local(cmicts, parameter, h, scale=scale_fig_dim_nrows,
                                                   xlabel=xlabel_for_cm_sweep_keys, scatter_on=False, nrows=2)
        ax.set_ylim(ylim_cm_gdpyt)
        ax2.set_ylabel(r'$\frac{\#\: particles}{20\: \mu m^2}$')
        ylim_per_measure_bin_z = [0, 0.5]  # [0, 75]
        ax2.set_ylim(ylim_per_measure_bin_z)
        labels_cm_gdpyt = [1e-4, 2.5e-4, 5e-4, 7.5e-4, 10e-4]
        #ax2.legend(labels_cm_gdpyt, loc='lower left')
        plt.tight_layout()
        plt.savefig(join(path_name, save_id + '_cm_sweep_density_rmse_z_and_num_meas.png'))
        plt.show()

    if save_dft_plots:
        parameter = ['dft_number']
        fig, ax = plotting.plot_dfbicts_local(cmicts, parameter, h=1, scale=scale_fig_dim, scatter_on=False,
                                                   xlabel=xlabel_for_cm_sweep_keys, ylabel=dft_number_ylabel, semilogx=False)
        ax.set_ylim(ylim_global_nd_tracking)
        ax.legend(labels_cm_gdpyt, loc='lower left', title=r'$bin_{z}$')
        plt.tight_layout()
        plt.savefig(join(path_name, save_id + '_cm_sweep_density_DfT_number.png'))
        plt.show()



    # calculate_results_by_cm_sweep
    """
    # sweep c_m
    xlabel_for_cm_sweep_keys = r'$c_{m}$'
    dft_number_ylabel = r'$\frac{\phi_{ID}^2}{\phi}$'
    ylim_global_cm_sweep = [-0.0005, 0.1505]
    ylim_global_nd_tracking = [-0.0005, 1.5]
    smooth_plots = True
    
    df_gdpyt = analyze.calculate_results_by_cm_sweep(dficts, dficts_ground_truth, cm_steps, z_range, take_mean_of_all=False)
    keys_splits = np.linspace(1, len(dficts), len(dficts))
    cmicts = modify.split_df_and_merge_dficts(df_gdpyt, keys=keys_splits, column_to_split='id', splits=keys_splits)
    
    for name, df in cmicts.items():
        io.export_df_to_excel(df,
                              path_name=join(save_path_name, save_id + '_gdpyt_pd{}_cm_sweep_measurement_results'.format(name)),
                              include_index=True, index_label='cm_threshold', filetype='.xlsx',
                              drop_columns=None)
    
    ylim_cm_gdpyt = [-0.0005, 0.05]
    labels_cm_gdpyt = [1e-4, 2.5e-4, 5.0e-4, 7.5e-4, 10e-4]
    parameter = ['rmse_z', 'percent_meas']
    fig, ax, ax2 = plotting.plot_dfbicts_local(cmicts, parameter, h, scale=scale_fig_dim,
                                               xlabel=xlabel_for_cm_sweep_keys, scatter_on=False)
    ax.set_ylim(ylim_cm_gdpyt)
    ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$')
    ax2.set_ylim([0, 101])
    ax.legend(labels_cm_gdpyt, loc='upper left')
    plt.tight_layout()
    plt.savefig(join(path_name, save_id + '_cm_sweep_density_rmse_z_and_percent_meas.png'))
    plt.show()
    
    ylim_global_nd_tracking = [0.4, 1.6]
    parameter = ['dft_number']
    fig, ax = plotting.plot_dfbicts_local(cmicts, parameter, h=1, scale=scale_fig_dim, scatter_on=False,
                                               xlabel=xlabel_for_cm_sweep_keys, ylabel=dft_number_ylabel)
    ax.set_ylim(ylim_global_nd_tracking)
    ax.legend(labels_cm_gdpyt, loc='lower left')
    plt.tight_layout()
    plt.savefig(join(path_name, save_id + '_cm_sweep_density_DfT_number.png'))
    plt.show()
    """

# ---------------------------------------------------------------

# compare static and SPC - local
analyze_local_static_and_spc = False

if analyze_local_static_and_spc:

    if save_plots or show_plots:

        # plot methods comparison local results
        parameter = 'rmse_z'
        fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h, scale=scale_fig_dim)
        ax.set_ylim(ylim_compare_all)
        #ax.legend(labels_compare, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=2)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_compare_all_local_rmse_z.png'))
        if show_plots:
            plt.show()

        parameter = ['rmse_z', 'true_percent_meas']
        fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h, scale=scale_fig_dim)
        ax.set_ylim(ylim_compare_all)
        ax2.set_ylabel(r'$\phi\left(z\right)$')
        ax2.set_ylim(ylim_percent_true_measured_compare)
        #ax.legend(labels_compare, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=2)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_compare_all_local_rmse_z_and_true_percent_meas.png'))
        if show_plots:
            plt.show()

        parameter = ['rmse_z', 'percent_meas']
        fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h, scale=scale_fig_dim)
        ax.set_ylim(ylim_compare_all)
        ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$')
        ax2.set_ylim(ylim_percent_measured_compare)
        #ax.legend(labels_compare, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=2)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_compare_all_local_rmse_z_and_percent_meas.png'))
        if show_plots:
            plt.show()

        parameter = ['rmse_z', 'cm']
        fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h, scale=scale_fig_dim)
        ax.set_ylim(ylim_compare_all)
        ax2.set_ylabel(r'$c_{m}$')
        ax2.set_ylim(ylim_cm_compare)
        #ax.legend(labels_compare, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=2)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_compare_all_local_rmse_z_and_cm.png'))
        if show_plots:
            plt.show()
    """
    
    
    # compare ONLY static or SPC - local
    """
    if save_plots or show_plots:

        # -----------------------------
        # split dataframes into subsets by method

        # get dictionary of only gdpyt
        gdpyt_ids = [xk for xk, xv in dfbicts.items() if xk < split_keys]
        gdpyt_dfs = [xv for xk, xv in dfbicts.items() if xk < split_keys]
        dfbicts_gdpyt = {gdpyt_ids[i]: gdpyt_dfs[i] for i in range(len(gdpyt_ids))}

        # get dictionary of only gdpt
        gdpt_ids = [xk for xk, xv in dfbicts.items() if xk > split_keys]
        gdpt_dfs = [xv for xk, xv in dfbicts.items() if xk > split_keys]
        dfbicts_spc = {gdpt_ids[i]: gdpt_dfs[i] for i in range(len(gdpt_ids))}

        # ----------------------------------------------------------
        # plot local - gdpyt

        parameter = 'rmse_z'
        fig, ax = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, scale=scale_fig_dim)
        ax.set_ylim(ylim_gdpyt)
        ax.legend(labels_local, loc=legend_loc)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_static_local_rmse_z.png'))
        if show_plots:
            plt.show()

        parameter = ['rmse_z', 'true_percent_meas']
        fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, scale=scale_fig_dim_outside_x_legend)
        ax.set_ylim(ylim_gdpyt)
        ax2.set_ylabel(r'$\phi\left(z\right)$')
        ax2.set_ylim(ylim_percent_true_measured_gdpyt)
        ax.legend(labels_local, loc='upper left', bbox_to_anchor=(1.2, 1), fancybox=True, shadow=False, ncol=1)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_static_local_rmse_z_and_true_percent_meas.png'))
        if show_plots:
            plt.show()

        parameter = ['rmse_z', 'percent_meas']
        fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, scale=scale_fig_dim)
        ax.set_ylim(ylim_gdpyt)
        ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$')
        ax2.set_ylim([0, 101])
        ax.legend(labels_local, loc=legend_loc)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_static_local_rmse_z_and_percent_meas.png'))
        if show_plots:
            plt.show()

        parameter = ['rmse_z', 'cm']
        fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, scale=scale_fig_dim)
        ax.set_ylim(ylim_gdpyt)
        ax2.set_ylabel(r'$c_{m}$')
        ax2.set_ylim([min_cm, 1.01])
        ax.legend(labels_local, loc=legend_loc)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_static_local_rmse_z_and_cm.png'))
        if show_plots:
            plt.show()

        # plot local - SPC
        parameter = 'rmse_z'
        fig, ax = plotting.plot_dfbicts_local(dfbicts_spc, parameter, h, scale=scale_fig_dim)
        ax.set_ylim(ylim_spc)
        ax.legend(labels_local, loc=legend_loc)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_SPC_local_rmse_z.png'))
        if show_plots:
            plt.show()

        parameter = ['rmse_z', 'true_percent_meas']
        fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_spc, parameter, h, scale=scale_fig_dim_outside_x_legend)
        ax.set_ylim(ylim_spc)
        ax2.set_ylabel(r'$\phi\left(z\right)$')
        ax2.set_ylim(ylim_percent_true_measured_spc)
        ax.legend(labels_local, loc='upper left', bbox_to_anchor=(1.2, 1), fancybox=True, shadow=False, ncol=1)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_SPC_local_rmse_z_and_true_percent_meas.png'))
        if show_plots:
            plt.show()

        parameter = ['rmse_z', 'percent_meas']
        fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_spc, parameter, h, scale=scale_fig_dim_outside_x_legend)
        ax.set_ylim(ylim_spc)
        ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$')
        ax2.set_ylim([0, 101])
        ax.legend(labels_local, loc='upper left', bbox_to_anchor=(1.2, 1), fancybox=True, shadow=False, ncol=1)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_SPC_local_rmse_z_and_percent_meas.png'))
        if show_plots:
            plt.show()

        parameter = ['rmse_z', 'cm']
        fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_spc, parameter, h, scale=scale_fig_dim_outside_x_legend)
        ax.set_ylim(ylim_spc)
        ax2.set_ylabel(r'$c_{m}$')
        ax2.set_ylim([min_cm, 1.01])
        ax.legend(labels_local, loc='upper left', bbox_to_anchor=(1.2, 1), fancybox=True, shadow=False, ncol=1)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_SPC_local_rmse_z_and_cm.png'))
        if show_plots:
            plt.show()


# ---------------------------------------------------------------

# Notes on particle densities and mean particle to particle spacing
"""
densities: 10E-4 - 1E-4:
num_particles: 1048, 786, 524, 262, 131
"""

# mean z-uncertainty by method (static v. SPC)
analyze_mean_static_and_spc = True
export_results = True

if analyze_mean_static_and_spc:

    # global
    labels_global = ['IDPT', 'IDPT', 'SPCT', 'SPCT']
    xlabel_for_keys = r'$\overline{\delta x} \:$ (pix)'  # r'$\rho_{p} \: (10^{-3})$'
    ylim_global = [-0.0005, 0.2505]
    ylim_percent_true_measured_global = [0, 105]
    ylim_percent_measured_global = [0, 105]
    ylim_cm_global = [min_cm, 1.01]

    if save_plots or show_plots:

        # diameter parameters
        dataset_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_overlap_noise-level1/'
        path_diameter_params = dataset_dir + 'grid/results/calibration-SPC-spct-no_noise_cal/calib_spct_pop_defocus_stats_GridOverlapSPC_calib_nll1_SPC_no_noise_cal.xlsx'

        dficts = analyze.calculate_dict_particle_spacing(dficts,
                                                         theoretical_diameter_params_path=path_diameter_params,
                                                         mag_eff=10,
                                                         max_n_neighbors=5,
                                                         dficts_ground_truth=dficts_ground_truth,
                                                         maximum_allowable_diameter=55)

        # -----------------------------
        # split dictionary into subsets by method

        # get only gdpyt
        dficts_gdpyt, dficts_gdpyt_ground_truth = modify.split_dficts(dficts, dficts_ground_truth, key_to_split=split_keys,
                                                          get_less_than=True, new_keys=keys)

        dficts_spc, dficts_spc_ground_truth = modify.split_dficts(dficts, dficts_ground_truth, key_to_split=split_keys,
                                                                      get_less_than=False, new_keys=keys)

        # calculate mean rmse_z
        dfmbicts_gdpyt = analyze.calculate_bin_local_rmse_z(dficts_gdpyt, column_to_bin_and_assess, bins, min_cm, z_range,
                                                     round_z_to_decimal, dficts_ground_truth=dficts_gdpyt_ground_truth)
        dfmbicts_spc = analyze.calculate_bin_local_rmse_z(dficts_spc, column_to_bin_and_assess, bins, min_cm, z_range,
                                                     round_z_to_decimal, dficts_ground_truth=dficts_spc_ground_truth)

        # -----------------------------

        # export to excel
        if export_results:
            dfm_gdpyt = modify.stack_dficts_by_key(dfmbicts_gdpyt, drop_filename=False)
            dfm_spc = modify.stack_dficts_by_key(dfmbicts_spc, drop_filename=False)

            dfms = pd.concat([dfm_gdpyt, dfm_spc])

            io.export_df_to_excel(dfms, path_name=join(save_path_name, save_id + '_mean_pdo_measurement_results'),
                                  include_index=False, filetype='.xlsx', drop_columns=results_drop_columns)

        # --- --- EVALUATE RMSE Z FOR PARTICLE DIAMETER OVERLAP
        plot_percent_diameter_overlap = True

        for nname, dfo in dficts_spc.items():
            # limit percent diameter overlap to -25% (not overlapping here)
            dfo['percent_dx_diameter'] = dfo['percent_dx_diameter'].where(dfo['percent_dx_diameter'] > -0.25, -0.25)

            # binning
            columns_to_bin = ['z_true', 'percent_dx_diameter']
            bin_z = [-27.5, -15, -2.5, 10, 22.5]
            bin_pdo = 5

            dfbicts = analyze.evaluate_2d_bin_local_rmse_z(df=dfo,
                                                           columns_to_bin=columns_to_bin,
                                                           bins=[bin_z, bin_pdo],
                                                           round_to_decimals=[3, 4],
                                                           min_cm=0.5,
                                                           equal_bins=[False, True])

            # --- --- PLOT RMSE Z

            if plot_percent_diameter_overlap:
                # Plot rmse z + number of particles binned as a function of percent diameter overlap for different z bins
                fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.35, size_y_inches * 1.5))
                for nameo, dfpdo in dfbicts.items():
                    ax.plot(dfpdo.bin, dfpdo.rmse_z, '-o', label=nameo)
                    ax2.plot(dfpdo.bin, dfpdo.num_bind, '-o')

                ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
                ax.set_yscale('log')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
                ax2.set_xlabel(r'$\gamma \: $(\%)')
                ax2.set_ylabel(r'$N_{p}$')
                plt.tight_layout()
                plt.savefig('/Users/mackenzie/Desktop/percent-overlap/{}_rmsez_num-binned_pdo.png'.format(nname))
                plt.show()

            # --- --- EXPORT RMSE Z TO EXCEL

            dfstack = modify.stack_dficts_by_key(dfbicts, drop_filename=False)
            dfstack.to_excel('/Users/mackenzie/Desktop/percent-overlap/{}_binned_rmsez_by_z_pdo.xlsx'.format(nname),
                             index=False)

        raise ValueError('ha')

        # plot

        # sort
        dfmbicts_gdpyt = modify.dficts_sort(dfmbicts_gdpyt, descending=True)
        dfmbicts_spc = modify.dficts_sort(dfmbicts_spc, descending=True)

        # relabel dficts
        new_keys = [31.6, 36.5,  44.7,  63.3, 89.5]
        dfmbicts_gdpyt = modify.dficts_rename_key(dfmbicts_gdpyt, new_keys)
        dfmbicts_spc = modify.dficts_rename_key(dfmbicts_spc, new_keys)

        # plot global uncertainty - gdpyt vs. spc
        fig, ax, ax2 = plotting.plot_dfbicts_list_global(dfbicts_list=[dfmbicts_gdpyt, dfmbicts_spc], parameters='rmse_z',
                                                         xlabel=xlabel_for_keys, h=h, scale=scale_fig_dim)
        ax.set_ylim(ylim_global)
        ax.legend(labels_global, loc=legend_loc)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_static_v_SPC_global_rmse_z.png'))
        if show_plots:
            plt.show()

        parameters = ['rmse_z', 'true_percent_meas']
        fig, ax, ax2 = plotting.plot_dfbicts_list_global(dfbicts_list=[dfmbicts_gdpyt, dfmbicts_spc], parameters=parameters,
                                                         xlabel=xlabel_for_keys, h=h, scale=scale_fig_dim_outside_x_legend,
                                                         ax2_ylim=ylim_percent_true_measured_global)
        ax.set_ylim(ylim_global)
        ax2.set_ylabel(r'$\phi$')
        ax2.set_ylim(ylim_percent_true_measured_global)
        ax.legend(labels_global, loc='upper left', bbox_to_anchor=(1.2, 1), fancybox=True, shadow=False, ncol=1)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_static_v_SPC_global_rmse_z_and_true_percent_meas.png'))
        if show_plots:
            plt.show()

        parameters = ['rmse_z', 'percent_meas']
        fig, ax, ax2 = plotting.plot_dfbicts_list_global(dfbicts_list=[dfmbicts_gdpyt, dfmbicts_spc], parameters=parameters,
                                                         xlabel=xlabel_for_keys, h=h, scale=scale_fig_dim_outside_x_legend,
                                                         ax2_ylim=ylim_percent_measured_global)
        ax.set_ylim(ylim_global)
        ax2.set_ylabel(r'$\phi_{ID}$')
        ax2.set_ylim(ylim_percent_measured_global)
        ax.legend(labels_global, loc='upper left', bbox_to_anchor=(1.2, 1), fancybox=True, shadow=False, ncol=1)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_static_v_SPC_global_rmse_z_and_percent_meas.png'))
        if show_plots:
            plt.show()

        parameters = ['rmse_z', 'cm']
        fig, ax, ax2 = plotting.plot_dfbicts_list_global(dfbicts_list=[dfmbicts_gdpyt, dfmbicts_spc], parameters=parameters,
                                                         xlabel=xlabel_for_keys, h=h, scale=scale_fig_dim_outside_x_legend,
                                                         ax2_ylim=ylim_cm_global)
        ax.set_ylim(ylim_global)
        ax2.set_ylabel(r'$c_{m}$')
        ax2.set_ylim(ylim_cm_global)
        ax.legend(labels_global, loc='upper left', bbox_to_anchor=(1.2, 1), fancybox=True, shadow=False, ncol=1)
        plt.tight_layout()
        plt.savefig(join(path_name, save_id+'_static_v_SPC_global_rmse_z_and_cm.png'))
        if show_plots:
            plt.show()

    # ----------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------


# analyze convergence of rmse_z wrt # of frames
"""
if show_plots or save_plots:
    # convergence assessment
    ylim_global_convergence = None  # static: [0.0095, 0.0105], spc: [0.08, 0.0915]
    xlabel_for_convergence = r'$N (\#\:of\:images)$'


    dficts_cumlative, dficts_ground_truth_cumlative = modify.split_dficts_cumulative_series(dficts, dficts_ground_truth,
                                                                                            series_parameter='frame',
                                                                                            increments=convergence_increments,
                                                                                            key=inspect_convergence_of_key)
    # calculate local rmse_z
    dfbicts_cumlative = analyze.calculate_bin_local_rmse_z(dficts_cumlative, column_to_bin_and_assess, bins, min_cm, z_range,
                                                 round_z_to_decimal, dficts_ground_truth=dficts_ground_truth_cumlative)
    
    # plot global uncertainty - gdpyt
    fig, ax, ax2 = plotting.plot_dfbicts_global(dfbicts_cumlative, parameters='rmse_z', xlabel=xlabel_for_convergence, h=h,
                                                scale=scale_fig_dim)
    if ylim_global_convergence:
        ax.set_ylim(ylim_global_convergence)
    plt.tight_layout()
    plt.savefig(join(path_name, save_id+'_convergence_key{}_global_rmse_z.png'.format(inspect_convergence_of_key)))
    if show_plots:
        plt.show()
    
    # calculate mean measurement results and export to excel
    dfm_cumlative = analyze.calculate_bin_measurement_results(dfbicts_cumlative, norm_rmse_z_by_depth=h, norm_num_by_bins=bins)
    io.export_df_to_excel(dfm_cumlative, path_name=join(save_path_name, save_id+'_measurement_convergence_key{}'.format(inspect_convergence_of_key)),
                          include_index=True, index_label='test_id', filetype='.xlsx', drop_columns=results_drop_columns)
"""


# ---------------------------------------------------------------