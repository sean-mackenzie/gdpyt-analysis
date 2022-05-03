# gdpyt-analysis: analyze
"""
Notes
"""

# imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
import itertools

from utils import functions, fit
from utils import plotting
from utils.bin import *

# other
plt.style.use(['science', 'ieee', 'std-colors'])
# plt.style.use(['science', 'scatter'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)


# ------------------------------------ PRECISION FUNCTIONS (DATAFRAMES) ------------------------------------------------


def evaluate_1d_static_precision(df, column_to_bin, precision_columns, bins, round_to_decimal=4, min_num_samples=3):
    """
    Bin dataframe 'df' into 'bins' # of bins on column 'column_to_bin' after rounding to 'round_to_decimal' places.

    Return: (1) precision of 'precision_columns' for each 'bin' in 'column_to_bin' and (2) mean precision thereof.
    """
    # ensure list for iterating
    if not isinstance(precision_columns, list):
        precision_columns = [precision_columns]

    # drop NaNs in 'column_to_bin' which cause an Empty DataFrame
    raw_length = len(df)
    df = df.dropna(subset=[column_to_bin])
    dropna_length = len(df)
    if raw_length > dropna_length:
        print("Dropped {} rows with NaN.".format(raw_length - dropna_length))

    # returns an identical dataframe but adds a column named "bin"
    if (isinstance(bins, int) or isinstance(bins, float)):
        df = bin_by_column(df, column_to_bin=column_to_bin, number_of_bins=bins, round_to_decimal=round_to_decimal)

    elif isinstance(bins, (list, tuple, np.ndarray)):
        df = bin_by_list(df, column_to_bin=column_to_bin, bins=bins, round_to_decimal=round_to_decimal)

    # bin list
    df = df.sort_values('bin')
    bin_list = df.bin.unique()

    # calculate static lateral precision
    data = []
    for bn in df.bin.unique():
        dfb = df[df['bin'] == bn]

        if len(dfb) < min_num_samples:
            continue

        temp_precision = []
        temp_mean = []
        temp_counts = len(dfb)

        # evaluate the precision for each column
        for pcol in precision_columns:
            temp_precision.append(calculate_precision(dfb[pcol]))
            temp_mean.append(dfb[pcol].mean())

        data.append([bn] + [temp_counts] + temp_precision + temp_mean)

    data = np.array(data)

    mean_columns = [c + '_m' for c in precision_columns]
    columns = [column_to_bin] + ['counts'] + precision_columns + mean_columns

    dfp = pd.DataFrame(data, columns=columns)

    dfpm = dfp[precision_columns].mean()

    return dfp, dfpm


def evaluate_2d_static_precision(df, column_to_bin, precision_columns, bins=25, round_to_decimal=4):
    """
    Bin dataframe 'df' into 'bins' # of bins on column 'column_to_bin' after rounding to 'round_to_decimal' places.

    Return: (1) mean, (2) standard deviation, (3) and counts for a single column
    """

    # drop NaNs in 'column_to_bin' which cause an Empty DataFrame
    raw_length = len(df)
    df = df.dropna(subset=[column_to_bin])
    dropna_length = len(df)
    if raw_length > dropna_length:
        print("Dropped {} rows with NaN.".format(raw_length - dropna_length))

    # bin - top level (usually a spatial parameter: x, y, z, r)
    df = bin_by_column(df, column_to_bin=column_to_bin, number_of_bins=bins, round_to_decimal=round_to_decimal)

    data = []

    # for each bin (x, y, z)
    for bn in df.bin.unique():

        # get the dataframe for this bin only
        dfb = df[df['bin'] == bn]

        # get particles in this bin
        dfb_pids = dfb.id.unique()

        # for each particle in this bin
        for b_pid in dfb_pids:

            # get the dataframe for this particle in this bin only
            dfbpid = dfb[dfb['id'] == b_pid]

            temp_precision = []
            temp_mean = []
            temp_counts = len(dfbpid)

            # evaluate the precision for each column
            for pcol in precision_columns:
                temp_precision.append(calculate_precision(dfbpid[pcol]))
                temp_mean.append(dfbpid[pcol].mean())

            data.append([bn] + [b_pid] + temp_precision + temp_mean + [temp_counts])

    data = np.array(data)

    mean_columns = [c + '_m' for c in precision_columns]
    columns = [column_to_bin] + ['id'] + precision_columns + mean_columns + ['counts']

    # dataframe(bin, id)
    df_bin_id = pd.DataFrame(data, columns=columns)

    # dataframe(bin)
    # 1. weighted average
    series_bin = df_bin_id.groupby(column_to_bin).apply(lambda x: np.average(x, axis=0, weights=x.counts)).to_numpy()
    series_data = []
    for sb in series_bin:
        series_data.append(sb)
    series_data_arr = np.array(series_data)
    df_bin = pd.DataFrame(series_data_arr, columns=columns)

    # 2. count number of particles + frames in each bin
    bin_counts = df_bin_id.groupby(column_to_bin).sum().counts
    df_bin = df_bin.drop(columns=['id', 'counts'])
    df_bin['counts'] = bin_counts.to_numpy()

    return df_bin_id, df_bin


def evaluate_3d_static_precision(df, columns_to_bin, precision_columns, bins, round_to_decimals):
    """
    Bin dataframe 'df' into 'bins' # of bins on column 'column_to_bin' after rounding to 'round_to_decimal' places.

    Return: (1) mean, (2) standard deviation, (3) and counts for a single column
    """
    column_to_bin_top_level = columns_to_bin[0]
    column_to_bin_low_level = columns_to_bin[1]

    bins_top_level = bins[0]
    bins_low_level = bins[1]

    round_to_decimals_top_level = round_to_decimals[0]
    round_to_decimals_low_level = round_to_decimals[1]

    # drop NaNs in 'column_to_bin' which cause an Empty DataFrame
    raw_length = len(df)
    df = df.dropna(subset=[column_to_bin_top_level, column_to_bin_low_level])
    dropna_length = len(df)
    if raw_length > dropna_length:
        print("Dropped {} rows with NaN.".format(raw_length - dropna_length))

    # bin - top level (usually an axial spatial parameter: z)
    if (isinstance(bins_top_level, int) or isinstance(bins_top_level, float)):
        df = bin_by_column(df,
                           column_to_bin=column_to_bin_top_level,
                           number_of_bins=bins_top_level,
                           round_to_decimal=round_to_decimals_top_level)

    elif isinstance(bins_top_level, (list, tuple, np.ndarray)):
        df = bin_by_list(df,
                         column_to_bin=column_to_bin_top_level,
                         bins=bins_top_level,
                         round_to_decimal=round_to_decimals_top_level)

    df = df.rename(columns={'bin': 'bin_tl'})

    # bin - low level (usually a lateral spatial parameter: x, y, r, dx, percent overlap diameter)
    if isinstance(bins_low_level, (int, float)):
        df = bin_by_column(df,
                           column_to_bin=column_to_bin_low_level,
                           number_of_bins=bins_low_level,
                           round_to_decimal=round_to_decimals_low_level)

    elif isinstance(bins_low_level, (list, tuple, np.ndarray)):
        df = bin_by_list(df,
                         column_to_bin=column_to_bin_low_level,
                         bins=bins_low_level,
                         round_to_decimal=round_to_decimals_low_level)

    df = df.rename(columns={'bin': 'bin_ll'})

    data = []

    # for each bin (z)
    for bntl in df.bin_tl.unique():

        # get the dataframe for this bin only
        dfbtl = df[df['bin_tl'] == bntl]

        # for each bin (x, y, r)
        for bnll in dfbtl.bin_ll.unique():
            # get the dataframe for this bin only
            dfbll = dfbtl[dfbtl['bin_ll'] == bnll]

            # get particles in this bin
            dfb_pids = dfbll.id.unique()

            # for each particle in this bin
            for b_pid in dfb_pids:

                # get the dataframe for this particle in this bin only
                dfbpid = dfbll[dfbll['id'] == b_pid]

                temp_precision = []
                temp_mean = []
                temp_counts = len(dfbpid)

                # evaluate the precision for each column
                for pcol in precision_columns:
                    temp_precision.append(calculate_precision(dfbpid[pcol]))
                    temp_mean.append(dfbpid[pcol].mean())

                data.append([bntl, bnll, b_pid] + [temp_counts] + temp_precision + temp_mean)

    data = np.array(data)

    mean_columns = [c + '_m' for c in precision_columns]
    columns = columns_to_bin + ['id', 'counts'] + precision_columns + mean_columns

    # dataframe(bin, id)
    df_bin_id = pd.DataFrame(data, columns=columns)

    # dataframe(bin)
    # 1. weighted average
    series_bin = df_bin_id.groupby(columns_to_bin).apply(lambda x: np.average(x, axis=0, weights=x.counts)).to_numpy()
    series_data = []
    for sb in series_bin:
        series_data.append(sb)
    series_data_arr = np.array(series_data)
    df_bin = pd.DataFrame(series_data_arr, columns=columns)

    # 2. count number of particles + frames in each bin
    bin_counts = df_bin_id.groupby(columns_to_bin).sum().counts
    df_bin = df_bin.drop(columns=['id', 'counts'])
    df_bin['counts'] = bin_counts.to_numpy()

    return df_bin_id, df_bin


# ------------------------------------ DISPLACEMENT MEASUREMENT (DATAFRAMES) -------------------------------------------


def evaluate_displacement_precision(df, group_by, split_by, split_value, precision_columns, true_dz=None,
                                    std_filter=None):
    # split dataframe into initial and final
    dfi = df[df[split_by] < split_value]
    dff = df[df[split_by] > split_value]

    # count number of measurements
    initial_bins = dfi[group_by].unique()
    i_num = len(dfi)
    f_num = len(dff)

    # filter out measurements > 2 * standard deviations from the the mean
    if std_filter is not None:
        dfi = dfi[np.abs(dfi[precision_columns].mean() - dfi[precision_columns]) <
                  np.max([dfi[precision_columns].std() * std_filter, 20])]
        dff = dff[np.abs(dff[precision_columns].mean() - dff[precision_columns]) <
                  np.max([dff[precision_columns].std() * std_filter, 20])]

    # kernel density estimation plot
    """
    fig = plt.figure()
    yscatter = dff.z.to_numpy()
    xscatter = dfi.z.to_numpy()[:len(yscatter)]
    yscatter = yscatter[:len(xscatter)]
    dist = np.max([dff[precision_columns].std() * 1, 7.5])

    fig, ax, ax_histx, ax_histy = plotting.scatter_hist(xscatter, yscatter, fig,
                                               color=None,
                                               colormap='coolwarm',
                                               scatter_size=0.25,
                                               kde=True,
                                               distance_from_mean=dist)

    ax.set_xlim([np.mean(xscatter) - dist, np.mean(xscatter) + dist])
    ax.set_ylim([np.mean(yscatter) - dist, np.mean(yscatter) + dist])
    ax.set_xlabel(r'$z_{initial} \: (\mu m)$')
    ax.set_ylabel(r'$z_{final} \: (\mu m)$')
    ax_histx.set_title(r'$\Delta z_{true}=$' + ' {} '.format(true_dz) + r'$\mu m$')
    sp = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/figs/test_displacement_precision'
    #plt.savefig(sp + '/kde_truedz{}_mean_zf{}.png'.format(true_dz, np.round(np.mean(yscatter), 2)))
    plt.show()
    """

    # count number of measurements after filtering
    i_num_filt = len(dfi)
    f_num_filt = len(dff)

    # evaluate precision: initial
    dfip, dfipm = evaluate_1d_static_precision(dfi,
                                               column_to_bin=group_by,
                                               precision_columns=precision_columns,
                                               bins=initial_bins,
                                               round_to_decimal=0)
    dfip = dfip.set_index(group_by)
    cols = dfip.columns
    for c in cols:
        dfip = dfip.rename(columns={c: c + '_i'})

    # evaluate precision: final
    final_bins = dff[group_by].unique()
    dffp, dffpm = evaluate_1d_static_precision(dff,
                                               column_to_bin=group_by,
                                               precision_columns=precision_columns,
                                               bins=final_bins,
                                               round_to_decimal=0)
    dffp = dffp.set_index(group_by)
    cols = dffp.columns
    for c in cols:
        dffp = dffp.rename(columns={c: c + '_f'})

    # merge dataframes
    dfp = pd.concat([dfip, dffp], axis=1, join='inner', sort=False)

    # calculate displacement precision and magnitude
    dfp['p_dz'] = dfp.z_i + dfp.z_f
    dfp['m_dz'] = dfp.z_m_f - dfp.z_m_i

    mean_displacement_precision = dfp.p_dz.mean()
    mean_displacement_magnitude = dfp.m_dz.mean()
    mean_displacement_magnitude_precision = functions.calculate_precision(dfp.m_dz)
    num_pids_measured = len(dfp.m_dz)
    percent_pids_measured = len(dfp.m_dz) / len(initial_bins)
    percent_pid_frames_measured = (i_num_filt + f_num_filt) / (i_num + f_num)

    # calculate root mean squared error
    if true_dz:
        # calculate error
        dfp['error_dz'] = true_dz - dfp.m_dz

        rmse = np.sqrt(np.sum(dfp.error_dz.to_numpy() ** 2) / len(dfp.error_dz.to_numpy()))

        return mean_displacement_precision, mean_displacement_magnitude, mean_displacement_magnitude_precision, \
               percent_pids_measured, percent_pid_frames_measured, rmse, num_pids_measured, i_num_filt, f_num_filt

    else:
        return mean_displacement_precision, mean_displacement_magnitude, mean_displacement_magnitude_precision, \
               percent_pids_measured, percent_pid_frames_measured, num_pids_measured, i_num_filt, f_num_filt


# ---------------------------------------- RMSE FUNCTIONS (BELOW) ------------------------------------------------------


def calculate_bin_local_rmse_z(dficts, column_to_bin='z_true', bins=20, min_cm=0.5, z_range=None, round_to_decimal=4,
                               dficts_ground_truth=None):
    """
    Calculate the local rmse_z uncertainty for every dataframe in a dictionary and return the binned dataframe.

    :param dficts:
    :param column_to_bin:
    :param bins:
    :param min_cm:
    :param z_range:
    :param round_to_decimal:
    :param true_num_particles:
    :return:
    """

    dfbs = {}
    for name, df in dficts.items():

        if dficts_ground_truth is not None:
            df_ground_truth = dficts_ground_truth[name]
        else:
            df_ground_truth = None

        if isinstance(bins, dict):
            if bins[name]['calib_volume'] == 0.5:
                num_bins = 12  # 12
            elif bins[name]['calib_volume'] == 1.0:
                num_bins = 24  # 24
            else:
                raise ValueError('Unable to read details: calibration volume')
        else:
            num_bins = bins

        # calculate the local rmse_z uncertainty
        dfb = bin_local_rmse_z(df, column_to_bin=column_to_bin, bins=num_bins, min_cm=min_cm, z_range=z_range,
                               round_to_decimal=round_to_decimal, df_ground_truth=df_ground_truth)

        # update dictionary
        dfbs.update({name: dfb})

    return dfbs


def evaluate_2d_bin_local_rmse_z(df, columns_to_bin, bins, round_to_decimals, min_cm=0.5, equal_bins=None,
                                 error_column=None):
    """
    Bin dataframe 'df' into 'bins' # of bins on column 'column_to_bin' after rounding to 'round_to_decimal' places.

    Return: (1) mean, (2) standard deviation, (3) and counts for a single column
    """
    column_to_bin_top_level = columns_to_bin[0]
    column_to_bin_low_level = columns_to_bin[1]

    if equal_bins is not None:
        if equal_bins[0] is True:
            out, eqbins = pd.qcut(df[column_to_bin_top_level], bins[0], retbins=True, duplicates='drop')
            eqbins = pd.Series(eqbins).rolling(2).mean()
            bins_top_level = eqbins.dropna().to_numpy()
        else:
            bins_top_level = bins[0]

        if equal_bins[1] is True:
            out, eqbins = pd.qcut(df[column_to_bin_low_level], bins[1], retbins=True, duplicates='drop')
            eqbins = pd.Series(eqbins).rolling(2).mean()
            bins_low_level = eqbins.dropna().to_numpy()
        else:
            bins_low_level = bins[1]
    else:
        bins_top_level = bins[0]
        bins_low_level = bins[1]

    round_to_decimals_top_level = round_to_decimals[0]
    round_to_decimals_low_level = round_to_decimals[1]

    # drop NaNs in 'column_to_bin' which cause an Empty DataFrame
    raw_length = len(df)
    df = df.dropna(subset=[column_to_bin_top_level, column_to_bin_low_level])
    dropna_length = len(df)
    if raw_length > dropna_length:
        print("Dropped {} rows with NaN.".format(raw_length - dropna_length))

    # bin - top level (usually an axial spatial parameter: z)
    if (isinstance(bins_top_level, int) or isinstance(bins_top_level, float)):
        df = bin_by_column(df,
                           column_to_bin=column_to_bin_top_level,
                           number_of_bins=bins_top_level,
                           round_to_decimal=round_to_decimals_top_level)

    elif isinstance(bins_top_level, (list, tuple, np.ndarray)):
        df = bin_by_list(df,
                         column_to_bin=column_to_bin_top_level,
                         bins=bins_top_level,
                         round_to_decimal=round_to_decimals_top_level)

    df = df.rename(columns={'bin': 'bin_tl'})
    df = df.sort_values('bin_tl')

    data = {}

    # for each bin (z)
    for bntl in df.bin_tl.unique():
        # bin - low level
        # get the dataframe for this bin only (usually a spatial parameter: x, y, r, dx, percent overlap diameter)
        dfbtl = df[df['bin_tl'] == bntl]

        # evaluate the local rmse_z uncertainty
        dfb = bin_local_rmse_z(dfbtl,
                               column_to_bin=column_to_bin_low_level,
                               bins=bins_low_level,
                               min_cm=min_cm,
                               round_to_decimal=round_to_decimals_low_level,
                               df_ground_truth=None,
                               error_column=error_column)

        dfb = dfb.reset_index()

        data.update({bntl: dfb})

    return data


# ---------------------------------------- BIN FUNCTIONS (BELOW) -------------------------------------------------------


def calculate_bin_local(dficts, column_to_bin='z_true', bins=20, min_cm=0.5, z_range=None, round_to_decimal=0,
                        true_num_particles=None, z0=0, take_abs=False):
    """
    Calculate the local 'column_to_bin' for every dataframe in a dictionary and return the binned dataframe.

    :param dficts:
    :param column_to_bin:
    :param bins:
    :param min_cm:
    :param z_range:
    :param round_to_decimal:
    :param true_num_particles:
    :return:
    """

    dfbs = {}
    for name, df in dficts.items():
        # calculate the local rmse_z uncertainty
        dfb = bin_local(df, column_to_bin=column_to_bin, bins=bins, min_cm=min_cm, z_range=z_range, z0=z0,
                        take_abs=take_abs, round_to_decimal=round_to_decimal, true_num_particles=true_num_particles)

        # update dictionary
        dfbs.update({name: dfb})

    return dfbs


def calculate_dict_particle_spacing(dficts, theoretical_diameter_params_path, mag_eff,
                                    max_n_neighbors=10, dficts_ground_truth=None, maximum_allowable_diameter=None):
    dfos = {}
    for (name, df), (tname, tdf) in zip(dficts.items(), dficts_ground_truth.items()):
        # calculate percent diameter overlap
        dfo = calculate_particle_to_particle_spacing(df, theoretical_diameter_params_path, mag_eff,
                                                     max_n_neighbors=10, true_coords_path=tdf,
                                                     maximum_allowable_diameter=maximum_allowable_diameter)
        # update dictionary
        dfos.update({name: dfo})

    return dfos


# ---------------------------------------- BIN FUNCTIONS (ABOVE) -------------------------------------------------------

# -------------------------------------- OVERLAP FUNCTIONS (BELOW) -----------------------------------------------------


def calculate_particle_to_particle_spacing(test_coords_path,
                                           theoretical_diameter_params_path,
                                           mag_eff,
                                           z_param='z',
                                           zf_at_zero=False,
                                           zf_param='zf_from_nsv',
                                           max_n_neighbors=10,
                                           true_coords_path=None,
                                           maximum_allowable_diameter=None,
                                           popt_contour=None,
                                           param_percent_diameter_overlap='gauss_diameter'):
    """
    1. Reads test_coords.xlsx and computes theoretical Gaussian diameter given parameters.
    2. Computes the percent diameter overlap of all particles in all frame (frame by frame).
    """

    # read true coords (if available)
    if true_coords_path is not None:
        if isinstance(test_coords_path, str):
            ground_truth = np.loadtxt(true_coords_path)
            ground_truth_xy = ground_truth[:, 0:2]
        elif isinstance(true_coords_path, pd.DataFrame):
            filename = true_coords_path.filename.unique()[0]
            ground_truth_xy = true_coords_path[true_coords_path['filename'] == filename][['x', 'y']].values

    # read test coords
    if isinstance(test_coords_path, str):
        df = pd.read_excel(test_coords_path)
    elif isinstance(test_coords_path, pd.DataFrame):
        df = test_coords_path
    df = df.sort_values(by=['frame', 'id'])

    # --- GAUSSIAN DIAMETER
    diameter_params = pd.read_excel(theoretical_diameter_params_path, index_col=0)
    if zf_at_zero is True:
        zf = 0
    else:
        zf = diameter_params.loc[[zf_param]]['mean'].values[0]

    if 'pop_c1' in diameter_params.columns:
        c1 = diameter_params.loc[['pop_c1']]['mean'].values[0]
        c2 = diameter_params.loc[['pop_c2']]['mean'].values[0]
        mag_eff = diameter_params.loc[['mag_eff']]['mean'].values[0]
    else:
        c1 = diameter_params.loc[['c1']]['mean'].values[0]
        c2 = diameter_params.loc[['c2']]['mean'].values[0]

    def theoretical_diameter_function(z):
        return mag_eff * np.sqrt(c1 ** 2 * (z - zf) ** 2 + c2 ** 2)

    df['gauss_diameter'] = theoretical_diameter_function(df[z_param])

    if maximum_allowable_diameter is not None:
        df['gauss_diameter'] = df['gauss_diameter'].where(df['gauss_diameter'] < maximum_allowable_diameter,
                                                          maximum_allowable_diameter)

    # --- CONTOUR DIAMETER
    if popt_contour is not None:
        df['contour_diameter'] = functions.general_gaussian_diameter(df[z_param], *popt_contour)

    data = []
    for fr in df.frame.unique():

        # get all the particles in this frame
        dfr = df[df['frame'] == fr]

        # get ids and locations of all particles in this frame
        pids = dfr.id.values
        locations = np.array([dfr.x.values, dfr.y.values]).T

        if len(locations) < 2:
            continue
        elif len(locations) < max_n_neighbors + 1:
            temp_max_n_neighbors = len(locations)
        else:
            temp_max_n_neighbors = max_n_neighbors + 1

        if true_coords_path is not None:
            nneigh = NearestNeighbors(n_neighbors=temp_max_n_neighbors, algorithm='ball_tree').fit(ground_truth_xy)
            distances, indices = nneigh.kneighbors(locations)
            distance_to_others = distances[:, 1:]
        else:
            nneigh = NearestNeighbors(n_neighbors=temp_max_n_neighbors, algorithm='ball_tree').fit(locations)
            distances, indices = nneigh.kneighbors(locations)
            distance_to_others = distances[:, 1:]

        for distance, pid in zip(distance_to_others, pids):

            # get series of just this particle id
            dfpid = dfr[dfr['id'] == pid]

            # calculate minimum distance and overlap
            diameter = dfpid[param_percent_diameter_overlap].values[0]
            mean_dx_all = np.mean(distance)
            min_dx_all = np.min(distance)
            percent_dx_diameter = (min_dx_all - diameter) / min_dx_all
            num_dx_all = temp_max_n_neighbors - 1

            overlapping_dists = distance[distance < diameter]
            if len(overlapping_dists) == 0:
                mean_dxo = min_dx_all
                num_dxo = 0
            elif len(overlapping_dists) == 1:
                mean_dxo = overlapping_dists[0]
                num_dxo = 1
            else:
                mean_dxo = np.mean(overlapping_dists)
                num_dxo = len(overlapping_dists)

            pid_to_particle_spacing = [fr, pid,
                                       mean_dx_all, min_dx_all, num_dx_all, mean_dxo, num_dxo, percent_dx_diameter]
            data.append(pid_to_particle_spacing)

    # overlap dataframe
    df_dxo = pd.DataFrame(data, columns=['frame', 'id', 'mean_dx', 'min_dx', 'num_dx', 'mean_dxo', 'num_dxo',
                                         'percent_dx_diameter'])

    # merge original dataframe and overlap dataframe
    dfo = pd.merge(left=df, right=df_dxo, on=['frame', 'id'])

    return dfo


def fit_contour_diameter(path_calib_spct_stats):
    df = pd.read_excel(path_calib_spct_stats)
    df = df.dropna()
    df = df.groupby('z_true').mean().reset_index()

    df = df[(df.z_corr > -30) & (df.z_corr < 30)]

    popt_contour, pcov_contour = curve_fit(functions.general_gaussian_diameter, df.z_corr, df.diameter_contour)

    return popt_contour


# ----------------------------------------- ANALYZE DATAFRAMES ---------------------------------------------------------


def df_calc_mean_std_count(df, by, keep_cols, std=True, count=True, return_original_df=False):
    """
    Calculate the dataframe mean, stdev, and counts by groupby keys, 'by'. Optional: stack these values onto the
    original dataframe as new columns.

    :param df:
    :param by:
    :param keep_cols:
    :param mean:
    :param std:
    :param count:
    :return:
    """

    dfm = df.groupby(by=by).mean()
    for col in keep_cols:
        dfm = dfm.rename(columns={col: '{}_mean'.format(col)})

    if std:
        dfstd = df.groupby(by=by).std()
        for col in keep_cols:
            dfstd = dfstd.rename(columns={col: '{}_std'.format(col)})
    else:
        dfstd = None

    if count:
        dfc = df.groupby(by=by).count()
        for col in keep_cols:
            dfc = dfc.rename(columns={col: '{}_counts'.format(col)})
    else:
        dfc = None

    if return_original_df:
        # 1.  dictionary of mapping values

        def map_dataframe(df, method='mean'):
            dff = df.reset_index()
            dff['mapper'] = dff['filename'].astype(str) + '_' + dff['id'].astype(str)
            dff = dff.set_index('mapper')
            dff = dff.drop(columns=['filename', 'id', 'cm_{}'.format(method)])
            mapping_dict = dff.to_dict()
            return mapping_dict

        mapper_dict = {}
        mapper_dict.update(map_dataframe(dfm, method='mean'))
        mapper_dict.update(map_dataframe(dfstd, method='std'))
        mapper_dict.update(map_dataframe(dfc, method='counts'))

        # 2. create columns for mapping in original dataframe
        df['z_mean'] = df['filename'].astype(str) + '_' + df['id'].astype(str)
        df['z_std'] = df['filename'].astype(str) + '_' + df['id'].astype(str)
        df['z_counts'] = df['filename'].astype(str) + '_' + df['id'].astype(str)

        # 3. map values from dict to new columns
        df = df.replace({'z_mean': mapper_dict['z_mean']})
        df = df.replace({'z_std': mapper_dict['z_std']})
        df = df.replace({'z_counts': mapper_dict['z_counts']})

        return df

    else:
        if all([std, count]):
            dfg = dfm.join([dfstd, dfc])
        elif std:
            dfg = dfm.join([dfstd])
        else:
            dfg = dfm

        return dfg


# -------------------------------------- ANALYZE COLLECTIONS (BELOW) ---------------------------------------------------


def calculate_mean_value(dficts, output_var='z', input_var='frame', span=(0, 25)):
    """
    Calculate the mean and stdev across a specified or automatically-determined span for parameter, column.
    """
    names = []
    mean_vals = []
    std_vals = []

    for name, df in dficts.items():
        # get column across span
        dfilt = df[(df[input_var] > span[0]) & (df[input_var] < span[1])]

        names.append(name)

        mean_val = dfilt[output_var].mean()
        std_val = dfilt[output_var].std()

        mean_vals.append(mean_val)
        std_vals.append(std_val)

        print("dataframe {}: average-{}({}, {}) = {} +/- {}".format(name, output_var, span[0], span[1], mean_val,
                                                                    std_val))

    results = np.vstack([names, mean_vals, std_vals]).T

    return results


def calculate_results_by_cm_sweep(dficts, dficts_ground_truth, cm_steps, z_range,
                                  take_mean_of_all=False,
                                  cm_i=0.5, cm_f=0.995, column_to_bin_and_assess='z_true', bins=1, h=80,
                                  round_z_to_decimal=6, split_keys=10):
    # initialize variables
    dfm_cm_sweep_gdpyt = None

    # setup the cm sweep space
    min_cm_sweep = np.round(np.linspace(cm_i, cm_f, cm_steps), 3)

    # calculate measurement results for each cm and concatenate to DataFrame
    for min_cm in min_cm_sweep:

        # analyze by method: bin by number of bins
        dfbicts = calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, bins, min_cm, z_range,
                                             round_z_to_decimal, dficts_ground_truth=dficts_ground_truth)

        if take_mean_of_all:
            # calculate mean measurement results
            dfm_gdpyt = calculate_bin_measurement_results(dfbicts, norm_rmse_z_by_depth=h, norm_num_by_bins=bins)
            dfm_gdpyt['percent_idd'] = dfm_gdpyt['num_bind'] / dfm_gdpyt['true_num_particles'] * 100
            dfm_gdpyt['dft_number'] = dfm_gdpyt['percent_meas'] / dfm_gdpyt['percent_idd']
            dfm_gdpyt['cm_threshold'] = min_cm
        else:
            for name, df in dfbicts.items():
                df = df.drop(
                    columns=['frame', 'stack_id', 'z_true', 'z', 'x', 'y', 'x_true', 'y_true', 'cm', 'max_sim'])
                df.loc[:, 'id'] = name
                df['percent_idd'] = df['num_bind'] / df['true_num_particles'] * 100
                df['dft_number'] = df['percent_meas'] / df['percent_idd']
                df['cm_threshold'] = min_cm
                df = df.set_index('cm_threshold')

                if dfm_cm_sweep_gdpyt is None:
                    dfm_cm_sweep_gdpyt = df.copy()
                else:
                    dfm_cm_sweep_gdpyt = pd.concat([dfm_cm_sweep_gdpyt, df])

        """# ------------------------ SPC -----------------------------------


        if take_mean_of_all:
            # calculate mean measurement results
            dfm_spc = analyze.calculate_bin_measurement_results(dfbicts_spc, norm_rmse_z_by_depth=h, norm_num_by_bins=bins)
            dfm_spc['percent_idd'] = dfm_spc['num_bind'] / dfm_spc['true_num_particles'] * 100
            dfm_spc['dft_number'] = dfm_spc['percent_meas'] / dfm_spc['percent_idd']
            dfm_spc['cm_threshold'] = min_cm
        for name, df in dfbicts_spc.items():
            df['percent_idd'] = df['num_bind'] / df['true_num_particles'] * 100
            df['dft_number'] = df['percent_meas'] / df['percent_idd']
            df['cm_threshold'] = min_cm

            if min_cm == np.min(min_cm_sweep):
                dfm_cm_sweep_spc = df.copy()
            else:
                dfm_cm_sweep_spc = pd.concat([dfm_cm_sweep_spc, df])"""

    return dfm_cm_sweep_gdpyt


# --------------------------------------------------- END --------------------------------------------------------------

# --------------------------------------- ANALYSIS FUNCTIONS (BELOW) ---------------------------------------------------


def evaluate_intrinsic_aberrations(df, z_f, min_cm, param_z_true='z_true', param_z_cm='z_cm', shift_z_by_z_f=True):
    """
    If dataframe (df) is already zero-centered, z_f should equal 0.
    :param df:
    :param z_f:
    :param param_z_true:
    :return:
    """
    ids = df.id.unique()
    num_ids = len(ids)
    num_frames = len(df.frame.unique())

    dataz = []
    datacm = []

    for i in ids:

        dfp = df[(df['id'] == i)]
        frames = dfp.frame.unique()
        dzs = []
        asyms = []

        for f in frames:
            dfpf = dfp[dfp['frame'] == f]
            asym = eval_focal_asymmetry(dfpf, z_f, min_cm=min_cm, param_z_cm=param_z_cm)

            if asym != np.nan:
                if shift_z_by_z_f:
                    dzs.append(dfpf[param_z_true].unique() - z_f)
                else:
                    dzs.append(dfpf[param_z_true].unique())

                asyms.append([asym])

        dataz.append(dzs)
        datacm.append(asyms)

    dataz = list(itertools.chain(*dataz))
    datacm = list(itertools.chain(*datacm))

    dfai = np.hstack([dataz, datacm])
    dfai = pd.DataFrame(dfai, columns=['zs', 'cms'])
    dfai = dfai.dropna()

    dict_intrinsic_aberrations = {'zf': z_f,
                                  'num_pids': num_ids,
                                  'num_frames': num_frames,
                                  'dfai': dfai,
                                  }

    return dict_intrinsic_aberrations


def fit_intrinsic_aberrations(dict_intrinsic_aberrations):
    zs = dict_intrinsic_aberrations['dfai'].zs
    cms = dict_intrinsic_aberrations['dfai'].cms

    zfit = np.linspace(np.min(zs), np.max(zs), 200)

    cpopt, cpcov = curve_fit(functions.cubic, zs, cms)
    qpopt, qpcov = curve_fit(functions.quartic, zs, cms)

    cmfit_cubic = functions.cubic(zfit, *cpopt)
    cmfit_quartic = functions.quartic(zfit, *qpopt)

    dict_intrinsic_aberrations.update({'zfit': zfit,
                                       'cpopt': cpopt,
                                       'cmfit_cubic': cmfit_cubic,
                                       'qpopt': qpopt,
                                       'cmfit_quartic': cmfit_quartic
                                       })

    return dict_intrinsic_aberrations


def evaluate_cm_gradient(df):
    pids = []
    z_trues = []
    dcmdzs = []

    for pid in df.id.unique():

        dfpid = df[df['id'] == pid]

        for fr in dfpid.frame.unique():

            dfr = dfpid[dfpid['frame'] == fr]

            dfr = dfr.reset_index(drop=True)
            z_cm_idx = dfr.cm.idxmax()

            if (z_cm_idx == len(dfr) - 1) or (z_cm_idx == 0):
                continue
            else:
                pids.append(pid)
                z_trues.append(dfr.z_true.unique()[0])
                dcmdzs.append(compute_cm_gradient(dfr, z_cm_idx))

    df_dcmdz = pd.DataFrame(np.vstack([pids, z_trues, dcmdzs]).T, columns=['id', 'z_true', 'dcmdz'])

    return df_dcmdz


def evaluate_self_similarity_gradient(dft, z_f, dcm_threshold=None, path_figs=False, z_range=None):
    # get only particle ID's present in all frames
    dft = dft[dft['layers'] == dft['layers'].max()]

    # center z on focal plane
    dft['z'] = dft['z'] - z_f

    # filter z_range
    if z_range is not None:
        dft = dft[(dft['z'] > -z_range) & (dft['z'] < z_range)]

    # compute gradient for each particle ID
    zs = []
    dcms = []
    pids = []
    for pid in dft.id.unique():
        dfpid = dft[dft['id'] == pid]
        zs.append(dfpid.z.to_numpy())
        dcms.append(dfpid.cm.diff().to_numpy())
        pids.append(dfpid.id.to_numpy())

    # store as dataframe and drop NaNs
    df = pd.DataFrame(np.vstack([np.array(pids).flatten(),
                                 np.array(zs).flatten(),
                                 np.array(dcms).flatten()]).T,
                      columns=['id', 'z', 'dcm'])
    df = df.dropna()

    # filter dcm/dz
    if dcm_threshold is not None:
        df = df[df['dcm'].abs() < dcm_threshold]

    # fit parabola
    popt, pcov = curve_fit(functions.quadratic, df.z, df.dcm.abs())
    z_fit = np.linspace(df.z.min(), df.z.max())
    dcm_fit = functions.quadratic(z_fit, *popt)

    # fit sliding parabola
    popt_slide, pcov = curve_fit(functions.quadratic_slide, df.z, df.dcm.abs())
    dcm_fit_slide = functions.quadratic_slide(z_fit, *popt_slide)

    # groupby z (for concise plotting)
    dfg = df.groupby('z').mean().reset_index()

    # plot
    if path_figs:
        fig, ax = plt.subplots(nrows=3, figsize=(size_x_inches, size_y_inches * 2))

        # raw data
        ax[0].scatter(dft.z, dft.cm, c=dft.id, s=1, alpha=0.25)
        ax[0].set_xlabel('z')
        ax[0].set_ylabel('cm')

        # raw gradient data + fit
        ax[1].scatter(df.z, df.dcm.abs(), c=df.id, s=1, alpha=0.25)
        ax[1].plot(z_fit, dcm_fit, color='black')
        ax[1].plot(z_fit, dcm_fit_slide, color='black', linestyle='--')
        ax[1].set_xlabel('z')
        ax[1].set_ylabel('|dcm/dz|')
        """ax[1].set_title(r'Fit: {}x^2 + {}x + {}'.format(np.format_float_scientific(popt[0], precision=3, exp_digits=2),
                                                        np.format_float_scientific(popt[1], precision=3, exp_digits=2),
                                                        np.round(popt[2], 3),
                                                        )
                        )"""

        # groupby-z gradient + fit
        ax[2].scatter(dfg.z, dfg.dcm.abs(), s=3)
        ax[2].plot(z_fit, dcm_fit, color='black')
        ax[2].plot(z_fit, dcm_fit_slide, color='black', linestyle='--')
        ax[2].set_xlabel('z')
        ax[2].set_ylabel('|dcm/dz|')
        """ax[2].set_title(
            r'Fit: {}(x+{})^2 + {}x + {}'.format(np.format_float_scientific(popt_slide[0], precision=3, exp_digits=2),
                                                 np.round(popt_slide[1], 3),
                                                 np.format_float_scientific(popt_slide[2], precision=3, exp_digits=2),
                                                 np.round(popt_slide[3], 3),
                                                 )
            )"""

        plt.tight_layout()
        plt.savefig(path_figs + '/self_similarity_gradient.png')
        plt.show()

    return df


def calculate_plane_tilt_angle(df, microns_per_pixel, z, x='x', y='y'):
    # fit plane (x, y, z units: microns)
    points_microns = np.stack((df[x] * microns_per_pixel, df[y] * microns_per_pixel, df[z])).T

    px_microns, py_microns, pz_microns, popt_microns = fit.fit_3d(points_microns, fit_function='plane')

    # tilt angle (degrees)
    tilt_x = np.rad2deg(np.arctan((pz_microns[0, 1] - pz_microns[0, 0]) / (px_microns[0, 1] - px_microns[0, 0])))
    tilt_y = np.rad2deg(np.arctan((pz_microns[1, 0] - pz_microns[0, 0]) / (py_microns[1, 0] - py_microns[0, 0])))
    print("x-tilt = {} degrees".format(np.round(tilt_x, 3)))
    print("y-tilt = {} degrees".format(np.round(tilt_y, 3)))

    return tilt_x, tilt_y


# --------------------------------------------------- END --------------------------------------------------------------

# ----------------------------------------- HELPER FUNCTIONS (BELOW) ---------------------------------------------------


def eval_focal_asymmetry(df, zf, min_cm, param_z_cm='z_cm'):
    cml = df[df[param_z_cm] < zf].cm.max()
    cmr = df[df[param_z_cm] > zf].cm.max()
    if np.min([cml, cmr]) < min_cm:
        asym = np.nan
    else:
        asym = cml / cmr - 1
    return asym


def compute_cm_gradient(df, z_cm_idx):
    # compute the left and right derivative
    dcmdz_l = (df.iloc[z_cm_idx].cm - df.iloc[z_cm_idx - 1].cm) / (df.iloc[z_cm_idx].z_cm - df.iloc[z_cm_idx - 1].z_cm)
    dcmdz_r = (df.iloc[z_cm_idx].cm - df.iloc[z_cm_idx + 1].cm) / (df.iloc[z_cm_idx].z_cm - df.iloc[z_cm_idx + 1].z_cm)

    # average derivatives
    mean_dcmdz = np.mean([np.abs(dcmdz_l), np.abs(dcmdz_r)])

    return mean_dcmdz