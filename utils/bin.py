# gdpyt-analysis: utils: binning
"""
Notes
"""

# imports
import pandas as pd
import numpy as np
from utils.functions import calculate_precision

# scripts

def bin_local_rmse_z(df, column_to_bin='z_true', bins=20, min_cm=0.5, z_range=None, round_to_decimal=5,
                     df_ground_truth=None, dropna=True, error_column=None):
    """
    Creates a new dataframe and calculates the RMSE-z for a specified: number of bins [integer] OR values in bins [list].

    Parameters
    ----------
    df: a dataframe containing at least: z and z_true.
    bins [integer, list]: integer = sort into bins (#) of equi-spaced buckets; list = sort into bins defined by list.
    min_cm: [0, 1]
    drop: None == keep all of the original columns; ['all', 'most'] == drop all the unnecessary columns for a majority
    of plots.

    Returns
    -------
    dfrmse: dataframe with added columns: "rmse_z" and "num_meas".
    """

    # copy so we don't change the original data
    dfc = df

    # rename column if mis-named
    if 'true_z' in dfc.columns:
        dfc = dfc.rename(columns={"true_z": "z_true"})

    # define extents of z-range
    if z_range:
        dfc = dfc[(dfc['z_true'] > z_range[0]) & (dfc['z_true'] < z_range[1])]

    # if c_m is below minimum c_m, change 'z' to NaN:
    dfc['z'] = np.where((dfc['cm'] < min_cm), np.nan, dfc['z'])

    # returns an identical dataframe but adds a column named "bin"
    if isinstance(bins, (int, float)):
        dfc = bin_by_column(dfc, column_to_bin=column_to_bin, number_of_bins=bins, round_to_decimal=round_to_decimal)
    elif isinstance(bins, (list, tuple, np.ndarray)):
        dfc = bin_by_list(dfc, column_to_bin=column_to_bin, bins=bins, round_to_decimal=round_to_decimal)

    # count the percent not-NaNs in 'z' due to this particular binning
    dfp = dfc.groupby('bin').count()
    dfp['num_bind'] = dfp['cm']
    dfp['num_meas'] = dfp['z']
    dfp['percent_meas'] = dfp['z'] / dfp['cm'] * 100

    # drop NaNs in full dataframe, dfc, so they aren't included in the rmse
    if dropna:
        dfc = dfc.dropna()

    # calculate the squared error for each measurement
    if error_column is not None:
        dfc['error'] = dfc[error_column]
    elif 'error' in dfc.columns:
        pass
    else:
        dfc['error'] = dfc['z_true'] - dfc['z']
    dfc['sqerr'] = dfc['error'] ** 2

    # group by z_true and calculate: mean, sum of square error, and number of measurements
    dfmean = dfc.groupby(by='bin').mean()
    dfsum = dfc.groupby(by='bin').sum().sqerr.rename('err_sum')

    # concatenate mean dataframe with: sum of square error and number of measurements
    # add a column for the true number of particles (if known)
    if df_ground_truth is not None:

        # get bin values
        bin_list = dfc.bin.unique()

        # define extents of ground truth z-range to match test collection
        df_ground_truth = df_ground_truth[(df_ground_truth['z'] > z_range[0]) &
                                          (df_ground_truth['z'] < z_range[1])]

        # adjust column_to_bin to match ground truth column names
        if column_to_bin == 'z_true':
            column_to_bin = 'z'
        elif column_to_bin == 'x_true':
            column_to_bin = 'x'
        elif column_to_bin == 'y_true':
            column_to_bin = 'y'

        # bin dataframe uses list of bin values
        df_ground_truth = bin_by_list(df_ground_truth, column_to_bin=column_to_bin, bins=bin_list, round_to_decimal=round_to_decimal)
        df_true_num_particles_per_bin = df_ground_truth.groupby('bin').count()
        df_true_num_particles_per_bin = df_true_num_particles_per_bin.rename(columns={'filename': 'true_num_particles'})

        dfp = pd.concat([dfp, df_true_num_particles_per_bin[['true_num_particles']]], axis=1, join='inner', sort=False)
        dfp['true_percent_meas'] = dfp['num_meas'] / dfp['true_num_particles'] * 100
        dfrmse = pd.concat(
            [dfmean, dfsum, dfp[['num_bind', 'num_meas', 'percent_meas', 'true_num_particles', 'true_percent_meas']]],
            axis=1, join='inner', sort=False)
    else:
        dfrmse = pd.concat([dfmean, dfsum, dfp[['num_bind', 'num_meas', 'percent_meas']]], axis=1, join='inner',
                           sort=False)

    # calculate the root mean squared error: RMSE = sqrt(sum(sq. error)/sum(num measurements))
    dfrmse['rmse_z'] = np.sqrt(dfrmse.err_sum.divide(dfrmse.num_meas))

    # lastly, drop any uneccessary columns
    dfrmse = dfrmse.drop(['error', 'sqerr', 'err_sum'], axis=1)

    return dfrmse


def bin_local(df, column_to_bin='z_true', bins=20, min_cm=0.5, z_range=None, round_to_decimal=0,
              true_num_particles=None, dropna=True):
    """
    Creates a new dataframe for a defined number of bins.

    Parameters
    ----------
    df: a dataframe containing at least: z and z_true.
    column_to_bin: the dataframe column that defines the binning space.
    num_bins: integer

    Returns
    -------
    dfb: dataframe binned into num_bins by z-coor
    """

    # rename column if mis-named
    if 'true_z' in df.columns:
        df = df.rename(columns={"true_z": "z_true"})

    # define extents of z-range
    if z_range is not None:
        df = df[(df['z_true'] > z_range[0]) & (df['z_true'] < z_range[1])]

    # if c_m is below minimum c_m, change 'z' to NaN:
    if 'cm' in df.columns:
        df.loc[:, 'z'] = np.where((df['cm'] < min_cm), np.nan, df['z'])

    # returns an identical dataframe but adds a column named "bin"
    if (isinstance(bins, int) or isinstance(bins, float)):
        df = bin_by_column(df, column_to_bin=column_to_bin, number_of_bins=bins, round_to_decimal=4)

    elif isinstance(bins, (list, tuple, np.ndarray)):
        df = bin_by_list(df, column_to_bin=column_to_bin, bins=bins, round_to_decimal=round_to_decimal)

    if 'cm' in df.columns:
        # get standard deviation of z
        dfstdev = df.groupby('bin').std()
        dfstdev['z_std'] = dfstdev['z']

        # count the number of NaNs in 'z' due to this particular binning
        dfc = df.groupby('bin').count()
        dfc['num_bind'] = dfc['cm']
        dfc['num_meas'] = dfc['z']
        dfc['percent_meas'] = dfc['z'] / dfc['cm'] * 100

        if dropna:
            df = df.dropna()

        # group by z_true and calculate: mean, sum of square error, and number of measurements
        df = df.groupby(by='bin').mean()

        # join the binned dataframe with the percent measured series.
        # add a column for the true number of particles (if known)
        if true_num_particles is not None:
            dfc['true_num_particles'] = true_num_particles
            dfc['true_percent_meas'] = dfc['num_meas'] / dfc['true_num_particles']
            df = df.join([dfc[['num_bind', 'num_meas', 'percent_meas', 'true_num_particles', 'true_percent_meas']],
                          dfstdev['z_std']])
        else:
            df = df.join([dfc[['num_bind', 'num_meas', 'percent_meas']], dfstdev['z_std']])

    else:
        # get standard deviation of z
        dfstdev = df.groupby('bin').std()
        dfstdev['z_std'] = dfstdev['z']

        dfc = df.groupby('bin').count()
        dfc['num_bind'] = dfc['id']

        # group by binned z_true
        df = df.groupby(by='bin').mean()

        # join the binned dataframe with the num bind.
        df = df.join([dfc[['num_bind']], dfstdev['z_std']])

    return df


def bin_generic(df, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby):
    """
    Bin dataframe 'df' into 'bins' # of bins on column 'column_to_bin' after rounding to 'round_to_decimal' places.

    Return: (1) mean, (2) standard deviation, (3) and counts for a single column
    """
    raw_length = len(df)
    df = df.dropna(subset=[column_to_bin])
    dropna_length = len(df)
    if raw_length > dropna_length:
        print("Dropped {} rows with NaN.".format(raw_length - dropna_length))

    # returns an identical dataframe but adds a column named "bin"
    if isinstance(bins, int):
        df = bin_by_column(df, column_to_bin=column_to_bin, number_of_bins=bins, round_to_decimal=round_to_decimal)

    elif isinstance(bins, (list, tuple, np.ndarray)):
        df = bin_by_list(df, column_to_bin=column_to_bin, bins=bins, round_to_decimal=round_to_decimal)

    if return_groupby is False:
        return df

    else:
        # groupby.mean()
        dfm = df.groupby('bin').mean()

        # groupby.std()
        dfstd = df.groupby('bin').std()

        # groupby.count()
        if column_to_count is not None:
            dfc = df.groupby('bin').count()
            count_column = 'count_' + column_to_count
            dfc[count_column] = dfc[column_to_count]

            dfm = dfm.join([dfc[[count_column]]])

        dfm = dfm.reset_index()
        dfstd = dfstd.reset_index()

        return dfm, dfstd


def bin_generic_2d(df, columns_to_bin, column_to_count, bins, round_to_decimals, min_num_bin, return_groupby):
    """
    Bin dataframe 'df' into 'bins' # of bins on column 'column_to_bin' after rounding to 'round_to_decimal' places.

    Return: (1) mean, (2) standard deviation, (3) and counts for a single column
    """
    column_to_bin_top_level = columns_to_bin[0]
    column_to_bin_low_level = columns_to_bin[1]

    columns_to_count_low_level = column_to_count

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
    if isinstance(bins_top_level, int):
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

    if return_groupby is False:
        return df

    else:

        dfms = []
        dfstds = []

        # for each bin (z)
        for bntl in df.bin_tl.unique():

            # get the dataframe for this bin only
            dfbtl = df[df['bin_tl'] == bntl]
            bins_tl_ll = dfbtl.bin_ll.unique()

            # for each bin (x, y, r)
            dfmll, dfstdll = bin_generic(dfbtl,
                                         column_to_bin_low_level,
                                         column_to_count=columns_to_count_low_level,
                                         bins=bins_tl_ll,
                                         round_to_decimal=round_to_decimals_low_level,
                                         return_groupby=return_groupby,
                                         )

            # filter dfstd
            dfmll = dfmll[dfmll['count_' + column_to_count] > min_num_bin]
            dfms.append(dfmll)

            # re-organize dfstd
            dfstdll['bin_tl'] = bntl
            dfstdll['bin_ll'] = dfstdll['bin']
            dfstdll = dfstdll.dropna()
            dfstdll = dfstdll[dfstdll['bin'].isin(dfmll.bin.unique())]
            dfstds.append(dfstdll)

        df_means = pd.concat(dfms, ignore_index=True)
        df_means = df_means.dropna()

        df_stds = pd.concat(dfstds, ignore_index=True)

        df_means = df_means.drop(columns=['bin'])
        df_stds = df_stds.drop(columns=['bin'])

        return df_means, df_stds


def bin_by_column(df, column_to_bin='z_true', number_of_bins=25, round_to_decimal=2):
    """
    Creates a new column "bin" of which maps column_to_bin to equi-spaced bins. Note, that this does not change the
    original dataframe in any way. It only adds a new column to enable grouping.

    # rename column if mis-named
    if 'true_z' in df.columns:
        df = df.rename(columns={"true_z": "z_true"})

    """

    # round the column_to_bin to integer for easier mapping
    temp_column = 'temp_' + column_to_bin
    df[temp_column] = np.round(df[column_to_bin].values, round_to_decimal)

    # copy the column_to_bin to 'mapped' for mapping
    df.loc[:, 'bin'] = df.loc[:, temp_column]

    # get unique values
    unique_vals = df[temp_column].astype(float).unique()

    # drop temp column
    df = df.drop(columns=[temp_column])

    # calculate the equi-width stepsize
    stepsize = (np.max(unique_vals) - np.min(unique_vals)) / number_of_bins

    # re-interpolate the space
    new_vals = np.linspace(np.min(unique_vals) + stepsize / 2, np.max(unique_vals) - stepsize / 2, number_of_bins)

    # round to reasonable decimal place
    new_vals = np.around(new_vals, decimals=round_to_decimal)

    # create the mapping list
    mappping = map_lists_a_to_b(unique_vals, new_vals)

    # create the mapping dictionary
    mapping_dict = {unique_vals[i]: mappping[i] for i in range(len(unique_vals))}

    # insert the mapped values
    df.loc[:, 'bin'] = df.loc[:, 'bin'].map(mapping_dict)

    return df


def bin_by_list(df, column_to_bin, bins, round_to_decimal=0):
    """
    Creates a new column "bin" of which maps column_to_bin to the specified values in bins [type: list, ndarray, tupe].
    Note, that this does not change the original dataframe in any way. It only adds a new column to enable grouping.
    """

    # rename column if mis-named
    if 'true_z' in df.columns:
        df = df.rename(columns={"true_z": "z_true"})

    # round the column_to_bin to integer for easier mapping
    df = df.round({'z_true': 4})

    if column_to_bin in ['x', 'y']:
        df = df.round({'x': round_to_decimal, 'y': round_to_decimal})

    # copy the column_to_bin to 'mapped' for mapping
    df['bin'] = df[column_to_bin].copy()

    # get unique values and round to reasonable decimal place
    unique_vals = df[column_to_bin].unique()

    # create the mapping list
    mappping = map_lists_a_to_b(unique_vals, bins)

    # create the mapping dictionary
    mapping_dict = {unique_vals[i]: mappping[i] for i in range(len(unique_vals))}

    # insert the mapped values
    df['bin'] = df['bin'].map(mapping_dict)

    return df


def map_lists_a_to_b(a, b):
    """
    returns a new list which is a mapping of a onto b.
    """
    mapped_vals = []
    for val in a:
        # get the distance of val from every element in our list to map to
        dist = np.abs(np.ones_like(b) * val - b)

        # append the value of minimum distance to our mapping list
        mapped_vals.append(b[np.argmin(dist)])

    return mapped_vals