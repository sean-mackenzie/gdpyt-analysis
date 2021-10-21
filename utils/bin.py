# gdpyt-analysis: utils: binning
"""
Notes
"""

# imports
import pandas as pd
import numpy as np

# scripts

def bin_local_rmse_z(df, column_to_bin='z_true', bins=20, min_cm=0.5, z_range=None, round_to_decimal=0,
                     true_num_particles=None):
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
    dfc = df.copy()

    # rename column if mis-named
    if 'true_z' in dfc.columns:
        dfc = dfc.rename(columns={"true_z": "z_true"})

    # define extents of z-range
    if z_range is not None:
        dfc = dfc[(dfc['z_true'] > z_range[0]) & (dfc['z_true'] < z_range[1])]

    # if c_m is below minimum c_m, change 'z' to NaN:
    dfc['z'] = np.where((dfc['cm'] < min_cm), np.nan, dfc['z'])

    # returns an identical dataframe but adds a column named "bin"
    if (isinstance(bins, int) or isinstance(bins, float)):
        dfc = bin_by_column(dfc, column_to_bin=column_to_bin, number_of_bins=bins)
    elif isinstance(bins, (list, tuple, np.ndarray)):
        dfc = bin_by_list(dfc, column_to_bin=column_to_bin, bins=bins, round_to_decimal=round_to_decimal)

    # count the percent not-NaNs in 'z' due to this particular binning
    dfp = dfc.groupby('bin').count()
    dfp['num_bind'] = dfp['cm']
    dfp['num_meas'] = dfp['cm']
    dfp['percent_meas'] = dfp['z'] / dfp['cm'] * 100

    # drop NaNs in full dataframe, dfc, so they aren't included in the rmse
    dfc = dfc.dropna()

    # calculate the squared error for each measurement
    dfc['error'] = dfc['z_true'] - dfc['z']
    dfc['sqerr'] = dfc['error'] ** 2

    # group by z_true and calculate: mean, sum of square error, and number of measurements
    dfmean = dfc.groupby(by='bin').mean()
    dfsum = dfc.groupby(by='bin').sum().sqerr.rename('err_sum')

    # concatenate mean dataframe with: sum of square error and number of measurements
    # add a column for the true number of particles (if known)
    if true_num_particles is not None:
        dfp['true_num_particles'] = true_num_particles
        dfp['true_percent_meas'] = dfp['num_meas'] / dfp['true_num_particles']
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


def bin_local(df, column_to_bin='z_true', bins=20, min_cm=0.5, z_range=None, z0=0, take_abs=False, round_to_decimal=0,
              true_num_particles=None):
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
    df.loc[:, 'z'] = np.where((df['cm'] < min_cm), np.nan, df['z'])

    # adjust z-coordinates
    df.loc[:, 'z'] -= z0

    # take absolute value about z0
    if take_abs is True:
        df.loc[:, 'z'] = df.loc[:, 'z'].abs()

    # returns an identical dataframe but adds a column named "bin"
    if (isinstance(bins, int) or isinstance(bins, float)):
        df = bin_by_column(df, column_to_bin=column_to_bin, number_of_bins=bins, round_to_decimal=3)
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


def bin_by_column(df, column_to_bin='z_true', number_of_bins=20, round_to_decimal=3):
    """
    Creates a new column "bin" of which maps column_to_bin to equi-spaced bins. Note, that this does not change the
    original dataframe in any way. It only adds a new column to enable grouping.
    """

    # rename column if mis-named
    if 'true_z' in df.columns:
        df = df.rename(columns={"true_z": "z_true"})

    # copy the column_to_bin to 'mapped' for mapping
    df.loc[:, 'bin'] = df.loc[:, column_to_bin]

    # get unique values
    unique_vals = df[column_to_bin].unique()

    # calculate the equi-width stepsize
    stepsize = (np.max(unique_vals) - np.min(unique_vals)) / (number_of_bins - 1)

    # re-interpolate the space
    new_vals = np.linspace(np.min(unique_vals) + stepsize / 2, np.max(unique_vals) - stepsize / 2, number_of_bins - 1)

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