# gdpyt-analysis: utils: modify
"""
Notes
"""

# imports
import pandas as pd
import numpy as np
from utils import bin

# scripts


def dficts_scale(dficts, columns, multipliers):

    if isinstance(multipliers, (float, int)):
        multipliers = [multipliers]

    if len(multipliers) != len(columns):
        multipliers = np.ones_like(columns, dtype=float) * multipliers

    for name, df in dficts.items():
        for column, multiplier in zip(columns, multipliers):

            if column == 'index':
                df.index = df.index * multiplier
            else:
                df.loc[:, column] *= multiplier

        # update the dictionary
        dficts.update({name: df})

    return dficts


def dficts_shift(dficts, columns, shifts):
    for name, df in dficts.items():
        for column, shift in zip(columns, shifts):

            if column == 'index':
                df.index = df.index + shift
            else:
                df.loc[:, column] += shift

        # update the dictionary
        dficts.update({name: df})

    return dficts


def dficts_flip(dficts, column):
    for name, df in dficts.items():

        if column == 'index':
            df.index = df.index.max() - df.index
        else:
            df.loc[:, column] = df.loc[:, column].max() - df.loc[:, column]
            """df.loc[:, column] *= -1
            df.loc[:, column] += df.loc[:, column].max()"""

        # update the dictionary
        dficts.update({name: df})

    return dficts

def stack_dficts_by_key(dficts, drop_filename=False):

    dfs = []

    for name, df in dficts.items():

        if not drop_filename:
            df['filename'] = name

        dfs.append(df)

    dfstack = pd.concat(dfs)

    return dfstack


def stack_dficts(dficts, keys):
    """
    Merge a list of dictionaries into a single dictionary with keys 'keys'.

    :param dficts:
    :param keys:
    :return:
    """

    stacked_dficts = {}

    for dfict, key in zip(dficts, keys):

        dfs = []
        for name, df in dfict.items():
            dfs.append(df)

        dfstack = pd.concat(dfs)

        stacked_dficts.update({key: dfstack})

    return stacked_dficts


def split_df_and_merge_dficts(df, keys, column_to_split, splits, round_to_decimal=0):
    """
    Split a dataframe by column 'column_to_split' along values 'splits' and merge into a single dictionary.

    :param df:
    :param keys:
    :param column_to_split:
    :param splits:
    :param round_to_decimal:
    :return:
    """

    if isinstance(splits, (list, tuple, np.ndarray)):
        dfc = bin.bin_by_list(df, column_to_bin=column_to_split, bins=splits, round_to_decimal=round_to_decimal)
    else:
        raise ValueError("'splits' must be a list of values to split dataframe along column 'column_to_bin'.")

    ks = []
    vs = []
    for key, split in zip(keys, splits):
        df_subset = dfc[dfc['bin'] == split].copy()
        df_subset = df_subset.drop(columns='bin')

        ks.append(key)
        vs.append(df_subset)

    sorted_by_keys = sorted(list(zip(ks, vs)), key=lambda x: x[0])

    ks = [x[0] for x in sorted_by_keys]
    vs = [x[1] for x in sorted_by_keys]

    dficts = dict(zip(ks, vs))

    return dficts


def split_dficts_cumulative_series(dficts, dficts_ground_truth, series_parameter='frame', increments=10, key=1):
    """
    Split dficts key='key' into subsets along 'series_parameter' defined by 'increments'.

    :param dficts: dictionary of names and dataframes with test coordinates.
    :param series_parameter: 'frame'
    :param increments: step size to assess convergence.
    :return:
    """
    # assign correct variable names
    if series_parameter == 'frame':
        series_parameter_gt = 'filename'

    # get dataframe
    df = dficts[key]
    df_gt = dficts_ground_truth[key]

    # get length of dataframe along 'series_parameter'
    series_length = len(df[series_parameter].unique())

    # define subset bounds according to 'increments'
    num_increments = series_length / increments
    num_increments = np.ceil(num_increments)
    increment_bounds = np.arange(1, num_increments + 1) * increments

    # split dataframe into subsets along 'series_parameter'
    dfcs = {}
    dfcs_gt = {}

    for bound in increment_bounds:
        name = bound

        # split test coords dataframe
        df_subset = df[df[series_parameter] < bound].copy()
        dfcs.update({name: df_subset})

        # split ground truth dataframe
        df_subset_gt = df_gt[df_gt[series_parameter_gt] < bound].copy()
        dfcs_gt.update({name: df_subset_gt})

    return dfcs, dfcs_gt