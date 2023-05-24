# gdpyt-analysis: utils: modify
"""
Notes
"""

# imports
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils import bin, io

# scripts

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------


def map_column_between_dataframes(map_to_df, map_to_col, map_on_col, mapping_df, mapping_col):

    # 1. create column to map onto
    map_to_df[map_to_col] = map_to_df[map_on_col]

    # 2. create mapping dictionary
    mapper_dict = mapping_df[[map_on_col, mapping_col]].set_index(map_on_col).to_dict()

    # 3. map
    map_to_df = map_to_df.replace({map_to_col: mapper_dict[mapping_col]})

    return map_to_df


# ----------------------------------------------------------------------------------------------------------------------

# -

# -------------------------------------- EVERYTHING BELOW IS PRETTY OLD ------------------------------------------------

# -

# -

# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------   DICTIONARIES (below)  ------------------------------------------------------------


def dficts_rename_key(dficts, new_keys):
    old_keys = list(dficts.keys())

    renamed_dficts = {}
    for ok, nk in zip(old_keys, new_keys):
        renamed_dficts.update({nk: dficts[ok]})

    return renamed_dficts


def dficts_sort(dficts, descending=False):
    keys = list(dficts.keys())

    if descending:
        keys.sort(reverse=True)
    else:
        keys.sort()

    sorted_dficts = {}
    for k in keys:
        sorted_dficts.update({k: dficts[k]})

    return sorted_dficts

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


def dficts_new_column(dficts, new_columns, columns, multipliers):

    if isinstance(multipliers, (float, int)):
        multipliers = [multipliers]

    if len(multipliers) != len(columns):
        if len(multipliers) > len(columns):
            iterable_multipliers = iter(multipliers)
        elif len(multipliers) < len(columns):
            multipliers = np.ones_like(columns, dtype=float) * multipliers
            iterable_multipliers = None
        else:
            iterable_multipliers = None
    else:
        iterable_multipliers = None

    for name, df in dficts.items():

        if iterable_multipliers:
            multipliers = [next(iterable_multipliers)]

        for new_column, column, multiplier in zip(new_columns, columns, multipliers):

            # new column == constant
            if column == '':
                df[new_column] = multiplier

            # new column == two columns multiplied together
            elif isinstance(multiplier, str):
                raise ValueError('something may be wrong here')
                df[new_column] = df.loc[:, column] / np.sqrt(df.loc[:, multiplier])

            # new column == column multiplied by constant
            elif column == 'index':
                df[new_column] = df.index * multiplier
            else:
                df[new_column] = df.loc[:, column] * multiplier

        # update the dictionary
        dficts.update({name: df})

    return dficts


def dficts_set_index(dficts, column_to_index):

    for name, df in dficts.items():

        df = df.set_index(keys=column_to_index)

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
    """
    Merge a dictionary of dataframes into a single dataframe with a new column for dict keys.
    :param dficts:
    :param drop_filename:
    :return:
    """

    dfs = []

    for name, df in dficts.items():

        dfc = df.copy()

        if not drop_filename:
            if 'filename' in dfc.columns:
                dfc.insert(0, 'dict_key', name)
            else:
                dfc.insert(0, 'filename', name)

        dfs.append(dfc)

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


def split_dficts(dficts, dficts_ground_truth=None, key_to_split=10.0, get_less_than=True, new_keys=None):

    if get_less_than:
        # get dictionary of only gdpyt
        ids = [xk for xk, xv in dficts.items() if xk < key_to_split]
        dfs = [xv for xk, xv in dficts.items() if xk < key_to_split]

        if new_keys:
            dficts_split = {new_keys[i]: dfs[i] for i in range(len(ids))}
        else:
            dficts_split = {ids[i]: dfs[i] for i in range(len(ids))}

        if dficts_ground_truth:
            ids = [xk for xk, xv in dficts_ground_truth.items() if xk < key_to_split]
            dfs = [xv for xk, xv in dficts_ground_truth.items() if xk < key_to_split]

            if new_keys:
                dficts_ground_truth_split = {new_keys[i]: dfs[i] for i in range(len(ids))}
            else:
                dficts_ground_truth_split = {ids[i]: dfs[i] for i in range(len(ids))}

    else:
        # get dictionary of only gdpyt
        ids = [xk for xk, xv in dficts.items() if xk > key_to_split]
        dfs = [xv for xk, xv in dficts.items() if xk > key_to_split]

        if new_keys:
            dficts_split = {new_keys[i]: dfs[i] for i in range(len(ids))}
        else:
            dficts_split = {ids[i]: dfs[i] for i in range(len(ids))}

        if dficts_ground_truth:
            ids = [xk for xk, xv in dficts_ground_truth.items() if xk > key_to_split]
            dfs = [xv for xk, xv in dficts_ground_truth.items() if xk > key_to_split]

            if new_keys:
                dficts_ground_truth_split = {new_keys[i]: dfs[i] for i in range(len(ids))}
            else:
                dficts_ground_truth_split = {ids[i]: dfs[i] for i in range(len(ids))}

    if dficts_ground_truth:
        return dficts_split, dficts_ground_truth_split
    else:
        return dficts_split


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


# ---------------------------------   DICTIONARIES (above)  ------------------------------------------------------------

# ---------------------------------   DATAFRAMES (below)  --------------------------------------------------------------


def split_df_and_merge_dficts(df, keys, column_to_split, splits, round_to_decimal=0, min_length=0):
    """
    Split a dataframe by column 'column_to_split' along values 'splits' and merge into a single dictionary.

    :param df:
    :param keys:
    :param column_to_split:
    :param splits:
    :param round_to_decimal:
    :param min_length:
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

        if len(df_subset) > min_length:
            ks.append(key)
            vs.append(df_subset)
        else:
            print("Dropped {} because {} rows is less than {} minimum length filter.".format(key,
                                                                                             len(df_subset),
                                                                                             min_length))

    sorted_by_keys = sorted(list(zip(ks, vs)), key=lambda x: x[0])

    ks = [x[0] for x in sorted_by_keys]
    vs = [x[1] for x in sorted_by_keys]

    dficts = dict(zip(ks, vs))

    return dficts


def groupby_stats(df, group_by='id', drop_columns=None):
    """
    Convenience method for calculating dataframe 'group by' quantities.

    :param df:
    :param group_by:
    :param drop_columns:
    :return:
    """
    if 'frame' in df.columns:
        frames = len(df.frame.unique())
    else:
        frames = None

    if drop_columns is not None:
        df = df.copy()
        drop_cols = [d for d in drop_columns if d in df.columns]
        df = df.drop(columns=drop_cols)

    # mean
    df_mean = df.groupby(group_by).mean()

    # standard deviation
    df_std = df.groupby(group_by).std()
    cols_std = {c: c + '_std' for c in df_std.columns}
    df_std = df_std.rename(columns=cols_std)
    # df_std = df_std.multiply(2) - changed on 4/13/22

    # counts
    df_counts = df.groupby(group_by).count()
    df_counts = df_counts.rename(columns={'z': 'z_counts'})
    if frames is None:
        frames = df_counts.z_counts.max()
    df_counts['z_counts_percent'] = df_counts.z_counts / frames * 100
    drop_cols = [d for d in df_counts.columns if d not in ['z_counts', 'z_counts_percent']]
    df_counts = df_counts.drop(columns=drop_cols)

    dfg_stats = pd.concat([df_mean, df_std, df_counts], axis=1, join='inner', sort=False)

    return dfg_stats


def map_values_on_frame_id(dfs, dft):
    """
    Purpose:
        * Problem:
            * columns 'z_est' and 'z_true' in particle similarity curves are not the same as test coords after processing.
            * columns 'frame' and 'id' are identical to test coords.
        * Method:
            1. rename 'z_est' and 'z_true' to 'z_est_raw' and 'z_true_raw'
            2. map 'z_true' and 'z' from test coords to p.s.c. using 'frame' and 'id' columns
    """

    # prepare dfs for mapping
    dfs['z_true_raw'] = dfs['z_true']
    dfs['z_true_map'] = dfs['frame'].astype(str) + '_' + dfs['id'].astype(str)

    dfs['z_est_raw'] = dfs['z_est']
    dfs['z_est_map'] = dfs['frame'].astype(str) + '_' + dfs['id'].astype(str)

    dfs = dfs.astype({'z_true_map': 'str', 'z_est_map': 'str'})

    # prepare dft for mapping
    dft['mapper'] = dft['frame'].astype(str) + '_' + dft['id'].astype(str)
    dft = dft.astype({'mapper': 'str'})

    # drop rows in dfs that aren't in dft
    dfs = dfs[dfs.z_true_map.isin(dft.mapper.values)]

    # create the mapping dictionary
    map_on = dft['mapper'].astype(str).values
    map_z_true_to = dft['z_true'].astype(str).values
    mapping_dict_z_true = {map_on[i]: map_z_true_to[i] for i in range(len(map_on))}

    map_z_to = dft['z'].astype(str).values
    mapping_dict_z = {map_on[i]: map_z_to[i] for i in range(len(map_on))}

    # insert the mapped values
    dfs.loc[:, 'z_true_map'] = dfs.loc[:, 'z_true_map'].map(mapping_dict_z_true)
    dfs.loc[:, 'z_est_map'] = dfs.loc[:, 'z_est_map'].map(mapping_dict_z)

    # change datatype to float
    dfs = dfs.astype({'z_true_map': 'float64', 'z_est_map': 'float64'})
    dfs = dfs.round({'z_true_map': 4, 'z_est_map': 4})
    dfs = dfs.drop(columns=['z_true', 'z_est'])
    dfs = dfs.rename(columns={'z_true_map': 'z_true', 'z_est_map': 'z_est'})

    return dfs


def map_adjacent_z_true(df, df_ground_truth, dfsettings, threshold, single_column_x=None, single_column_dx=None):
    """
    Map 'z_adj' and 'dz' values to test_coords using NearestNeighbors.
        * 'z_adj' == the z_true value of the adjacent (overlapping) particle.
        * 'dz' == the difference in z (depth) between the two adjacent particles.

    :param df: the raw test_coords output from IDPT analysis.
    :param df_ground_truth:
    :param dfsettings: used to scale dx by microns-per-pixels to calculate theta.
    :param threshold: for standard dz dataset: threshold = 30
    :param single_column_x: the specified "dx" for the single, isolated particle
    :return:
    """

    # get microns-per-pixels
    microns_per_pixels = float(dfsettings['microns_per_pixel'])
    if microns_per_pixels < 0.1:
        microns_per_pixels = microns_per_pixels * 1e6

    # drop NaNs
    df = df.dropna()

    # initialize list of dataframes
    dfs = []

    # get this frame only
    for fr in df.frame.unique():

        # get ground truth
        ground_truth_baseline = df_ground_truth[df_ground_truth['filename'] == fr]
        ground_truth_baseline = ground_truth_baseline.sort_values(['y', 'x'])
        ground_truth_xy = ground_truth_baseline[['x', 'y']].values

        # get this frame only
        dff = df[df['frame'] == fr]

        # test: get ID's and coords
        dff = dff.sort_values(['y', 'x'])
        coords_ids = dff['id'].values
        coords_xy = dff[['x', 'y']].values

        if len(ground_truth_xy) < 100:
            print("Frame: {}, ground truth length: {}".format(fr, len(ground_truth_xy)))

        # perform NearestNeighbors
        nneigh = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(ground_truth_xy)
        distances, indices = nneigh.kneighbors(np.array(coords_xy))

        # get only 2nd column
        distances = distances[:, 1]
        indices = indices[:, 1]

        # --- mapping: 2 parts

        # Part 1: map 'z_adj' to all segmented particles (i.e. dx > 7.5)

        # create columns to fill
        dff['z_adj'] = 0
        dff['x_adj'] = 0

        # create the mapping of adjacent particle ID's to particle ID's
        cids_not_mapped = []

        for distance, idx, cid in zip(distances, indices, coords_ids):
            if distance < threshold:
                dff.loc[dff['id'] == cid, 'z_adj'] = ground_truth_baseline.iloc[idx].z
                dff.loc[dff['id'] == cid, 'x_adj'] = ground_truth_baseline.iloc[idx].x
            else:
                cids_not_mapped.append([cid, distance, idx])

        # Part 2: if single column is present
        if single_column_x is not None:
            df_single = dff[dff['x'] < single_column_x][['z_true', 'x']]
            dff.loc[dff['x'] < single_column_x, 'z_adj'] = df_single['z_true']
            dff.loc[dff['x'] < single_column_x, 'x_adj'] = df_single['x'] - single_column_dx

        # append to list
        dfs.append(dff)

    # concat list of dataframes
    df_z_adj = pd.concat(dfs)

    # add 'dz' column
    df_z_adj['dz'] = df_z_adj.z_true - df_z_adj.z_adj

    # add 'dx' column
    df_z_adj['dx'] = df_z_adj.x - df_z_adj.x_adj

    # add 'theta' column
    df_z_adj['theta'] = np.rad2deg(np.arctan(df_z_adj['dz'] / (df_z_adj['dx'] * microns_per_pixels)))
    df_z_adj['theta'] = df_z_adj['theta'].fillna(0)

    df_z_adj = df_z_adj.sort_values('frame')

    # discard any rows that weren't matched appropriately
    # NOTES: this may be a stickier problem (for the future) but for right now, a distance filter seems to work.
    i_len = len(df_z_adj)
    df_z_adj = df_z_adj[df_z_adj['dx'].abs() < np.max([single_column_dx, threshold]) + 1]
    if len(df_z_adj) < i_len:
        print("{} rows removed by this odd filter. Need to investigate this more.".format(i_len - len(df_z_adj)))

    # print length of particles that were not matched
    print("{} rows not mapped (i.e., cids_not_mapped)".format(len(cids_not_mapped)))

    return df_z_adj


# ---------------------------------   DATAFRAMES (above)  --------------------------------------------------------------

# ----------------------------------------------- HELPER FUNCTIONS -----------------------------------------------------


def merge_calib_pid_defocus_and_correction_coords(path_calib_coords, method, dfs=None):

    if dfs is not None:
        dfxy = dfs[0]
        dfcpid = dfs[1]

        if len(dfxy) > len(dfxy.id.unique()):
            dfxy = dfxy.groupby('id').mean()
    else:
        dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method=method)

        # calib coords with (x, y)
        if dfcstats is not None:
            dfxy = dfcstats.groupby('id').mean()
        elif dfc is not None:
            dfxy = dfc.groupby('id').mean()
        else:
            raise ValueError('Either dfcstats or dfc (calibration correction coords) must be readable.')

    # add an 'r' column
    if 'r' not in dfxy.columns:
        dfxy['r'] = np.sqrt((dfxy.x - 256) ** 2 + (dfxy.y - 256) ** 2)
        print("!!!! WARNING: 'r' column assumes that image center is at (256, 256) !!!!")

    # per-particle coords
    dfcpid = dfcpid.set_index('id')

    # merge
    dfcpidxy = pd.concat([dfxy[['x', 'y', 'r']], dfcpid], axis=1, join='inner').reset_index()
    dfcpidxy.to_excel(path_calib_coords + '/calib_{}_pid_defocus_stats_xy.xlsx'.format(method), index=False)

    return dfcpidxy