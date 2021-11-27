# gdpyt-analysis: analyze
"""
Notes
"""

# imports
import pandas as pd
import numpy as np

from scipy.optimize import curve_fit
from scipy.interpolate import griddata

from utils.bin import *

# scripts

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

        # calculate the local rmse_z uncertainty
        dfb = bin_local_rmse_z(df, column_to_bin=column_to_bin, bins=bins, min_cm=min_cm, z_range=z_range,
                               round_to_decimal=round_to_decimal, df_ground_truth=df_ground_truth)

        # update dictionary
        dfbs.update({name: dfb})

    return dfbs

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


def calculate_bin_measurement_results(dfbicts, norm_rmse_z_by_depth=None, norm_num_by_bins=None):
    """
    Calculate the depth-averaged measurement results by taking the mean of the local binned parameters for every
    dataframe in a dictionary and return a dataframe with ID's as the index and mean values as columns.

    :param dfbicts: dictionary of keys (names) and values (dataframes of binned measurement results).
    :param norm_rmse_z_by_depth: normalize by measurement depth
    :param norm_num_by_bins: normalize particle counts by number of bins.

    :return: dfms: dataframe of mean measurement results for all keys/values in dictionary.
    """

    names = []
    data = []

    for name, df in dfbicts.items():

        names.append(name)
        data.append(df.mean().to_numpy())

    data = np.array(data)
    dfms = pd.DataFrame(np.array(data), index=names, columns=df.columns.values)

    if norm_rmse_z_by_depth:
        dfms['rmse_z'] = dfms['rmse_z'] / norm_rmse_z_by_depth

    if norm_num_by_bins:
        dfms['num_bind'] = dfms['num_bind'] * norm_num_by_bins
        dfms['num_meas'] = dfms['num_meas'] * norm_num_by_bins
        dfms['true_num_particles'] = dfms['true_num_particles'] * norm_num_by_bins

    return dfms


def calculate_mean_value(dficts, output_var='z', input_var='frame', span=(0, 25)):
    """
    Calculate the mean and stdev across a specified or automatically-determined span for parameter, column.
    """
    mean_vals = []
    std_vals = []

    for name, df in dficts.items():

        # get column across span
        dfilt = df[(df[input_var] > span[0]) & (df[input_var] < span[1])]

        mean_val = dfilt[output_var].mean()
        std_val = dfilt[output_var].std()

        mean_vals.append(mean_val)
        std_vals.append(std_val)

        print("dataframe {}: average-{}({}, {}) = {} +/- {}".format(name, output_var, span[0], span[1], mean_val, std_val))

    return zip(mean_vals, std_vals)


def calculate_angle_between_planes(a, b, s=400):

    dz_xx = b[0, 0, 2] - a[0, 0, 2]
    dz_xy = b[0, 1, 2] - a[0, 1, 2]
    dz_yx = b[1, 0, 2] - a[1, 0, 2]
    dz_yy = b[1, 1, 2] - a[1, 1, 2]

    thetax = np.arctan((dz_xy - dz_xx) / s) * 360 / (2 * np.pi)
    thetay = np.arctan((dz_yx - dz_xx) / s) * 360 / (2 * np.pi)

    return thetax, thetay