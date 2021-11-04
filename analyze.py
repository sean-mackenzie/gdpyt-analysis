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

def calculate_bin_local_rmse_z(dficts, column_to_bin='z_true', bins=20, min_cm=0.5, z_range=None, round_to_decimal=0,
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

        df_ground_truth = dficts_ground_truth[name]

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
    for item in dficts.items():

        # get name and dataframe (for readability)
        name = item[0]
        df = item[1]

        # calculate the local rmse_z uncertainty
        dfb = bin_local(df, column_to_bin=column_to_bin, bins=bins, min_cm=min_cm, z_range=z_range, z0=z0,
                        take_abs=take_abs, round_to_decimal=round_to_decimal, true_num_particles=true_num_particles)

        # update dictionary
        dfbs.update({name: dfb})

    return dfbs