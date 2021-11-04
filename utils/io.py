# gdpyt-analysis: utils: read
"""
Notes
"""

# imports
import ast
from os.path import join
import os

import pandas as pd
import numpy as np

# scripts
from utils import io, modify


def read_files(read_to, path_name, sort_strings, filetype='.xlsx', subset=None, startswith=None, columns=[],
               dtype=float):
    """
    Notes:
        sort_strings: if sort_strings[1] == filetype, then sort_strings[1] should equal empty string, ''.

    :param path_name:
    :param sort_strings (iterable; e.g. tuple): ['z', 'um']
    :return:
    """
    # read files in directory
    if startswith is not None:
        files = [f for f in os.listdir(path_name) if f.startswith(startswith) and f.endswith(filetype)]
    else:
        files = [f for f in os.listdir(path_name) if f.endswith(filetype)]

    if subset:
        files = files[:subset]

    # sort files and get names
    if sort_strings[1] == '':
        files = sorted(files, key=lambda x: float(x.split(sort_strings[0])[-1].split(sort_strings[1]+filetype)[0]))
        names = [float(f.split(sort_strings[0])[-1].split(sort_strings[1] + filetype)[0]) for f in files]
    else:
        files = sorted(files, key=lambda x: float(x.split(sort_strings[0])[-1].split(sort_strings[1])[0]))
        names = [float(f.split(sort_strings[0])[-1].split(sort_strings[1])[0]) for f in files]

    # organize dataframes into list for iteration
    data = {}
    for n, f in zip(names, files):

        if read_to == 'df':
            df = read_dataframe(join(path_name, f), filetype, dtype=dtype)
            data.update({n: df})

        elif read_to == 'dict':
            df = read_dataframe(join(path_name, f), filetype, columns=['parameter', n], dtype=dtype)
            df = df.set_index(keys='parameter')
            data.update(df.to_dict(orient='dict'))

    return data


def read_ground_truth_files(settings_dict):

    # organize dataframes into list for iteration
    data = {}

    for n, settings in settings_dict.items():

        ground_truth_path_name = settings['test_ground_truth_image_path']
        test_basestring = settings['test_base_string']
        test_subset = settings['test_image_subset']
        test_num_images = settings['test_col_number_of_images']
        test_cropping = settings['test_cropping_params']

        if isinstance(test_subset, str):
            test_subset = ast.literal_eval(test_subset)

        # read .txt ground truth files to dictionary
        num_files = int(test_num_images)
        ground_truth_filetype = '.txt'
        ground_truth_sort_strings = [test_basestring, ground_truth_filetype]
        gt_dficts = io.read_dataframes(ground_truth_path_name, ground_truth_sort_strings, ground_truth_filetype, subset=num_files)
        df_ground_truth = modify.stack_dficts_by_key(gt_dficts)

        # filter according to cropping specs

        if isinstance(test_cropping, str):
            test_cropping = ast.literal_eval(test_cropping)
            df_ground_truth = df_ground_truth[df_ground_truth['x'] > test_cropping['xmin']]
            df_ground_truth = df_ground_truth[df_ground_truth['x'] < test_cropping['xmax']]
            df_ground_truth = df_ground_truth[df_ground_truth['y'] > test_cropping['ymin']]
            df_ground_truth = df_ground_truth[df_ground_truth['y'] < test_cropping['ymax']]

        data.update({n: df_ground_truth})

    return data



def read_dataframe(path_name, filetype, sort_strings=[], columns=[], dtype=float):
    """
    :param path_name:
    :param sort_strings:
    :return:
    """
    if filetype == '.xlsx':
        if len(columns) > 0:
            df = pd.read_excel(io=path_name, names=columns, dtype=dtype, skiprows=1)
        else:
            df = pd.read_excel(io=path_name, dtype=dtype)

    elif filetype == '.txt':
        dataset = 'Dataset_I'

        if dataset == 'Dataset_I':
            gt = np.loadtxt(path_name)
            df = pd.DataFrame(gt)
        else:
            df = pd.read_csv(path_name, header=None, sep=" ")

        df.columns = ['x', 'y', 'z', 'p_d']

    return df


def read_excel(path_name, filetype='.xlsx', sort_strings=[]):

    df = pd.read_excel(io=path_name, dtype=str)

    return df


"""
Deprecated Functions
"""


def read_dataframes(path_name, sort_strings, filetype='.xlsx', subset=None):
    """

    ----- DEPRECATED -----

    Notes:
        sort_strings: if sort_strings[1] == filetype, then sort_strings[1] should equal empty string, ''.

    :param path_name:
    :param sort_strings (iterable; e.g. tuple): ['z', 'um']
    :return:
    """
    # read files in directory
    files = [f for f in os.listdir(path_name) if f.endswith(filetype)]

    if subset:
        files = files[:subset]

    # sort files and get names
    if sort_strings[1] == '':
        files = sorted(files, key=lambda x: float(x.split(sort_strings[0])[-1].split(sort_strings[1]+filetype)[0]))
        names = [float(f.split(sort_strings[0])[-1].split(sort_strings[1] + filetype)[0]) for f in files]
    else:
        files = sorted(files, key=lambda x: float(x.split(sort_strings[0])[-1].split(sort_strings[1])[0]))
        names = [float(f.split(sort_strings[0])[-1].split(sort_strings[1])[0]) for f in files]

    # organize dataframes into list for iteration
    data = {}
    for n, f in zip(names, files):
        df = read_dataframe(join(path_name, f), filetype)
        data.update({n: df})

    return data