# gdpyt-analysis: utils: read
"""
Notes
"""

# imports
from os.path import join
import os

import pandas as pd
import numpy as np

# scripts


def read_dataframes(path_name, sort_strings, filetype='.xlsx'):
    """
    Notes:
        sort_strings: if sort_strings[1] == filetype, then sort_strings[1] should equal empty string, ''.

    :param path_name:
    :param sort_strings (iterable; e.g. tuple): ['z', 'um']
    :return:
    """
    # read files in directory
    files = [f for f in os.listdir(path_name) if f.endswith(filetype)]

    # sort files
    files = sorted(files, key=lambda x: float(x.split(sort_strings[0])[-1].split(sort_strings[1]+filetype)[0]))

    # get names of files
    names = [float(f.split(sort_strings[0])[-1].split(sort_strings[1]+filetype)[0]) for f in files]

    # organize dataframes into list for iteration
    data = {}
    for n, f in zip(names, files):
        df = read_dataframe(join(path_name, f))
        data.update({n: df})

    return data


def read_dataframe(path_name, sort_strings=[]):
    """

    :param path_name:
    :param sort_strings:
    :return:
    """

    df = pd.read_excel(io=path_name, dtype=float)

    return df