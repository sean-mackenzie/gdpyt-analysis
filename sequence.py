# gdpyt-analysis.collection.py

# imports
from os.path import join, isdir
from os import listdir
import re
from collections import OrderedDict

import pandas as pd

"""
gdpyt-analysis.collection.py

A "collection" is a group of "sequence"s that are grouped for data processing, analysis, and data visualization.

Collection functions:
    Processing:
        1. io (read/write)
        2. filter
    Analysis:
        1. calculate mean initial value
            * find the average + stdev parameter value for all sequences (e.g. z-coordinate)
        3. calculate mean maximum value
            * find the average maximum value wrt a reference value for all sequences (e.g. z-coordinate displacement)
    Data visualization:
        1. Plot single particle trajectory.
        2. Plot average single particle trajectory. 
        3. Plot average maximum value (e.g. z-displacement).
"""


class GdpytTrackingSequence(object):

    def __init__(self, id_, file_path):

        super(GdpytTrackingSequence, self).__init__()

        # properties of the traking collection
        self._id = id_
        self._file_path = file_path

        # data
        self.df = self.load()

    def load(self):
        return pd.read_excel(io=self._file_path, dtype=float)