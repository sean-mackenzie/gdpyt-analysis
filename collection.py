# gdpyt-analysis.collection.py

# imports
from os.path import join, isdir
from os import listdir
import re
from collections import OrderedDict

from sequence import GdpytTrackingSequence

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
        2. uniformize initial value
            * modify each sequence's parameter value to a specified value (e.g. uniformize zero plane z-coord)
        3. calculate mean maximum value
            * find the average maximum value wrt a reference value for all sequences (e.g. z-coordinate displacement)
    Data visualization:
        1. Plot single particle trajectory for each sequence
        2. Plot average single particle trajectory for each sequence. 
        3. Plot average maximum value (e.g. z-displacement) for all sequences.
"""

class GdpytTrackingCollection(object):

    def __init__(self, path_name, sort_strings, filetype, exclude=[]):

        super(GdpytTrackingCollection, self).__init__()

        if not isdir(path_name):
            raise ValueError("Specified folder {} does not exist".format(path_name))

        # properties of the traking collection
        self._path_name = path_name
        self._sort_strings = sort_strings
        self._filetype = filetype

        # data
        self.dfs = None
        self.collect(exclude=exclude)

    def collect(self, exclude):
        """
        Notes:
            sort_strings: if sort_strings[1] == filetype, then sort_strings[1] should equal empty string, ''.

        :param path_name:
        :param sort_strings (iterable; e.g. tuple): ['z', 'um']
        :return:
        """
        # read files in directory
        files = [f for f in listdir(self._path_name) if f.endswith(self._filetype)]

        files = sorted(files, key=lambda x: float(x.split(self._sort_strings[0])[-1].split(self._sort_strings[1])[0]))
        names = [int(f.split(self._sort_strings[0])[-1].split(self._sort_strings[1])[0]) for f in files]

        # organize dataframes into list for iteration
        dfs = {}
        for name, file in zip(names, files):
            if name not in exclude:
                dfs.update({name: GdpytTrackingSequence(id_=name, file_path=join(self._path_name, file))})

        self.dfs = dfs