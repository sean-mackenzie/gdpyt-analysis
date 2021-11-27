# test bin, analyze, and plot functions
from collection import GdpytTrackingCollection
from utils import io, plotting, bin, modify, fit
import filter, analyze
from tracking import plotting as trackplot

import random

import matplotlib.pyplot as plt

# read .xlsx files to dictionary
path_name = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/test_coords/rotate'
sort_strings = ['test_id', '_SILPURAN_']
filetype = '.xlsx'

# load collection
test_col = GdpytTrackingCollection(path_name=path_name, sort_strings=sort_strings, filetype=filetype)