from utils import io, bin

# path to read
fdir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/10.07.21-BPE_Pressure_Deflection/analyses/test_coords'

# read dataframes from directory
data = io.read_dataframes(path_name=fdir,
                          sort_strings=['z', 'um'])

j = 1