from utils import io, bin, plotting

path_name = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/10.07.21-BPE_Pressure_Deflection/results/' \
            'test/z500um-20imgs-lowerhalf/' \
            'gdpyt_test_coords_10.07.21-BPE_Pressure_Deflection-test_10.07.21-BPE_Pressure_Deflection-calib_2021-10-12 15:50:21.862826.xlsx'

df = io.read_dataframe(path_name, sort_strings=[])

dfb = bin.bin_local(df, column_to_bin='y', num_bins=20)

fig = plotting.plot_df_scatter(dfb)

fig.show()

j = 1