import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from images import processing as imp

# formatting
plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ---

# SETUP

base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization'
image_dir = 'images/10X/calibration_2umSteps'
path_results = base_dir + '/analyses/shared-results/illumination-uniformity'

if not os.path.exists(path_results):
    os.makedirs(path_results)

image_path = base_dir + '/' + image_dir
base_string = 'calib_'
file_type = '.tif'
split_strings = [base_string, file_type]
sort_strings = split_strings

image_ids = np.arange(57, 77)  # np.arange(47, 54)
thresh_val = 150  # 155
grid_size = 16
thresh_multiplier = 2.5  # 3.5

# ---

# READ FILES

nums_and_files = imp.get_files(image_path, image_ids, split_strings, sort_strings, file_type)
images = imp.read_images(image_path, nums_and_files, average_stack=True)

# ---

# PROCESSING

dfs = []

for i, img in images.items():
    df, bkg = imp.calculate_intensity_uniformity(img,
                                                 thresh_val,
                                                 grid_size,
                                                 thresh_multiplier,
                                                 show_image=False,
                                                 )

    df['img_id'] = i
    dfs.append(df)

# ---

# EXPORT RESULTS

dfs = pd.concat(dfs)
dfs.to_excel(path_results + '/illumination_uniformity_by_grid_and_frame.xlsx')

# ---

# PLOTTING

dfg = dfs.groupby(by=['i', 'j']).mean()

fig, ax = imp.plot_image_non_uniformity(dfg, grid_size)
plt.tight_layout()
plt.savefig(path_results + '/illumination-uniformity_average-all.svg')
plt.show()
plt.close()

# plot intensity by frame
dfgid = dfs.groupby('img_id').mean()
fig, (axr, ax) = plt.subplots(nrows=2, sharex=True)

axr.plot(dfgid.index, dfgid.sub_std, '-o')
axr.set_ylabel(r'$\overline{\sigma_{bkg}} \: (A.U.)$')

ax.plot(dfgid.index, dfgid.sub_mean, '-o')
ax.set_xlabel('Frame')
ax.set_ylabel(r'$\overline{I_{bkg}} \: (A.U.)$')

plt.tight_layout()
plt.savefig(path_results + '/illumination-uniformity_by_frame.svg')
plt.show()
plt.close()

# ---

print("Analysis completed without errors.")