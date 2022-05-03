from os import listdir
from os.path import join

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import analyze
from utils import io
from utils.plot_collections import plot_spct_stats, plot_rigid_displacement_test


# ------------------------------------------------- PLOT SPCT STATS ----------------------------------------------------

# spct stats plots
spct_stats = True

if spct_stats:

    # filpaths
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/' \
               'analyses/results-04.16.22_10X-spct-idpt-meta-assessment/spct'

    # read
    plot_spct_stats(base_dir)

raise ValueError('h')
# ------------------------------------------------- PLOT PRECISION -----------------------------------------------------


# idpt - displacement
# analyze precision of all tests
analyze_precision = False

if analyze_precision:
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/test_coords/test/step/test_id2_coords_30micron_step_towards.xlsx'

    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/test_coords/test/step_signed'
    filetype = '.xlsx'

    sort_ids = ['test_id', '_coords_']
    sort_dzs = ['_coords_', 'micron_step_']

    files = [f for f in listdir(base_dir) if f.endswith(filetype)]
    files = sorted(files, key=lambda x: float(x.split(sort_ids[0])[-1].split(sort_ids[1])[0]))
    names = [float(f.split(sort_ids[0])[-1].split(sort_ids[1])[0]) for f in files]
    dzs = [float(f.split(sort_dzs[0])[-1].split(sort_dzs[1])[0]) for f in files]

    data_spct = []
    data_idpt = []
    for fp, name, dz in zip(files, names, dzs):

        df = pd.read_excel(join(base_dir, fp))
        mdp, mdm, mdmp, rmse = analyze.evaluate_displacement_precision(df,
                                                                       group_by='id',
                                                                       split_by='frame',
                                                                       split_value=50.5,
                                                                       precision_columns='z',
                                                                       true_dz=dz)
        if name < 10:
            data_idpt.append([dz, mdm, mdp, mdmp, rmse])
        else:
            data_spct.append([dz, mdm, mdp, mdmp, rmse])

    dfp_idpt = pd.DataFrame(np.array(data_idpt), columns=['true_dz', 'dz', 'pz', 'pdz', 'rmse'])
    dfp_spct = pd.DataFrame(np.array(data_spct), columns=['true_dz', 'dz', 'pz', 'pdz', 'rmse'])

    # plot rms error
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, sharex=True)

    ax1.scatter(dfp_spct.index + 1, dfp_spct.pz, label='SPCT')
    ax1.scatter(dfp_idpt.index + 1, dfp_idpt.pz, label='IDPT')
    ax1.set_ylabel(r'$\sigma_{i} + \sigma_{f} \: (\mu m)$')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax2.errorbar(dfp_spct.index + 1, dfp_spct.dz, yerr=dfp_spct.pdz, fmt='o', elinewidth=3, capsize=4)
    ax2.errorbar(dfp_idpt.index + 1, dfp_idpt.dz, yerr=dfp_idpt.pdz, fmt='o', elinewidth=3, capsize=4)
    ax2.set_ylabel(r'$\overline{\Delta z} \pm \sigma_{\Delta z} \: (\mu m)$')

    ax3.scatter(dfp_spct.index + 1, dfp_spct.rmse)
    ax3.scatter(dfp_idpt.index + 1, dfp_idpt.rmse)
    ax3.set_ylabel(r'$\Delta z$ r.m.s. error $(\mu m)$')
    ax3.set_xticks(ticks=[y + 1 for y in range(len(dfp_idpt.true_dz.unique()))], labels=dfp_idpt.true_dz.unique())
    ax3.set_xlabel(r'$\Delta z_{true} \: (\mu m)$')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------- PLOT RIGID DISPLACEMENT -----------------------------------------------


# gdpyt
test_rigid = False
if test_rigid:
    # idpt - meta assessment
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/results/idpt-meta-assessment-avg-2calib-1test/calibration-idpt-[17, 13]/'
    fn = 'calib_idpt_stats_11.06.21_z-micrometer-v2_cSILPURAN_17.xlsx'

    fp = '/Users/mackenzie/Desktop/dummy_coords_30micron_step_towards.xlsx'
    plot_rigid_displacement_test(test_coords_path=fp, spct_stats_path=base_dir + fn)


# ------------------------------------------------- PLOT SPCT STATS ----------------------------------------------------


# single particle calibration
spct = False
if spct:
    # base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/20X_1Xmag_2.15umNR_HighInt_0.03XHg/results/meta-assessment/calibration-plott-gen_cal/'
    # fn = 'calib_spct_stats_20X_1Xmag_2.15umNR_HighInt_0.03XHg_cGlass_g.xlsx'

    # idpt - meta assessment
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/results/idpt-meta-assessment-avg-2calib-1test/calibration-idpt-[17, 13]/'
    fn = 'calib_idpt_stats_11.06.21_z-micrometer-v2_cSILPURAN_17.xlsx'

    plot_spct_stats(spct_stats_path=base_dir + fn)

j = 1