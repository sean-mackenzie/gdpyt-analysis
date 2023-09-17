# plot image stats
from utils import plot_collections

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
           'results-04.27.22_particle-similarities/mixed-datasets'

# ----------------------------------------------------------------------------------------------------------------------
# 1. READ FILES

analyze_similarity = True
if analyze_similarity:

    # simple
    plot_collections.plot_similarity_stats_simple(base_dir, min_percent_layers=0.5)

    # full analysis
    mean_min_dx = 25.5
    plot_collections.plot_similarity_analysis(base_dir, method='spct', mean_min_dx=mean_min_dx)

# ----------------------------------------------------------------------------------------------------------------------
# 2. SPCT STATS

analyze_spct_stats = False
if analyze_spct_stats:
    plot_collections.plot_spct_stats(base_dir)

# ----------------------------------------------------------------------------------------------------------------------
# 3. TEST COORDS SIMILARITY

analyze_test_similarity = False
if analyze_test_similarity:
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.style.use(['science', 'ieee', 'std-colors'])
    fig, ax = plt.subplots()
    sci_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    sci_color_cycler = ax._get_lines.prop_cycler
    size_x_inches, size_y_inches = fig.get_size_inches()
    plt.close(fig)

    sciblue = '#0C5DA5'
    scigreen = '#00B945'

    fpi = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/FINAL-04.21-22_IDPT_1um-calib_5um-test/coords/test-coords/test_coords_idpt-corrected-on-test.xlsx'
    fps = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/FINAL-04.25.22_SPCT_1um-calib_5um-test/coords/test-coords/test_coords_spct-corrected-on-test.xlsx'
    save_path = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/results-04.12.22-SPCT-5umStep-meta/figs/similarity-analysis'

    dfi = pd.read_excel(fpi)
    dfgi = dfi.groupby('z_true').mean()
    dfgistd = dfi.groupby('z_true').std()

    dfs = pd.read_excel(fps)
    dfgs = dfs.groupby('z_true').mean()
    dfgsstd = dfs.groupby('z_true').std()

    # plot

    fig, ax = plt.subplots()

    ax.scatter(dfi.z_true, dfi.cm, s=1, color=sciblue, alpha=0.125, label='IDPT')
    ax.scatter(dfs.z_true, dfs.cm, s=1, color=scigreen, alpha=0.125, label='SPCT')
    ax.axhline(0.5, linewidth=0.5, linestyle='--', color='red', alpha=0.5, label=r'$c_{m, threshold}$')

    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylabel(r'$c_{m}$')
    ax.set_ylim(bottom=0.25, top=1.01)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path + '/cm_by_z-true_IDPT-SPCT_all.png')
    plt.show()

    fig, ax = plt.subplots()

    ax.plot(dfgi.index, dfgi.cm, '-o', color=sciblue, label='IDPT')
    ax.plot(dfgs.index, dfgs.cm, '-o', color=scigreen, label='SPCT')

    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylabel(r'$c_{m}$')
    ax.set_ylim(bottom=0.5, top=1.01)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path + '/cm_by_z-true_IDPT-SPCT.png')
    plt.show()

    fig, ax = plt.subplots()

    ax.errorbar(dfgi.index, dfgi.cm, yerr=dfgistd.cm, fmt='-o', capsize=2, elinewidth=1, markersize=1, color=sciblue, label='IDPT')
    ax.errorbar(dfgs.index, dfgs.cm, yerr=dfgsstd.cm, fmt='-o', capsize=2, elinewidth=1, markersize=1, color=scigreen, label='SPCT')
    ax.errorbar(dfgi.index, dfgi.cm, yerr=dfgistd.cm, fmt='-o', capsize=2, elinewidth=1, markersize=1, color=sciblue)

    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylabel(r'$c_{m}$')
    ax.set_ylim(bottom=0.5, top=1.01)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path + '/cm_by_z-true_IDPT-SPCT_errorbar.png')
    plt.show()



print("Analysis completed without errors.")