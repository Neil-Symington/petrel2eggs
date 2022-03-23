import netCDF4
import numpy as np
import os
import matplotlib.pyplot as plt
from garjmcmctdem_utils import aem_utils, misc_utils, spatial_functions
from garjmcmctdem_utils import plotting_functions as plots
import rasterio
import pandas as pd

debugging = True

# for interactive mode change to directory with script
if debugging:
    os.chdir(r"C:\Users\u77932\PycharmProjects\AEM_interp_uncert\utilities")


# bring in the transD

ncinfile = r"C:\Users\u77932\Documents\DalyBasin\data\AEM\transD\Daly_transD_summary.nc"
transd_cond_grid_dir = r"C:\Users\u77932\Documents\DalyBasin\data\AEM\transD\cond_grid"
lci_cond_grid_dir = r"C:\Users\u77932\Documents\DalyBasin\data\AEM\lci\cond_grid"

transd = aem_utils.AEM_inversion(name='transd', inversion_type='stochastic',
                                 netcdf_dataset=netCDF4.Dataset(ncinfile))

ncinfile = r"C:\Users\u77932\Documents\DalyBasin\data\2017_DalyRiver_SkyTEM\03_LCI\01_Data\DalyR_WB_MGA52.nc"

lci = aem_utils.AEM_inversion(name='lci', inversion_type='deterministic',
                              netcdf_dataset=netCDF4.Dataset(ncinfile))

# import our model
inRaster = r"C:\Users\u77932\Documents\DalyBasin\export\topJinduckin_13Jan.tif"

src = rasterio.open(inRaster)

lines = [103501, 103502, 104501, 105901, 105902, 109301,
         109302, 109801, 110002, 110201]

grid_vars = ['phid_mean', 'phid_sdev', 'layer_centre_depth', 'conductivity_p10',
             'conductivity_p50', 'conductivity_mean', 'conductivity_p90', 'ddz_mean', 'ddz_sdev']
grid = False

if grid:
    transd.grid_sections(variables=grid_vars, lines=lines,
                         xres=200, yres=4,
                         return_interpolated=False,
                         save_to_disk=True, output_dir=transd_cond_grid_dir, sort_on='fiducial')
    lci.grid_sections(variables=['conductivity', 'resi1', 'tx_height_measured'], lines=lines,
                      xres=25, yres=4,
                      return_interpolated=False,
                      save_to_disk=True, output_dir=lci_cond_grid_dir, sort_on='fiducial')

# import our AEM interpretation from the eggs database

df_interp = pd.read_csv("../data/OollooJinduckin_interp_EGGS_compressed.csv")

# Uncertainty modelling using additive and multiplicative noise
additive_noise = 5.
mulitplicative_noise = 0.2

df_interp['UNCERTAINTY'] = df_interp['DEPTH'] * mulitplicative_noise + additive_noise
df_interp['UNCERTAINTY_DESC'] = '95% confidence'

# for the uncertainty estimation we need a function for converting depth of interpretation to an uncertainty

ltd = spatial_functions.layer_centre_to_top(transd.data['layer_centre_depth'][:])[0]
thickness = spatial_functions.depth_to_thickness(ltd)

for line_number in lines:
    plt.close("all")

    df_line = df_interp[df_interp['SURVEY_LINE'] == line_number]

    lci_xr = misc_utils.pickle2xarray(os.path.join(lci_cond_grid_dir,
                                                   '{}.pkl'.format(line_number)))
    transd_xr = misc_utils.pickle2xarray(os.path.join(transd_cond_grid_dir,
                                                      '{}.pkl'.format(line_number)))
    # create a spatial mask for the lci data as it has a larger extent

    xmin, xmax = np.min(transd_xr['easting'][:].values), np.max(transd_xr['easting'][:].values)
    spatial_mask = np.logical_and(lci_xr['easting'][:] < xmax, lci_xr['easting'][:] > xmin)

    # get the new grid distances, conductvity and other arrays
    grid_distances = lci_xr['grid_distances'].values[spatial_mask] - lci_xr['grid_distances'].values[
        spatial_mask].min()

    conductivity = lci_xr['conductivity'].values[:, spatial_mask]
    phid = lci_xr['resi1'].values[spatial_mask]
    tx_height = lci_xr['tx_height_measured'].values[spatial_mask]
    elevation = lci_xr['elevation'].values[spatial_mask]
    easting, northing = lci_xr['easting'].values[spatial_mask], lci_xr['northing'].values[spatial_mask]

    # create a new xarray
    new_xr = misc_utils.dict2xr(d={'grid_distances': grid_distances, 'conductivity': conductivity, 'phid': phid,
                                   'tx_height': tx_height, 'grid_elevations': lci_xr['grid_elevations'],
                                   'elevation': elevation,
                                   'easting': easting, 'northing': northing})
    # we need to find a grid distances array for our interpretation
    dist, ind = spatial_functions.nearest_neighbours(df_line[['X', 'Y']].values, np.column_stack((easting, northing)))

    mask = ~np.isnan(dist)

    df_line = df_line[mask]

    df_line['grid_distances'] = grid_distances[ind[mask]]

    # create a new xarray

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, figsize=(16, 16))

    cax = fig.add_axes([0.95, 0.3, 0.01, 0.4])

    ax1.plot(grid_distances, phid, c='blue', label='lci data residual')
    ax1.plot(transd_xr['grid_distances'], transd_xr['phid_mean'], c='red', label='transD mean residual')

    ax1.set_yscale('log')
    ax1.legend()

    cond_section = plots.plot_grid(ax2, new_xr, 'conductivity',
                                   panel_kwargs={'title': 'conductivity', 'max_depth': 500., 'vmin': 0.005,
                                                 'vmax': 1.0, 'cmap': 'viridis', 'ylabel': 'elevation \n (mAHD)',
                                                 'shade_doi': False, "log_plot": True})

    ax2.set_ylim(ax2.get_ylim()[0], elevation.max() + 40.)
    ax2.set_title('lci')
    ax2.set_ylabel('elevation')

    ax2.errorbar(df_line['grid_distances'].values, df_line['ELEVATION'].values,yerr=df_line['UNCERTAINTY'], linestyle="None",
                 label='Oolloo-Jinduckin', c='k')

    ax2.legend()
    fig.colorbar(cond_section, cax=cax)

    ax_vars = ['conductivity_p50', 'conductivity_p10', 'conductivity_p90']

    for i, ax in enumerate([ax3, ax4, ax5]):
        cond_section = plots.plot_grid(ax, transd_xr, ax_vars[i],
                                       panel_kwargs={'title': 'conductivity', 'max_depth': 500., 'vmin': 0.005,
                                                     'vmax': 1.0, 'cmap': 'viridis',
                                                     'ylabel': 'elevation \n (mAHD)', 'shade_doi': False,
                                                     "log_plot": True})

        ax.errorbar(df_line['grid_distances'].values, df_line['ELEVATION'].values,yerr=df_line['UNCERTAINTY'], c = 'k',
                 label='Oolloo-Jinduckin', linestyle="None")
        ax.set_title(ax_vars[i])

    plt.savefig("{}_model_section_compare.png".format(line_number), dpi=300.)