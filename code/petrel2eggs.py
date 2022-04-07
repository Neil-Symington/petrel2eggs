import pandas as pd
import numpy as np
import os, sys
import segyio
import rasterio
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from rdp import rdp

debugging = False

if debugging:
    dname = r"C:\Users\u77932\PycharmProjects\AEM_interp_uncert\code"
else:
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)

os.chdir(dname)
sys.path.append(dname)

from utility_functions import estimate_uncertainty, coords2distance

# important variables
IESX_file = r"..\data\OollooJinduckin" # IESX seismic horizon file

segy_dir = r"C:\Users\u77932\Documents\DalyBasin\data\2017_DalyRiver_SkyTEM\03_LCI\segy"

# define a digital elevation model raster if you want an independent estimate of ground elevation
compare_dem = False

dem_file = r"C:\Users\u77932\Documents\DalyBasin\data\2017_DalyRiver_SkyTEM\02_DEM\Grid\AUS_10021_DalyR_DTM_AHD.ers"

template_file = r"..\data\OollooJinduckin_EGGS_template.csv"

compressedOutfile = r"../data/OollooJinduckin_interp_EGGS_compressed.csv"

uncompressedOutfile = r"../data/OollooJinduckin_interp_EGGS.csv"

# uncertainty parameters

use = "layer_thickness"
additive_noise = 10.
mulitplicative_noise = 3
uncertDescr = "95% confidence"

thickness = np.array([5., 5.4, 5.8, 6.3, 6.7, 7.3, 7.8, 8.4, 9.1, 9.8, 10.6, 11.4, 12.3, 13.2, 14.3 ,15.4, 16.6,  17.9,
                     19.3,20.8,22.4, 24.1, 26.0, 28.0, 30.2, 32.5, 35.1, 37.8, 40.7, 44.0]) # from skytem lci inversion
# parse data

df_interp = pd.read_csv(IESX_file, header=None, usecols = [0,1,2,4,9], skiprows = [0,1], delim_whitespace=True,
                    skipfooter=1,  names = ["X", "Y", "SEGMENT_ID", "PETREL_ELEVATION", "SURVEY_LINE"])

df_interp['ELEVATION'] = -1*df_interp['PETREL_ELEVATION']

# add the DEM
df_interp['DEM'] = np.nan

# now we want to find the ground elevation of the interpretation based on the trace number


# iterate through survey lines and extract the X,Y,Z and trace data for each fiducial from the segy file
for line in df_interp.SURVEY_LINE.unique():
    # subset the interpretations for the line
    df_line = df_interp[df_interp['SURVEY_LINE'] == line]
    filename = os.path.join(segy_dir, "{}.segy".format(line))
    # open segy
    with segyio.open(filename, ignore_geometry=True) as f:
        # get data as arrays
        groupX = f.attributes(segyio.TraceField.GroupX)[:]
        groupY = f.attributes(segyio.TraceField.GroupY)[:]
        trace = f.attributes(segyio.TraceField.TRACE_SEQUENCE_LINE)[:]
        groundElevation = f.attributes(segyio.TraceField.ElevationScalar)[:]
        # use a nearest neighbour search to find the ground elevation and trace at each interpretation point
        tree = KDTree(np.column_stack((groupX, groupY)))
        dist, ind = tree.query(df_line[['X', 'Y']])
        df_interp.at[df_line.index, "DEM"] = groundElevation[ind]
        df_interp.at[df_line.index, "TRACE"] = trace[ind]

# open the template
template_file = r"..\data\OollooJinduckin_EGGS_template.csv"

df_template = pd.read_csv(template_file)

# sample a DEM if you want to check the values extracted from the segy

if compare_dem:


    src = rasterio.open(dem_file)

    easting = df_interp.X.values
    northing= df_interp.Y.values

    DEM = np.nan*np.ones(shape = len(df_interp), dtype = float)

    for i, item in enumerate(src.sample(zip(easting,northing))):
        DEM[i] = item[0]

    # plot to check correctness
    plt.scatter(DEM, df_interp['DEM'])
    plt.show()
#calculate the depth below ground level for each interpretation
df_interp['DEPTH'] = df_interp['DEM'] - df_interp["ELEVATION"]

# now add the template variables
for col in df_template.columns:
    df_interp[col] = df_template[col][0]

# Calculate uncertainty

# we are going TO make uncertainty = to 0.2 multiplied by the depth plus 10 metres

df_interp['UNCERTAINTY'] = estimate_uncertainty(df_interp.DEPTH.values, multiplicative_noise = mulitplicative_noise,
                                                additive_noise = additive_noise, use = use, thickness = thickness)
df_interp['UNCERTAINTY_DESC'] = uncertDescr

df_interp.to_csv(uncompressedOutfile, index=False)

df_interp['VERTEX_NO'] = 0
df_interp['rtp_mask'] = 0
epsilon = 1. # for the rtp algorithm

# now we want to do some compression using the Ramer Douglas Peuker algorithm
# iterate through each line
for line in df_interp.SURVEY_LINE.unique():
    df_line = df_interp[df_interp['SURVEY_LINE'] == line]
    df_line = df_line.sort_values('TRACE') # sort
    df_line['distance'] = coords2distance(np.column_stack((df_line.X, df_line.Y)))
    delta_d = np.diff(df_line['distance'])
    # now implement the rtp
    for segment in df_line['SEGMENT_ID'].unique():
        df_segment = df_line[df_line['SEGMENT_ID'] == segment]
        rtp_mask = rdp(df_segment[['distance', 'ELEVATION']].values,
                                                         epsilon = epsilon, algo="iter", return_mask=True)
        df_interp.at[df_segment.index, 'rtp_mask'] = rtp_mask

# export the compressed data
df_interp_compressed = df_interp[df_interp['rtp_mask'].values].drop(columns = ['PETREL_ELEVATION', 'rtp_mask'])

# add the vertex number to the compressed data

for line in df_interp_compressed.SURVEY_LINE.unique():
    df_line = df_interp_compressed[df_interp_compressed['SURVEY_LINE'] == line]
    # sort on trace
    df_line = df_line.sort_values('TRACE')
    # iterate through each segment
    for segment in df_line['SEGMENT_ID'].unique():
        df_segment = df_line[df_line['SEGMENT_ID'] == segment]
        df_interp_compressed.at[df_segment.index, 'VERTEX_NO'] = np.arange(1, len(df_segment) + 1)


df_interp_compressed.to_csv(compressedOutfile, index=False)