import pandas as pd
import numpy as np
import os
import segyio
import rasterio
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import math
from rdp import rdp


def line_length(line):
    '''
    Function to return length of line
    @param line: iterable containing two two-ordinate iterables, e.g. 2 x 2 array or 2-tuple of 2-tuples

    @return length: Distance between start & end points in native units
    '''
    return math.sqrt(math.pow(line[1][0] - line[0][0], 2.0) +
                     math.pow(line[1][1] - line[0][1], 2.0))

def coords2distance(coordinate_array):
    '''
    From geophys_utils, transect_utils

    Function to calculate cumulative distance in metres from native (lon/lat) coordinates
    @param coordinate_array: Array of shape (n, 2) or iterable containing coordinate pairs

    @return distance_array: Array of shape (n) containing cumulative distances from first coord
    '''
    coord_count = coordinate_array.shape[0]
    distance_array = np.zeros((coord_count,), coordinate_array.dtype)
    cumulative_distance = 0.0
    distance_array[0] = cumulative_distance
    last_point = coordinate_array[0]

    for coord_index in range(1, coord_count):
        point = coordinate_array[coord_index]
        distance = line_length((point, last_point))
        cumulative_distance += distance
        distance_array[coord_index] = cumulative_distance
        last_point = point

    return distance_array

debugging = True

# for interactive mode change to directory with script
if debugging:
    os.chdir(r"C:\Users\u77932\PycharmProjects\AEM_interp_uncert\utilities")

infile = r"..\data\OollooJinduckin" # IESX seismic horizon file

df_interp = pd.read_csv(infile, header=None, usecols = [0,1,2,4,9], skiprows = [0,1], delim_whitespace=True,
                    skipfooter=1,  names = ["X", "Y", "SEGMENT_ID", "PETREL_ELEVATION", "SURVEY_LINE"])

df_interp['ELEVATION'] = -1*df_interp['PETREL_ELEVATION']

# add the DEM
df_interp['DEM'] = np.nan

# now we want to find the ground elevation of the interpretation based on the trace number
segy_dir = r"C:\Users\u77932\Documents\DalyBasin\data\2017_DalyRiver_SkyTEM\03_LCI\segy"

for line in df_interp.SURVEY_LINE.unique():
    df_line = df_interp[df_interp['SURVEY_LINE'] == line]
    filename = os.path.join(segy_dir, "{}.segy".format(line))
    with segyio.open(filename, ignore_geometry=True) as f:
        groupX = f.attributes(segyio.TraceField.GroupX)[:]
        groupY = f.attributes(segyio.TraceField.GroupY)[:]
        trace = f.attributes(segyio.TraceField.TRACE_SEQUENCE_LINE)[:]
        groundElevation = f.attributes(segyio.TraceField.ElevationScalar)[:]
        tree = KDTree(np.column_stack((groupX, groupY)))
        dist, ind = tree.query(df_line[['X', 'Y']])
        df_interp.at[df_line.index, "DEM"] = groundElevation[ind]
        df_interp.at[df_line.index, "TRACE"] = trace[ind]

template_file = r"..\data\OollooJinduckin_EGGS_template.csv"

df_template = pd.read_csv(template_file)

# sample a DEM

compare_dem = False

if compare_dem:
    dem_file = r"C:\Users\u77932\Documents\DalyBasin\data\2017_DalyRiver_SkyTEM\02_DEM\Grid\AUS_10021_DalyR_DTM_AHD.ers"

    src = rasterio.open(dem_file)

    easting = df_interp.X.values
    northing= df_interp.Y.values

    DEM = np.nan*np.ones(shape = len(df_interp), dtype = float)

    for i, item in enumerate(src.sample(zip(easting,northing))):
        DEM[i] = item[0]

    # plot to check correctness
    plt.scatter(DEM, df_interp['DEM'])
    plt.show()

df_interp['DEPTH'] = df_interp['DEM'] - df_interp["ELEVATION"]

# now add the template variables
for col in df_template.columns:
    df_interp[col] = df_template[col][0]

df_interp.to_csv(r"../data/OollooJinduckin_interp_EGGS.csv", index=False)

df_interp['VERTEX_NO'] = 0
df_interp['rtp_mask'] = 0
epsilon = 1. # for the rtp algorithm

# now we want to do some compression using the
for line in df_interp.SURVEY_LINE.unique():
    df_line = df_interp[df_interp['SURVEY_LINE'] == line]
    df_line = df_line.sort_values('TRACE')
    df_line['distance'] = coords2distance(np.column_stack((df_line.X, df_line.Y)))
    delta_d = np.diff(df_line['distance'])
    # now implement the rtp
    for segment in df_line['SEGMENT_ID'].unique():
        df_segment = df_line[df_line['SEGMENT_ID'] == segment]
        rtp_mask = rdp(df_segment[['distance', 'ELEVATION']].values,
                                                         epsilon = epsilon, algo="iter", return_mask=True)
        df_interp.at[df_segment.index, 'rtp_mask'] = rtp_mask

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


df_interp_compressed.to_csv(r"../data/OollooJinduckin_interp_EGGS_compressed.csv", index=False)