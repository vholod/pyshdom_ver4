import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pyshdom
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import scipy.io as sio
import os
import logging
from collections import OrderedDict

def float_round(x):
    """Round a float or np.float32 to a 3 digits float"""
    if type(x) == np.float32:
        x = x.item()
    return round(x,3) 

try:
    from roipoly import RoiPoly
    # I take it from: https://github.com/jdoepfert/roipoly.py
    # Based on a code snippet originally posted by Daniel Kornhauser
    # (http://matplotlib.1069221.n5.nabble.com/How-to-draw-a-region-of-interest-td4972.html).
except ModuleNotFoundError as e:
    print(e)  # so do pip install roipoly

# ask the user to set the name of the cloud field
while True:
    try:
        is_bomex_or_cass = input("Enter B for BOMEX or C for CASS Clouds Fields: ")
        if is_bomex_or_cass in 'BC':
            break
        print("Invalid cloud field entered.")
    except Exception as e:
        print(e)

cloud_id = input("Enter last 5 digits of cloud id:")

base_path = '/wdata/clouds_from_weizmann/CASS_50m_256x256x139_600CCN' if is_bomex_or_cass == 'C' else '/wdata/clouds_from_weizmann/BOMEX_512X512X170_500CCN_20m'
cloud_name = f'CASS_256x256x139_50m_600CNN_micro_256_00000{cloud_id}_ONLY_RE_VE_LWC.mat' if is_bomex_or_cass == 'C' else f'BOMEX_512x512x170_500CCN_20m_micro_256_00000{cloud_id}_ONLY_RE_VE_LWC.mat'

data_path = os.path.join(base_path, 'processed', cloud_name)


def create_and_configer_logger(log_name):
    """
    TODO
    Args:
        log_name ():

    Returns:

    """
    # set up logging to file
    logging.basicConfig(
        filename=log_name,
        level=logging.INFO,
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    )

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger(__name__)
    return logger


logger = create_and_configer_logger('WIZ_Fields_to_mat_log.log')


# -----functions------------------------------------:
def calc_high_map(volumefield, zgrid):
    """
    Extracts top of the clouds.

    """
    nx, ny, nz = volumefield.shape
    K = np.zeros([nx, ny])
    K = np.cumsum(volumefield, axis=2)
    K = np.argmax(K, axis=2)
    high_map = zgrid[K]
    return high_map


# load 3d data:
logger.info('------------- New Crop ---------------')
logger.info(f'load {data_path}')
data3d = sio.loadmat(data_path)
lwc = data3d['lwc']
reff = data3d['reff']
veff = data3d['veff']
# get relevant params
xgrid, ygrid, zgrid = np.round(data3d['x'].flatten() * 1e-3, 3), np.round(data3d['y'].flatten() * 1e-3, 3), np.round(
    data3d['z'].flatten() * 1e-3, 3)
nx, ny, nz = len(xgrid), len(ygrid), len(zgrid)
logger.info(f"nx:{nx}, ny:{ny}, nz:{nz}")
x_min, y_min, z_min = xgrid.min(), ygrid.min(), zgrid.min()
x_max, y_max, z_max = xgrid.max(), ygrid.max(), zgrid.max()
logger.info(f"x_min, y_min, z_min, x_max, y_max, z_max:{x_min}, {y_min}, {z_min}, {x_max}, {y_max}, {z_max}")
# Compute dx, dy
dx = np.unique(np.round(np.diff(xgrid), 3))
dy = np.unique(np.round(np.diff(ygrid), 3))
dz = np.unique(np.round(np.diff(zgrid), 3))[0] # not to use without extra care
logger.info(f'dx:{dx},dy:{dy}')
assert (len(dx) == 1) and (len(dy) == 1), 'dx or dy are not uniform'
dx, dy = dx[0], dy[0]

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------

"""
Cut one piece from the cloud/cloud field.

1. Extract the high map of the cloud field
2. The user interactively draw a polygon within the image by clicking with the left mouse button 
to select the vertices of the polygon. To close the polygon, click with the right mouse button.
After finishing the ROI, the current figure is closed so that the execution of the code can continue.
3. The function get_mask(image) creates a binary mask for a certain ROI instance, that is, a 2D numpy 
array of the size of the image array, whose elements are True if they lie inside the ROI polygon, 
and False otherwise.
notes:
The new mediume will be padded with zeros on its outer boundaries.

"""

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------

# 1.
# -----------------------------------------------------
# ---------------calc_high_map from LWC data-----------
# -----------------------------------------------------
REGULAR_LWC_DATA = np.nan_to_num(lwc)
# type(EGULAR_LWC_DATA) is numpy.ndarray
High_map = calc_high_map(REGULAR_LWC_DATA, zgrid)

fig = plt.figure(figsize=(20, 20))

plt.imshow(High_map, vmin=0, vmax=np.amax(High_map))
plt.title('Full High map')
plt.gca().invert_yaxis()
my_roi = RoiPoly(color='r')  # draw new ROI in red color

# lets the user choose the roi.
my_roi.display_roi()
# Extracting a binary mask image.
mask = my_roi.get_mask(High_map)

XX = np.arange(0, nx)
YY = np.arange(0, ny)
XX, YY = np.meshgrid(XX, YY, indexing='ij')

Xpoints = np.trim_zeros((XX[mask]).ravel())
Ypoints = np.trim_zeros((YY[mask]).ravel())

min_x_index = min(Xpoints)
min_y_index = min(Ypoints)

max_x_index = max(Xpoints)
max_y_index = max(Ypoints)

logger.info("when cut from high map")
logger.info("The min_x_index is {}".format(min_x_index))
logger.info("The min_y_index is {}".format(min_y_index))
logger.info("The max_x_index is {}".format(max_x_index))
logger.info("The max_y_index is {}".format(max_y_index))

# I always have the problem with the rounding (e.g. 0.2400000000014 or 0.2399999999), so I use Utils.float_round
min_x_coordinates = float_round(xgrid[min_x_index])
min_y_coordinates = float_round(ygrid[min_y_index])

max_x_coordinates = float_round(xgrid[max_x_index] - dx)
max_y_coordinates = float_round(ygrid[max_y_index] - dy)
# why -dx and -dy?
# Becouse The Xmin is here ->|_|_|_|_|_|_|_|_|.
# The Xmax is one dx befor the last corner (|_|_|_|_|_|_|_->|_|). I made a lot of miskaes when I implemented it.

cumsum_volume_axis22 = np.cumsum(REGULAR_LWC_DATA, axis=2, dtype=float)
# GET RID OF ZEROS: find the minimum excluding zeros,
tmp = np.sum(np.sum(cumsum_volume_axis22, axis=1), axis=0)
tmp[tmp == 0] = max(tmp)
# GET RID OF ZEROS -> done
Bottom_minimum = np.argmin(tmp)

min_z_index = Bottom_minimum
max_z_index = np.amax(np.argmax(cumsum_volume_axis22, axis=2))

# crop the fields:
CROPED_DATA_DICT = OrderedDict()
for data_name in ('lwc' , 'reff', 'veff'):
    field = eval(data_name)
    CROPED_DATA_DICT[data_name] = field[min_x_index:max_x_index, \
                                 min_y_index:max_y_index, \
                                 min_z_index:max_z_index]

# ----------------------------------------------------------

min_z_coordinates = float_round(zgrid[min_z_index])
max_z_coordinates = float_round(zgrid[max_z_index - 1])


# I always do the padding. 
# I Pad with zeros on the sides and bottom top.
# So:
PAD_ON_SIDE   = 2 # how much to pad each side?
PAD_ON_BOTTOM = 2 # how much to pad bottom?
PAD_ON_TOP    = 2 # how much to pad top?

# before the padding, the coordinates are:
# min_x_coordinates -> max_x_coordinates
# min_y_coordinates -> max_y_coordinates
# min_z_coordinates -> max_z_coordinates
# these coordinates are of new_field.

# APPLY PADDING:

# The +- ds is becouse of the padding
min_x_coordinates = float_round(min_x_coordinates - PAD_ON_SIDE*dx)
max_x_coordinates = float_round(max_x_coordinates + PAD_ON_SIDE*dx)

min_y_coordinates = float_round(min_y_coordinates - PAD_ON_SIDE*dy)
max_y_coordinates = float_round(max_y_coordinates + PAD_ON_SIDE*dy)

PAD_ON_BOTTOM = min( (min_z_coordinates/dz), PAD_ON_BOTTOM )
min_z_coordinates = float_round(min_z_coordinates - PAD_ON_BOTTOM*dz)
max_z_coordinates = float_round(max_z_coordinates + PAD_ON_TOP*dz)

Xrange = [min_x_coordinates, max_x_coordinates]
Yrange = [min_y_coordinates, max_y_coordinates]
Zrange = [min_z_coordinates, max_z_coordinates]
# The + 2X is becouse of the padding
new_nx = max_x_index - min_x_index + 2*PAD_ON_SIDE
new_ny = max_y_index - min_y_index + 2*PAD_ON_SIDE
new_nz = max_z_index - min_z_index + PAD_ON_BOTTOM + PAD_ON_TOP

min_z_index = min_z_index - PAD_ON_BOTTOM
max_z_index = max_z_index + PAD_ON_TOP
updated_zgrid = zgrid[min_z_index:max_z_index]

logger.info("After the cut from high map")
logger.info("The min_x_coordinates is {}".format(min_x_coordinates))
logger.info("The min_y_coordinates is {}".format(min_y_coordinates))
logger.info("The max_x_coordinates is {}".format(max_x_coordinates))
logger.info("The max_y_coordinates is {}".format(max_y_coordinates))
logger.info("The min_z_coordinates is {}".format(min_z_coordinates)) # TODO
logger.info("The max_z_coordinates is {}".format(max_z_coordinates))

NEW_DATA_DICT = OrderedDict() # after crop and pad
# set grid using new pyshdom:
# make a grid for microphysics which is just the cloud grid.
cloud_scatterer = pyshdom.grid.make_grid(dx,new_nx,\
                          dy,new_ny,updated_zgrid)

non_zero_indexes = np.where(CROPED_DATA_DICT['lwc']>0)
i, j, k = non_zero_indexes

for data_name in ('lwc' , 'reff', 'veff'):
    field = CROPED_DATA_DICT[data_name]
    new_field = np.pad(field, ((PAD_ON_SIDE, PAD_ON_SIDE), \
                                   (PAD_ON_SIDE, PAD_ON_SIDE), \
                                   (PAD_ON_BOTTOM, PAD_ON_TOP)),\
                       'constant', constant_values=0)
    
    NEW_DATA_DICT[data_name] = new_field
    
    #initialize with np.nans so that empty data is np.nan    
    this_data = np.zeros((cloud_scatterer.sizes['x'], \
                cloud_scatterer.sizes['y'], cloud_scatterer.sizes['z']))*np.nan
    this_data[i, j, k] = new_field[i, j, k]
    cloud_scatterer[data_name] = (['x', 'y', 'z'], this_data)
    

# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------

print(cloud_scatterer.info())
# The type of cloud_scatterer is <xarray.Dataset>.
# lwc, reff and veff iare the Data variables of cloud_scatterer.
print(cloud_scatterer.data_vars)

# Visualize LWC data to validate the crop:
REGULAR_LWC_DATA = np.nan_to_num(cloud_scatterer.lwc)
# type(EGULAR_LWC_DATA) is numpy.ndarray

xgrid = cloud_scatterer.x
ygrid = cloud_scatterer.y
zgrid = cloud_scatterer.z

mlab.figure(size=(600, 600))
X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
figh = mlab.gcf()
src = mlab.pipeline.scalar_field(X, Y, Z, REGULAR_LWC_DATA, figure=figh)
src.spacing = [dx, dy, dz]
src.update_image_data = True

isosurface = mlab.pipeline.iso_surface(src, contours=[0.1*REGULAR_LWC_DATA.max(),\
                                                      0.2*REGULAR_LWC_DATA.max(),\
                                                      0.3*REGULAR_LWC_DATA.max(),\
                                                      0.4*REGULAR_LWC_DATA.max(),\
                                                      0.5*REGULAR_LWC_DATA.max(),\
                                                      0.6*REGULAR_LWC_DATA.max(),\
                                                      0.7*REGULAR_LWC_DATA.max(),\
                                                      0.8*REGULAR_LWC_DATA.max(),\
                                                      0.9*REGULAR_LWC_DATA.max(),\
                                                      ],opacity=0.9,figure=figh)

mlab.show()






# save the txt file:
new_cloud_name = f"{cloud_name.split('_')[0]}_{cloud_id}_{new_nx}x{new_ny}x{new_nz}_{''.join([str(elem) for elem in np.random.randint(low=0, high=9, size=4)])}"
file_name = os.path.join(base_path, 'cropped_for_new_pyshdom', new_cloud_name)
logger.info(f'saving to {file_name}')

# ---------------------------------------------------------
# -------------SAVE TO CSV --------------------------------
# ---------------------------------------------------------

"""
------------- LIAM TODO ---------------.

A utility function to save a microphysical medium.
After implementation put as a function in util.py under the name 
save_to_csv.

Format:


Parameters
----------
path: str
    Path to file.
comment_line: str, optional
    A comment line describing the file.


Notes
-----
CSV format is as follows:
# comment line (description)
nx,ny,nz # nx,ny,nz
dx,dy # dx,dy [km, km]
z_levels[0]     z_levels[1] ...  z_levels[nz-1] 
x,y,z,lwc,reff,veff
ix,iy,iz,lwc[ix, iy, iz],reff[ix, iy, iz],veff[ix, iy, iz]
.
.
.
ix,iy,iz,lwc[ix, iy, iz],reff[ix, iy, iz],veff[ix, iy, iz]



"""

mlab.show()

logger.info('done')
