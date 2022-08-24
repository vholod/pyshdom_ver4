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

CLOUD_FILE_NAME = 'BOMEX_57600_58x65x42_7844.txt'

base_path = "/wdata/clouds_from_weizmann/BOMEX_512X512X170_500CCN_20m/cropped_for_new_pyshdom/"



file_name = os.path.join(base_path, CLOUD_FILE_NAME)
vol_filename = CLOUD_FILE_NAME.split('.')[0] + '.vol'
vol_filename = os.path.join(base_path, vol_filename)
objfilename  = CLOUD_FILE_NAME.split('.')[0] + '.obj'
objfilename = os.path.join(base_path, objfilename)



cloud_scatterer = pyshdom.util.load_from_csv(file_name, density='lwc')

# Visualize LWC data to validate the load:
REGULAR_LWC_DATA = np.nan_to_num(cloud_scatterer.density)
# type(EGULAR_LWC_DATA) is numpy.ndarray

# The Nones make problems in the pyshdom.grid.resample_onto_grid()
# So I get rid of them.
cloud_scatterer['density'] = (['x', 'y', 'z'], np.nan_to_num(cloud_scatterer.density))
cloud_scatterer['reff'] = (['x', 'y', 'z'], np.nan_to_num(cloud_scatterer.reff))
cloud_scatterer['veff'] = (['x', 'y', 'z'], np.nan_to_num(cloud_scatterer.veff))

xgrid = cloud_scatterer.x
ygrid = cloud_scatterer.y
zgrid = cloud_scatterer.z

dx = cloud_scatterer.delx.item()
dy = cloud_scatterer.dely.item()
dz = round(np.diff(zgrid)[0],5) 

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

# find reff/veff min/max:
a=np.array(cloud_scatterer['reff'])
b = a[a>0]
reff_min = b.min() - 0.3
reff_max = b.max()

a=np.array(cloud_scatterer['veff'])
b = a[a>0]
veff_min = b.min()
veff_max = b.max() + 0.01

# -----------------------------------------------------
# make a grid for microphysics which is just the cloud grid.
rte_grid = pyshdom.grid.make_grid(dx,cloud_scatterer.x.data.size,
                          dy,cloud_scatterer.y.data.size,
                          cloud_scatterer.z)

cloud_scatterer_on_rte_grid = pyshdom.grid.resample_onto_grid(rte_grid, cloud_scatterer)

# We choose a gamma size distribution and therefore need to define a 'veff' variable.
size_distribution_function = pyshdom.size_distribution.gamma

# Exact OpticalPropertyGenerator:
mie_mono_table = pyshdom.mie.get_mono_table(
    'Water',(0.65,0.65),
    max_integration_radius=65.0,
    minimum_effective_radius=0.1,
    relative_dir='../mie_tables',
    verbose=False
)
mie_mono_tables = OrderedDict()
mie_mono_tables[0.65] = mie_mono_table

optical_prop_gen = pyshdom.medium.OpticalPropertyGenerator(
    'cloud',
    mie_mono_tables, 
    size_distribution_function,
    particle_density=1.0, 
    maxnphase=None,
    interpolation_mode='exact',
    density_normalization='density',#The density_normalization argument is a convenient
    reff=np.linspace(reff_min-0.3,30.0,50),
    veff=np.linspace(0.01,veff_max,15)
)

optical_properties = optical_prop_gen(cloud_scatterer_on_rte_grid)
# The optical properties produced by this contain all of the information 
# required for the RTE solver. Note that the attributes track the inputs
# and the microphysical properties are also brought along for traceability
# purposes.

# If you generate your own optical properties they must pass this check to be used in the solver.
pyshdom.checks.check_optical_properties(optical_properties[0.65])

# Extinction:
extinction = np.array(optical_properties[0.65].extinction)

mlab.figure(size=(600, 600))
figh = mlab.gcf()
src = mlab.pipeline.scalar_field(X, Y, Z, extinction, figure=figh)
src.spacing = [dx, dy, dz]
src.update_image_data = True
isosurface = mlab.pipeline.iso_surface(src, contours=[0.1*extinction.max(),\
                                                      0.2*extinction.max(),\
                                                      0.3*extinction.max(),\
                                                      0.4*extinction.max(),\
                                                      0.5*extinction.max(),\
                                                      0.6*extinction.max(),\
                                                      0.7*extinction.max(),\
                                                      0.8*extinction.max(),\
                                                      0.9*extinction.max(),\
                                                      ],opacity=0.9,figure=figh)

color_bar = mlab.colorbar(title='extinction', orientation='vertical', nb_labels=5)

mlab.show()

# -----------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------------
"""
 -----------ONLY SUPPORT ONE CHANNEL:---------

Convert only the volumefield to .vol file (Mitsuba1).
The volumefield must be the exctinction [1/m or 1/km] otherwise the .vol file isn't relavant. 
"""
# Convert 3D matrix (.mat) to binary file (.vol) as it is used in Mitsuba1.
# inpute, volfilename - output .vol file name 
# Code is based on .vol file description presented in [1]
#
# [1] -  http://www.mitsuba-renderer.org/releases/current/documentation.pdf
# chapter 8.7.2. Grid-based volume data source (gridvolume)
_VOL_VERSION = 3
import struct
values = extinction
scale = values.max()
values = values/scale
print('The scale is {}'.format(scale))

N_x, N_y, N_z = values.shape       
N_chan = 1

xmin = 0
xmax = round(xgrid.max().item() + dx,5)

ymin = 0
ymax = round(ygrid.max().item() + dy,5)

zmin = zgrid.min()
zmax = round(zgrid.max().item() + dz,5)

box = [xmin,ymin,zmin,xmax,ymax,zmax]

with open(vol_filename, 'wb') as f:
    f.write(b'V')
    f.write(b'O')
    f.write(b'L')
    f.write(np.uint8(3).tobytes())  # Version
    f.write(np.int32(1).tobytes())  # type
    f.write(np.int32(values.shape[0]).tobytes())  # size
    f.write(np.int32(values.shape[1]).tobytes())
    f.write(np.int32(values.shape[2]).tobytes())
    if values.ndim == 3:
        f.write(np.int32(1).tobytes())  # channels
    else:
        f.write(np.int32(values.shape[3]).tobytes())  # channels
    f.write(np.float32(xmin).tobytes())  # bbox
    f.write(np.float32(ymin).tobytes())
    f.write(np.float32(zmin).tobytes())
    f.write(np.float32(xmax).tobytes())
    f.write(np.float32(ymax).tobytes())
    f.write(np.float32(zmax).tobytes())
    f.write(values.ravel(order='F').astype(np.float32).tobytes())


# Save obj file:
with open(objfilename, 'w') as f:
                
    # Define all the vertices
    f.write('v '+str(xmax)+' '+str(ymax)+' '+str(zmin)+'\n') 
    f.write('v '+str(xmax)+' '+str(ymin)+' '+str(zmin)+'\n') 
    f.write('v '+str(xmin)+' '+str(ymin)+' '+str(zmin)+'\n')
    
    f.write('v '+str(xmin)+' '+str(ymax)+' '+str(zmin)+'\n')
    f.write('v '+str(xmax)+' '+str(ymax)+' '+str(zmax)+'\n')
    f.write('v '+str(xmax)+' '+str(ymin)+' '+str(zmax)+'\n')
    
    f.write('v '+str(xmin)+' '+str(ymin)+' '+str(zmax)+'\n')
    f.write('v '+str(xmin)+' '+str(ymax)+' '+str(zmax)+'\n')
    # Define all the normals 
    f.write('vn 0.000000 1.000000 0.000001\n')
    f.write('vn 0.000000 1.000000 0.000000\n')
    f.write('vn -1.000000 0.000000 -0.000000\n')
    
    f.write('vn -0.000000 -1.000000 -0.000001\n')
    f.write('vn -0.000000 -1.000000 0.000000\n')
    f.write('vn 1.000000 0.000000 -0.000001\n')
    
    f.write('vn 1.000000 -0.000001 0.000001\n')
    f.write('vn -0.000000 -0.000000 1.00000\n')
    f.write('vn 0.000000 0.000000 -1.000000\n')
    
    # Define the connectivity
    f.write('f 5//1 1//1 4//1\n')
    f.write('f 5//2 4//2 8//2\n')
    f.write('f 3//3 7//3 8//3\n')
    
    f.write('f 3//3 8//3 4//3\n')
    f.write('f 2//4 6//4 3//4\n')
    f.write('f 6//5 7//5 3//5\n')
    
    f.write('f 1//6 5//6 2//6\n')
    f.write('f 5//7 6//7 2//7\n')
    f.write('f 5//8 8//8 6//8\n')
    
    f.write('f 8//8 7//8 6//8\n')
    f.write('f 1//9 2//9 3//9\n')
    f.write('f 1//9 3//9 4//9\n')       
    
print('done')