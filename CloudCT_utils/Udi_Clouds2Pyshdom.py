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
import glob
import re 
from CloudCTUtils import *




"""
Currently, the lwc is given and I assume reff of 10.
"""
# -----------------------------------------------------
# -------- Input parameters by the user:---------------
# -----------------------------------------------------
IFVISUALIZE = False

dx = 0.02 #km
dy = 0.02 #km
dz = 0.02 #km
# -----------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------------


ref_path = "/wdata/pyshdom_yael/CloudGeometry/clouds from Udi/16samples"
FILES = sorted(glob.glob(ref_path + '/*.m'))

for lwc_path in FILES:
    vol_name = os.path.split(lwc_path)[-1] 
    pysdom_output_file_name = os.path.join(ref_path, vol_name+'_new_pyshdom.txt')
    
    print("loading the 3D mat from: {}".format(lwc_path))
    lwc = sio.loadmat(lwc_path)['data'] 
    nz, nx, ny = lwc.shape
    lwc = np.transpose(lwc, (1, 2, 0)) # Since udi use z , x , y shape
    # TODO - read also the REFF and transfer to the output csv file.
    reff = np.zeros_like(lwc)
    veff = np.zeros_like(lwc)
    reff[lwc>0] = 10
    veff[lwc>0] = 0.1
    DATA_DICT = OrderedDict()
    DATA_DICT['lwc']  = lwc
    DATA_DICT['reff'] = reff
    DATA_DICT['veff'] = veff
    
    #----------------------------------
    
    XX = np.linspace(0, (nx-1)*dx, nx)
    YY = np.linspace(0, (ny-1)*dy, ny)
    zgrid = np.linspace(0, (nz-1)*dz, nz)
    XX, YY ,ZZ = np.meshgrid(XX, YY, zgrid, indexing='ij')
    
    index_x = np.arange(0, nx)
    index_y = np.arange(0, ny)
    index_z = np.arange(0, nz)
    index_X, index_Y ,index_Z = np.meshgrid(index_x, index_y, index_z, indexing='ij')
    
    
    # visualization:
    if IFVISUALIZE:
        
        mlab.figure(size=(600, 600))
        figh = mlab.gcf()
        src = mlab.pipeline.scalar_field(XX, YY, ZZ, lwc, figure=figh)
        src.spacing = [dx, dy, dz]
        src.update_image_data = True
        
        isosurface = mlab.pipeline.iso_surface(src, contours=[0.1*lwc.max(),\
                                                              0.2*lwc.max(),\
                                                              0.3*lwc.max(),\
                                                              0.4*lwc.max(),\
                                                              0.5*lwc.max(),\
                                                              0.6*lwc.max(),\
                                                              0.7*lwc.max(),\
                                                              0.8*lwc.max(),\
                                                              0.9*lwc.max(),\
                                                              ],opacity=0.9,figure=figh)
        
        mlab.show()
    
    # Convert medium to pyshdom:
    # set grid using new pyshdom:
    # make a grid for microphysics which is just the cloud grid.
    cloud_scatterer = pyshdom.grid.make_grid(dx,nx,\
                              dy,ny,zgrid)
    
    non_zero_indexes = np.where(DATA_DICT['lwc']>0)
    i, j, k = non_zero_indexes
    
    # I Pad with zeros on the sides and bottom top.
    # So:
    PAD_ON_SIDE   = 2 # how much to pad each side?
    PAD_ON_BOTTOM = 2 # how much to pad bottom?
    PAD_ON_TOP    = 2 # how much to pad top?
    
    NEW_DATA_DICT = OrderedDict() # after pad
    
    for data_name in ('lwc' , 'reff', 'veff'):
        field = DATA_DICT[data_name]
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
    
    print(cloud_scatterer.info())
    print(cloud_scatterer.data_vars)
    
    comment_line = '# A part of the Udis data 2022. Original cloud is {}'.format(vol_name)
    save_to_csv(cloud_scatterer, pysdom_output_file_name, comment_line)


print('done')