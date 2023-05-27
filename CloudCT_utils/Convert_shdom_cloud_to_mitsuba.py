import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import scipy.io as sio
import os
import logging
from collections import OrderedDict
import at3d
from argparse import ArgumentParser
from CloudCTUtils import *

"""
Example:
Convert_shdom_cloud_to_mitsuba.py --CF "/wdata/vadim/CloudGeometry/clouds from Udi/16samples/BOMEX_512x512x200_20m_20m_1s_512_0000002300_3_3_new_pyshdom.txt"
--WB 0.65 0.65

"""
def main():
    parser = ArgumentParser(description='This script takes the Pyshdom format clouds txt file and converts it to Mitsuba format.\n The output was tested only ion Mitsuba version 3.\n The first run can take some time since the Mie tables are calculated.')
    
    # help:        
    parser.add_argument('--CF', type=str,
                        dest='CLOUD_FILE_NAME',
                        help='CLOUD FILE PATH',
                        metavar='CLOUD_FILE_NAME', required=True)
    
    parser.add_argument('--OP', type=str,
                        dest='output_path',
                        help='OUTPUT PATH, path to store the output Mitsuba file. If not provided, save the output in the input folder',
                        metavar='output_path', default=False)
    
    
    parser.add_argument('--WB',
                        dest='wavelength_band',
                        nargs=2,
                        default=[0.65, 0.65],
                        type=float,
                        help='wavelength_band: (float, float), (minimum, maximum) wavelength in microns. This defines the spectral band over which to integrate, if both are equal monochrome quantities are computed. If not equal, it does average scattering properties over the wavelength_band.')    
    
    
    opts = parser.parse_args()
    
    IFVISUALIZE = True
    
    wavelength_band = (opts.wavelength_band[0] , opts.wavelength_band[1])
    wavelen1, wavelen2 = wavelength_band
    wavelength_averaging = False
    formatstr = 'TEST_Water_{}nm.nc'.format(int(1e3*wavelength_band[0]))
    safe_mkdirs('../mie_tables')
    if not (wavelen1 == wavelen2):
        wavelength_averaging = True   
        formatstr = 'TEST_averaged_Water_{}-{}nm.nc'.format(int(1e3*wavelength_band[0]), int(1e3*wavelength_band[1]))
    mono_path = os.path.join('../mie_tables', formatstr)
    
    
    file_name = opts.CLOUD_FILE_NAME
    dirname, CLOUD_FILE_NAME = os.path.split(file_name)
    if not opts.output_path:
        output_path = dirname
    else:
        output_path = opts.output_path
        
    vol_filename = CLOUD_FILE_NAME.split('.')[0] + '.vol'
    vol_filename = os.path.join(output_path, vol_filename)
    objfilename  = CLOUD_FILE_NAME.split('.')[0] + '.obj'
    objfilename = os.path.join(output_path, objfilename)
    # save the scale for Mitsuba
    scale_mat_file = CLOUD_FILE_NAME.split('.')[0] + '_scale.mat'
    scale_mat_file = os.path.join(output_path, scale_mat_file)



    cloud_scatterer = at3d.util.load_from_csv(file_name, density='lwc')

    if IFVISUALIZE:
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

    if IFVISUALIZE:
        REGULAR_LWC_DATA = np.nan_to_num(cloud_scatterer.density)
        
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
        
        mlab.colorbar()
        mlab.show()

    # find reff/veff min/max:
    a=np.array(cloud_scatterer['reff'])
    b = a[a>0]
    reff_min = min(b.min() - 0.3, 1)
    reff_max = b.max()
    
    a=np.array(cloud_scatterer['veff'])
    b = a[a>0]
    veff_min = b.min()
    veff_max = b.max() + 0.01
    
    # -----------------------------------------------------
    # make a grid for microphysics which is just the cloud grid.
    rte_grid = at3d.grid.make_grid(dx,cloud_scatterer.x.data.size,
                              dy,cloud_scatterer.y.data.size,
                              cloud_scatterer.z)
    
    cloud_scatterer_on_rte_grid = at3d.grid.resample_onto_grid(rte_grid, cloud_scatterer)
    
    # We choose a gamma size distribution and therefore need to define a 'veff' variable.
    size_distribution_function = at3d.size_distribution.gamma
    
    # Exact OpticalPropertyGenerator:
    # get_mono_table will first search a directory to see if the requested table exists otherwise it will calculate it. 
    # You can save it to see if it works.
    mie_mono_table = at3d.mie.get_mono_table(
        'Water',wavelength_band,
        max_integration_radius=65.0,
        wavelength_averaging = wavelength_averaging, 
        minimum_effective_radius=0.1,
        relative_dir='../mie_tables',
        verbose=False
    )
    
    mie_mono_table.to_netcdf(mono_path)
    mie_mono_tables = OrderedDict()
    mie_mono_tables[wavelength_band[0]] = mie_mono_table
    
    optical_prop_gen = at3d.medium.OpticalPropertyGenerator(
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
    at3d.checks.check_optical_properties(optical_properties[wavelength_band[0]])
    
    # Extinction:
    extinction = np.array(optical_properties[wavelength_band[0]].extinction)
    
    if IFVISUALIZE:
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
    values = 1e-3*extinction
    scale = values.max()
    values = values/scale
    sio.savemat(scale_mat_file, dict(scale=scale))
    print('The scale is {}'.format(scale))
    
    N_x, N_y, N_z = values.shape       
    N_chan = 1
    
    xmin = 0
    xmax = 1e3*round(xgrid.max().item() + dx,5)
    
    ymin = 0
    ymax = 1e3*round(ygrid.max().item() + dy,5)
    
    zmin = zgrid.min().item()
    zmax = 1e3*round(zgrid.max().item() + dz,5)
    
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
    
    
if __name__ == '__main__':
    main()
