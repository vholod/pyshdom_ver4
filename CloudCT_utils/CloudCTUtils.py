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


def save_to_csv(cloud_scatterer, file_name, comment_line=''):
    """
    
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
  
    
    xgrid = cloud_scatterer.x
    ygrid = cloud_scatterer.y
    zgrid = cloud_scatterer.z
    
    dx = cloud_scatterer.delx.item()
    dy = cloud_scatterer.dely.item()
    dz = round(np.diff(zgrid)[0],5) 
    
    with open(file_name, 'w') as f:
        f.write(comment_line + "\n")
        # nx,ny,nz # nx,ny,nz
        f.write('{}, {}, {} '.format(int(cloud_scatterer.sizes.get('x')),\
                                    int(cloud_scatterer.sizes.get('y')),\
                                    int(cloud_scatterer.sizes.get('z')),\
                                    ) + "# nx,ny,nz\n")
        # dx,dy # dx,dy [km, km]
        f.write('{:2.3f}, {:2.3f} '.format(dx, dy) + "# dx,dy [km, km]\n")    
    
    
    
        # z_levels[0]     z_levels[1] ...  z_levels[nz-1] 
        
        np.savetxt(f, \
                   X=np.array(zgrid).reshape(1,-1), \
                   fmt='%2.3f',delimiter=', ',newline='')
        f.write(" # altitude levels [km]\n") 
        f.write("x,y,z,lwc,reff,veff\n")
    
        REGULAR_LWC_DATA = np.nan_to_num(cloud_scatterer.lwc)
        REGULAR_REFF_DATA = np.nan_to_num(cloud_scatterer.reff)
        REGULAR_VEFF_DATA = np.nan_to_num(cloud_scatterer.veff)
        
        y, x, z = np.meshgrid(range(cloud_scatterer.sizes.get('y')), \
                              range(cloud_scatterer.sizes.get('x')), \
                              range(cloud_scatterer.sizes.get('z')))
        
        data = np.vstack((x.ravel(), y.ravel(), z.ravel(),\
                          REGULAR_LWC_DATA.ravel(), REGULAR_REFF_DATA.ravel(), REGULAR_VEFF_DATA.ravel())).T
        # Delete unnecessary rows e.g. zeros in lwc
        mask = REGULAR_LWC_DATA.ravel() > 0
        data = data[mask,...]
        np.savetxt(f, X=data, fmt='%d ,%d ,%d ,%.5f ,%.3f ,%.5f')
    
    
