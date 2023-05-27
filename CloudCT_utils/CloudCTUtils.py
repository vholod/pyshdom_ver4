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
import xarray as xr
import transformations as transf

# -------------------------------------------------------------------------------
# ----------------------CONSTANTS------------------------------------------
# -------------------------------------------------------------------------------
r_earth = 6371.0  # km
origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def float_round(x):
    """Round a float or np.float32 to a 3 digits float"""
    if type(x) == np.float32:
        x = x.item()
    return round(x,3) 


def safe_mkdirs(path):
    """Safely create path, warn in case of race."""

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            import errno
            if e.errno == errno.EEXIST:
                warnings.warn(
                    "Failed creating path: {path}, probably a race".format(path=path)
                )


def save_to_csv(cloud_scatterer, file_name, comment_line='', OLDPYSHDOM = False):
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
    OLDPYSHDOM: boll, if it is True, save the txt in old version of pyshdom.
    
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
    
    REGULAR_LWC_DATA = np.nan_to_num(cloud_scatterer.lwc)
    REGULAR_REFF_DATA = np.nan_to_num(cloud_scatterer.reff)
    REGULAR_VEFF_DATA = np.nan_to_num(cloud_scatterer.veff)
    
    y, x, z = np.meshgrid(range(cloud_scatterer.sizes.get('y')), \
                          range(cloud_scatterer.sizes.get('x')), \
                          range(cloud_scatterer.sizes.get('z')))

        
    if not OLDPYSHDOM:

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
        
            
            data = np.vstack((x.ravel(), y.ravel(), z.ravel(),\
                              REGULAR_LWC_DATA.ravel(), REGULAR_REFF_DATA.ravel(), REGULAR_VEFF_DATA.ravel())).T
            # Delete unnecessary rows e.g. zeros in lwc
            mask = REGULAR_LWC_DATA.ravel() > 0
            data = data[mask,...]
            np.savetxt(f, X=data, fmt='%d ,%d ,%d ,%.5f ,%.3f ,%.5f')
        
    else:
        # save in the old version:
        with open(file_name, 'w') as f:
            f.write(comment_line + "\n")
            # nx,ny,nz # nx,ny,nz
            f.write('{} {} {} '.format(int(cloud_scatterer.sizes.get('x')),\
                                        int(cloud_scatterer.sizes.get('y')),\
                                        int(cloud_scatterer.sizes.get('z')),\
                                        ) + "\n")
            
            # dx,dy ,z
            np.savetxt(f, X=np.concatenate((np.array([dx, dy]), zgrid)).reshape(1,-1), fmt='%2.3f')
            # z_levels[0]     z_levels[1] ...  z_levels[nz-1] 
            
            
            
            data = np.vstack((x.ravel(), y.ravel(), z.ravel(),\
                              REGULAR_LWC_DATA.ravel(), REGULAR_REFF_DATA.ravel(), REGULAR_VEFF_DATA.ravel())).T
            # Delete unnecessary rows e.g. zeros in lwc
            mask = REGULAR_LWC_DATA.ravel() > 0
            data = data[mask,...]
            np.savetxt(f, X=data, fmt='%d %d %d %.5f %.3f %.5f')

            
            
        
def show_scatterer(cloud_scatterer):
    
    """
    Show the scatterer in 3D with Mayavi.
    """
    
    ShowVolumeBox = True
    
    xgrid = cloud_scatterer.x
    ygrid = cloud_scatterer.y
    zgrid = cloud_scatterer.z
    
    dx = cloud_scatterer.delx.item()
    dy = cloud_scatterer.dely.item()
    dz = round(np.diff(zgrid)[0],5) 
    
    REGULAR_LWC_DATA = np.nan_to_num(cloud_scatterer.density)
    REGULAR_REFF_DATA = np.nan_to_num(cloud_scatterer.reff)
    REGULAR_VEFF_DATA = np.nan_to_num(cloud_scatterer.veff)
    
    
    show_field = REGULAR_LWC_DATA
    data_type = 'LWC [g/m^3]'
    
    mlab.figure(size=(600, 600))
    X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
    figh = mlab.gcf()
    src = mlab.pipeline.scalar_field(X, Y, Z, show_field, figure=figh)
    
    src.spacing = [dx, dy, dz]
    src.update_image_data = True
    
    isosurface = mlab.pipeline.iso_surface(src, contours=[0.1*show_field.max(),\
                                                          0.2*show_field.max(),\
                                                          0.3*show_field.max(),\
                                                          0.4*show_field.max(),\
                                                          0.5*show_field.max(),\
                                                          0.6*show_field.max(),\
                                                          0.7*show_field.max(),\
                                                          0.8*show_field.max(),\
                                                          0.9*show_field.max(),\
                                                          ],opacity=0.9,figure=figh)
    
    mlab.outline(figure=figh,color = (1, 1, 1))  # box around data axes
    mlab.orientation_axes(figure=figh)
    mlab.axes(figure=figh, xlabel="x (km)", ylabel="y (km)", zlabel="z (km)") 
    color_bar = mlab.colorbar(title=data_type, orientation='vertical', nb_labels=5)    
        
    if(ShowVolumeBox):
        # The _max is one d_ after the last point of the xgrid (|_|_|_|_|_|_|_->|).
        x_min = xgrid[0]
        x_max = round(xgrid[-1].item() + dx,5)
        
        y_min = ygrid[0]
        y_max = round(ygrid[-1].item() + dy,5)
        
        z_min = zgrid[0]
        z_max = round(zgrid[-1].item() + dz,5)    
        
        xm = [x_min, x_max, x_max, x_min, x_max, x_max, x_min, x_min ]
        ym = [y_min, y_min, y_min, y_min, y_max, y_max, y_max, y_max ]
        zm = [z_min, z_min, z_max, z_max, z_min, z_max, z_max, z_min ]
        # Medium cube
        triangles = [[0,1,2],[0,3,2],[1,2,5],[1,4,5],[2,5,6],[2,3,6],[4,7,6],[4,5,6],[0,3,6],[0,7,6],[0,1,4],[0,7,4]];
        obj = mlab.triangular_mesh( xm, ym, zm, triangles,color = (0.0, 0.17, 0.72),opacity=0.3,figure=figh)

             
    mlab.show()    
    
#---------------------------------------------------
def StringOfPearls(SATS_NUMBER=10, orbit_altitude=500, widest_view = False, move_nadir_x=0, move_nadir_y=0):

    """
    Set orbit parmeters:
         input:
         SATS_NUMBER - int, how many satellite to put?
         move_nadir_x/y - move in x/y to fit perfect nadir view.

         WIDEST_VIEW - bool, If WIDEST_VIEW False, the setup is the original with 100km distance between satellites.
         If it is True the distance become 200km.

         returns sat_positions: np.array of shape (SATS_NUMBER,3).
         The satellites setup alwas looks like \ \ | / /.
    """
    Rsat = orbit_altitude  # km orbit altitude
    R = r_earth + Rsat
    r_orbit = R

    if (widest_view):
        Darc = 200
    else:
        Darc = 100# km # distance between adjecent satellites (on arc).

    Dtheta = Darc / R  # from the center of the earth.

    # where to set the satelites?
    theta_config = np.arange(-0.5 * SATS_NUMBER, 0.5 * SATS_NUMBER) * Dtheta # double for wide angles

    theta_config = theta_config[::-1]  # put sat1 to be the rigthest
    #print('Satellites angles relative to center of earth:')
    #for i,a in enumerate(theta_config):
        #print("{}: {}".format(i,a))

    theta_max, theta_min = max(theta_config), min(theta_config)

    X_config = r_orbit * np.sin(theta_config) + move_nadir_x
    Z_config = r_orbit * np.cos(theta_config) - r_earth
    Y_config = np.zeros_like(X_config) + move_nadir_y

    sat_positions = np.vstack([X_config, Y_config, Z_config])  # path.shape = (3,#sats) in km.

    Satellites_angles = np.rad2deg(np.arctan(X_config/Z_config))
    print('Satellites angles are:')
    print(Satellites_angles)
    print("max angle {}\nmin angle {}\n".format(theta_max, theta_min))

    # find near nadir view:
    # since in this setup y=0:
    near_nadir_view_index = np.argmin(np.abs(X_config))


    return sat_positions.T, near_nadir_view_index, theta_max, theta_min


#---------------------------------------------------
#---------------------------------------------------
#---------------------------------------------------

def apply_platform_noise(in_sensor, sigma):
    """
    TODO: add noisy in the position in the future.
    
    Meanwhile it adds only orientation noise.
    
    Parameters:
    ------------
    in_sensor: xr.Dataset
        A dataset containing all of the information required to define a sensor
        for which synthetic measurements can be simulated;
        positions and angles of all pixels, sub-pixel rays and their associated weights,
        and the sensor's observables. It is the input sensor. The output is out_sensor.
        
    
    sigma: float
        Orientation noise amplitude (std) in degrees.
        The ralative angles roll , pitch, yaw will be sampled depending on sigma.
        
        
    Returns
    -------
    out_sensor : xr.Dataset
        An output dataset containing all of the information required to define a sensor
        for which synthetic measurements can be simulated;
        positions and angles of all pixels, sub-pixel rays and their associated weights,
        and the sensor's observables.    
    """
    assert 'Perspective' == in_sensor.attrs['projection'], "This method fits only Perspective projection."

    # Sample relative rotation angles:
    roll  =  np.deg2rad( np.random.normal(0, sigma, 1) ) 
    pitch =  np.deg2rad( np.random.normal(0, sigma, 1) ) 
    yaw =  np.deg2rad(0)    
    
    Rx = transf.rotation_matrix(roll, xaxis)
    Ry = transf.rotation_matrix(pitch, yaxis)
    Rz = transf.rotation_matrix(yaw, zaxis)
    R = transf.concatenate_matrices(Rx, Ry, Rz)# order: Rz then Ry then Rx, e.g. np.dot(Rz[0:3,0:3],np.dot(Rx[0:3,0:3],Ry[0:3,0:3]))
    # R is a relative rotation.
    
    image_shape = in_sensor['image_shape']
    
    """
    in_sensor is of class xarray.core.dataset.Dataset.
    Inside it, there are many xarray.DataArray s.
    Like:
    in_sensor.cam_x is an xarray.DataArray (in_sensor.cam_x.variable <xarray.Variable (npixels: 10000)> is array([1., ..., 1.]) )
    in_sensor.image_shape is an xarray.DataArray
    But, the in_sensor.cam_x  has Dimensions without coordinates: npixels. There are 10000 npixels, e.g. 10000 values.
    You can not index a pixel in cam_x rather than just use the index of a pixel.
    In the in_sensor.image_shape, the Coordinates  are regulat (image_dims) <U2 'nx' 'ny'.
    Good reference to read about terminology is here https://docs.xarray.dev/en/stable/user-guide/terminology.html
    
    
    in_sensor.coords
    Coordinates:
    * stokes_index  (stokes_index) <U1 'I' 'Q' 'U' 'V'
    * image_dims    (image_dims) <U2 'nx' 'ny'
  
    Variables:
    wavelength
    stokes
    cam_x
    cam_y
    cam_z
    cam_mu
    cam_phi
    image_shape
    ray_mu
    ray_phi
    ray_x
    ray_y
    ray_z
    pixel_index
    ray_weight

    """
    out_sensor = in_sensor.copy(deep=True)
    
    
    
    #load old parameteres:
    old_lookat = in_sensor.attrs['lookat']
    old_position = in_sensor.attrs['position']
    old_direction = old_lookat - old_position
    old_direction = norm(old_direction)
    
    old_rotation_matrix = in_sensor.attrs['rotation_matrix'].reshape(3,3)
    old_k = in_sensor.attrs['sensor_to_camera_transform_matrix'].reshape(3,3) # sensor_to_camera_transform_matrix
    
    old_cam_dir_x =  np.dot(old_rotation_matrix,xaxis)
    old_cam_dir_y =  np.dot(old_rotation_matrix,yaxis) 
    old_cam_dir_z =  np.dot(old_rotation_matrix,zaxis)   
    assert np.allclose(old_cam_dir_z,old_direction), "The vectors must be similar, chack the input."
    
    # TODO, change out_sensor.attrs['position'] 
    # TODO, change out_sensor.attrs['lookat'] 
    # TODO, change out_sensor.attrs['rotation_matrix']
    # TODO, change out_sensor.attrs['projection_matrix']     
    # TODO, change out_sensor.attrs['sensor_to_camera_transform_matrix'] 
    
    """
    ADD THE NOISE:
    
    """
    out_sensor.attrs['is_ideal_pointing'] = False
    new_cam_dir_z =  np.dot(R[0:3,0:3],old_cam_dir_z)   
    report_angle_deviation = np.rad2deg( np.arccos( np.dot(new_cam_dir_z,old_cam_dir_z) ) )
    print('The angle deviation form ideal pointing is {}[deg]'.format(report_angle_deviation))
    # ----------------------------
    #new_dir_z =  np.dot(R[0:3,0:3],zaxis)
    #A_t = rad2deg*np.arccos(np.dot(zaxis,new_dir_z))   
    #assert A_t <= A , "Problem in pointing noise generation."
    # ---------------------------------
    
        
    old_pointing_vector = self.get_pointing_vector()  # for test porpuses  
    Ronly = self._T[0:3,0:3] # rotation
    cam_dir_x =  np.dot(Ronly,xaxis)
    cam_dir_y =  np.dot(Ronly,yaxis) 
    cam_dir_z =  np.dot(Ronly,zaxis) 

    Rx = transf.rotation_matrix(roll, cam_dir_x)
    Ry = transf.rotation_matrix(pitch, cam_dir_y)
    Rz = transf.rotation_matrix(yaw, cam_dir_z)
    R_rel = transf.concatenate_matrices(Rx, Ry, Rz)# order: Ry then Rx
    
    """Vadim implemented random noise in x,y directions independently.
    Maybe the right nose model is different. Vadim should coordinate it with Alex.
    some references:
    1. https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
    """
    r = np.identity(4)
    r[0:3,0:3] = self._T[0:3,0:3]
    r_before_noise = r
    
    r = np.dot(R_rel,r) 
    
    # only for test:
    r_t = np.dot(R_rel.T,r) 
    test_dist = np.linalg.norm(r_before_noise  - r_t)
    assert test_dist<1e-6 , "Problem with the noise apllied to the rotation matrix"
    
    self._T[0:3,0:3] = r[0:3,0:3] # update the rotation with the noisy one, back to GT ratoadion do r = np.dot(self._rel_noisy.T,r)
    
    NewUp = np.dot(R_rel[0:3,0:3],self._up)# update the up with the noisy one, back to GT (?)
    # I had it befor but it is a bug, NewUp = np.dot(r[0:3,0:3],self._up)
    
    NewOpticalDirection = np.dot(r[0:3,0:3],np.array(zaxis)) # cameras z axis.
    # fined new lookAt point:
    pinhole_point = self._T[0:3,3]
    
    # intersection of a line with the ground surface (flat):
    """p_co, p_no: define the plane:
        p_co is a point on the plane (plane coordinate).
        p_no is a normal vector defining the plane direction.
        """
    p_co = np.array([0,0,0])
    p_no = np.array([0,0,1])
    epsilon = 1e-6
    u = NewOpticalDirection
    Q = np.dot(p_no, u)

    
    
    if abs(Q) > epsilon:
        d = np.dot((p_co - pinhole_point),p_no)/Q
        Newlookat = pinhole_point + (d*u)

    else:
        raise Exception("Can't find look at vector")
      
        
    self._rel_noisy = R_rel
    self._lookat = Newlookat
    self._up = NewUp     
    self._was_noise_applied = True
    
    # make another test:
    new_pointing_vector = NewOpticalDirection
    test_dist = np.linalg.norm(A_t  - rad2deg*np.arccos(np.dot(new_pointing_vector,old_pointing_vector)))
    assert test_dist<1e-6 , "Problem accured in the when noise was applyed to pointing."       
    #print("Pointing error: {}[deg] error was simulated.".format(A_t))
    ##R = np.dot(self._rel_noisy[0:3,0:3].T,self._T[0:3,0:3]) # rotation
    ##test_dist = np.linalg.norm(r_before_noise  - R)
    ##print(test_dist)

        
    print(direction)
    
    