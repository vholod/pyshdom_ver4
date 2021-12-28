import pyshdom
import numpy as np
import pandas as pd
r_earth = 6371.0  # km

def StringOfPearls(SATS_NUMBER=10, orbit_altitude=500, widest_view=False, move_nadir_x=0, move_nadir_y=0):
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
        Darc = 100  # km # distance between adjecent satellites (on arc).

    Dtheta = Darc / R  # from the center of the earth.

    # where to set the satelites?
    theta_config = np.arange(-0.5 * SATS_NUMBER, 0.5 * SATS_NUMBER) * Dtheta  # double for wide angles

    theta_config = theta_config[::-1]  # put sat1 to be the rigthest
    print('Satellites angles relative to center of earth:')
    for i, a in enumerate(theta_config):
        print("{}: {}".format(i, a))

    theta_max, theta_min = max(theta_config), min(theta_config)

    X_config = r_orbit * np.sin(theta_config) + move_nadir_x
    Z_config = r_orbit * np.cos(theta_config) - r_earth
    Y_config = np.zeros_like(X_config) + move_nadir_y

    sat_positions = np.vstack([X_config, Y_config, Z_config])  # path.shape = (3,#sats) in km.

    Satellites_angles = np.rad2deg(np.arctan(X_config / Z_config))
    print('Satellites angles are:')
    print(Satellites_angles)
    print("max angle {}\nmin angle {}\n".format(theta_max, theta_min))

    # find near nadir view:
    # since in this setup y=0:
    near_nadir_view_index = np.argmin(np.abs(X_config))

    return sat_positions.T, near_nadir_view_index, theta_max, theta_min


def aim_all_satellites(cloud_xarr, Rsat):
    """
    Calculates the imager resolution and where all satellites lookat.
    Inputs:
       cloud_xarr - xarray contains all microphysical data
       Rsat - satellites orbit height.
    returns:
       cnx, cny - integers, the imager resolution in x and y directions.
       SAT_LOOKATS - np.array - satellites lookats vector.
       fov - field of view
    """
    dx, dy = cloud_xarr.delx.values, cloud_xarr.delx.values

    nx, ny, nz = cloud_xarr.dims['x'],cloud_xarr.dims['y'],cloud_xarr.dims['z']

    Lx = cloud_xarr.x.max() - cloud_xarr.x.min()
    Ly = cloud_xarr.y.max() - cloud_xarr.y.min()
    Lz = cloud_xarr.z.max() - cloud_xarr.z.min()
    L = max(Lx, Ly)

    Lz_droplets =  cloud_xarr.z.max() - cloud_xarr.z.min()
    dz = Lz_droplets / (nz - 1)

    # USED FOV, RESOLUTION and SAT_LOOKATS:
    # cny x cnx is the camera resolution in pixels
    pixel_footprint, camera_footprint = 0.02, 0  # Get pixel footprint and camera footprint at nadir view only
    CENTER_OF_MEDIUM_BOTTOM = [0.5 * nx * dx, 0.5 * ny * dy, 0]
    LOOKAT = CENTER_OF_MEDIUM_BOTTOM

    # Here we force tuning - extend FOV and riase the lookat to ensure the the clouds are seen well from all oblique
    # views.
    L *= 2  # tuning 1
    LOOKAT[2] = cloud_xarr.z.min() + 0.1  # tuning 2
    fov = 2 * np.rad2deg(np.arctan(0.5 * L / Rsat))
    pixel_footprint = round(pixel_footprint, 2) # Foe ex. to convert 0.02000555 to 0.02 and have stable resolution calculation
    cnx = cny = int(np.floor(L / pixel_footprint))

    # currently, all satellites lookat the same point so LOOKAT is common for all.

    return cnx, cny, LOOKAT, fov


def Load_costum_formation(cloud_xarr, sats_formation_path, wavelength=0.86):
    df = pd.read_csv(sats_formation_path)
    locations = df["sat ENU coordinates [km]"].apply(lambda x : np.array(eval(x)))
    Rstat = np.mean([loc[2] for loc in locations])
    cnx, cny, LOOKAT, fov = aim_all_satellites(cloud_xarr, Rstat)
    sensor_dict = pyshdom.containers.SensorsDict()
    for i, pos in enumerate(locations):
        sensor = pyshdom.sensor.perspective_projection(wavelength, fov, cnx, cny,
                                                       position_vector=pos, lookat_vector=LOOKAT,
                                                       up_vector=[0, 1, 0], stokes='I')
        sensor_dict.add_sensor(f'Camera {i}', sensor)

    return sensor_dict
