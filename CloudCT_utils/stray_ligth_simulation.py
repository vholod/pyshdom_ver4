import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pyshdom
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import scipy.io as sio
import pickle
import os
import logging
from collections import OrderedDict
from tqdm import tqdm
from datetime import datetime, timedelta
import re
import cartopy.crs as ccrs # conda install -c conda-forge cartopy
import pymap3d as pm # conda install -c conda-forge pymap3d
from collections import OrderedDict

import skyfield
from skyfield.api import load, wgs84, EarthSatellite, Topos
from skyfield.framelib import itrs
import matplotlib.patches as mpatches
import pandas as pd


print('Skyfield version is {}'.format(skyfield.VERSION))
print('Skyfield version should be above 1.40')
norm = lambda x: x / np.linalg.norm(x, axis=0)

ts = load.timescale(builtin=True)
planets = load('de421.bsp')
earth   = planets['earth']
sun    = planets['sun']
EARTH_RADIUS = 6371 # km


def stringScientificNotationToFloat(sn):
    "Specific format is 5 digits, a + or -, and 1 digit, ex: 01234-5 which is 0.01234e-5"
    return 0.00001*float(sn[5]) * 10**int(sn[6:])

"""
See nice interactive website for sun simulation - http://andrewmarsh.com/apps/staging/sunpath3d.html

--------------------------------------------------------------------------
Skyfield is able to predict the positions of Earth satellites by loading satellite orbital
elements from Two-Line Element (TLE) files.

Here we simulate a satellites orbits and predict their propogation with Skyfield.
Skyfield runs the SGP4 satellite propagation routine.
The accuracy of the satellite positions is not perfect. It is only for our local simulations
for CloudCT project by the Technion. This accuracy is still good for imaging analizis.

Terms:

* Two-line element (TLE) - Two-line element data is a standard
data to describe satellite's orbital elements. TLE data sets from many satellites are 
updated daily and can be found on open source websites like CelesTrak.com.
if we ignore atmospheric drag and maneuvering, the propagation of the TLE data would be sufficient 
to the past and future.


* Epoch [Wiwipedia] -  In astronomy, an epoch or reference epoch is 
a moment in time used as a reference point for some time-varying 
astronomical quantity. It is useful for the celestial coordinates or 
orbital elements of a celestial body, as they are subject to perturbations 
and vary with time.

* GCRS [Wiwipedia] - geocentric celestial reference system. It is created 
by the IAU in 2000, it is coordinate system used to 
specify the location 
and motions of near-Earth objects, such as satellites.

* ICRS - International Celestial Reference System.
The ICRS is a higher-accuracy replacement for the old J2000 reference system.
The ICRF is a realization of ICRS. 
Skyfield always stores positions internally as Cartesian (x,y,z) 
vectors oriented along the axes of the International Celestial 
Reference System (ICRS). More information about the  ICRS (x,y,z) coordinates is in 
https://rhodesmill.org/skyfield/positions.html.



* ECI - The earth-centered inertial(ECI) frame is a global reference frame that 
has its origin at the center of the Earth. This reference frame does not 
rotate with Earth and serves as an inertial reference frame for satellites 
orbiting the Earth. Due to this, the ECI frame is used primarily in space 
applications. Skyfield’s default ICRS reference frame is inertial. 
So when generating positions centered on the Earth, Skyfield produces ECI 
coordinates by default. So I infere that Geocentric Celestial Reference System (GCRS)
is an ECI.

* There are two of the main reference systems used in satellite navigation:
The Conventional Celestial Reference System (also named Conventional Inertial
System, CIS) and the Conventional Terrestrial Reference System (also named 
Coordinated Terrestrial System, CTS).
(inertial frame of reference is a frame of reference that is not undergoing 
acceleration)
Conventional Terrestrial Reference System (TRS) This is a reference system
co-rotating with the earth in its diurnal rotation, also called Earth-Centred,
Earth-Fixed (ECEF).   The TRS has its origin in the earth's center of mass. 
Z-axis is identical to the direction of the earth's rotation axis defined by
the Conventional Terrestrial Pole (CTP), X-axis is defined as the intersection
of the orthogonal plane to Z-axis (fundamental plane) and Greenwich mean 
meridian, and Y-axis is orthogonal to both of them, making the system 
directly oriented. An example of TRF is the International Terrestrial
Reference Frame (ITRF) introduced by the International Earth Rotation and 
Reference Systems Service (IERS), which is updated every year 
(ITRF98, ITRF99, etc.). Other terrestrial reference frames are the 
World Geodetic System 84 (WGS-84), which is applied for GPS.


* ECEF - The earth-centered, earth-fixed (ECEF) frame is a global reference 
frame. The origin (point 0, 0, 0) is defined as the center of mass of 
Earth (hence the term geocentric Cartesian coordinates). 
There are three orthogonal axes fixed to the Earth. The
Ez axis points through the North Pole, the Ex axis points through 
the intersection of the IERS Reference Meridian (IRM) and the equator,
and the Ey axis completes the right-handed system. This reference frame 
rotates with Earth at an angular velocity of approximately 15 °/hour 
(360 ° over 24 hour).


* MLTAN (reference https://issfd.org/ISSFD_2007/11-2.pdf) - Mean Local Time of the Ascending Node
The MLTAN of an orbit is defined as the
angle between the orbit’s ascending node and the mean Sun.
Sun-synchronous orbit, such as Aqua’s, is designed to maintain a constant MLTAN by
matching the J2 nodal rate of the satellite with the nodal rate of the mean Sun. The
MLTAN is often presented in units of time with 12:00 PM  or noon  describing a Sunsynchronous orbit that places the Sun directly at zenith when the spacecraft is at the
ascending node. Orbital perturbation caused by the Sun and the Moon will cause the
actual MLTAN of a spacecraft to deviate from a fixed value. 
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

Very good references:
https://www.youtube.com/watch?v=BvjlBpP4zU8
https://orbitalmechanics.info/ # use for explaining orbital parameters.
https://platform.leolabs.space/visualization
https://github.com/uhho/orbit_simulator # great example with python and skyfield + notebooks.
https://github.com/snkas/hypatia # nice but not so relevant python code.

https://coreykruger.wordpress.com/programming-and-scripting/3d-orbit-plotter-python/

Interactive visualization:
https://geodesy.geology.ohio-state.edu/course/ES4310/orbit/home/
https://github.com/orbitalindex/awesome-space

"""

def float_round(x):
    """Round a float or np.float32 to a 3 digits float"""
    if type(x) == np.float32:
        x = x.item()
    return round(x,3) 

def generate_rest_views(satellite, SatNumber, init_ref_time, \
                        SZA, SAA, separation_distance, verbos = True):
    """
    TODO
    """
    # --------------------------------------------------------------
    # ---------------Generating a satellite position----------------
    # --------------------------------------------------------------
    # The simplest form in which you can generate a satellite position is 
    # to call its at() method, which will return an |xyz| position relative to the Earth’s 
    # center in the Geocentric Celestial Reference System. 
    # (GCRS coordinates are based on even more precise
    # axes than those of the old J2000 system.)
    
    geocentric = satellite.at(init_ref_time) # Geocentric ICRS position at t,
    lookat_lat_lon = wgs84.subpoint_of(geocentric) # GeographicPosition WGS84   
    lat0 = lookat_lat_lon.latitude.degrees
    lon0 = lookat_lat_lon.longitude.degrees
    h0 = 0  
    
    # Another example is:
    # barycentric = earth.at(t), here This places its (x,y,z) vector in the Barycentric Celestial Reference System (BCRS)
    # Remainder - Geocentric Celestial Reference System (GCRS) coordinates,
    # means, the position is measured from the Earth’s center.
    # ==============================================================
    #GeographicPositionObject = wgs84.geographic_position_of(geocentric)
    #satellite_lat_ = GeographicPositionObject.latitude.degrees
    #satellite_lon_ = GeographicPositionObject.longitude.degrees
    #elevation      = GeographicPositionObject.elevation
    # Return the GeographicPosition of a geocentric position.
    # Given a geocentric position, returns a GeographicPosition providing 
    # its latitude, longitude, and elevation above or below the surface 
    # of the ellipsoid.
    sat_names = ['sat_'+str(int(i+1)) for i in range(SatNumber-1)]
    center_index = int(SatNumber*0.5)
    sat_names.insert(center_index-1, 'sat_0')
    list1 = sat_names[:center_index-1]      
    list1.reverse()
    list2 = sat_names[center_index:]
    total_sats = 1 # since we have the nadir already.
    ref_sat = 'sat_0'
    ref_time = init_ref_time
    sats_dict = OrderedDict()
    sats_dict[ref_sat] = geocentric
    for new_sat in list1:
        # Satellite velocity:
        sat_velocity = np.linalg.norm( sats_dict[ref_sat].velocity.km_per_s ) # Still in Geocentric ICRS.
        # time to pas 100km:
        time_gap_insec = separation_distance/sat_velocity # seconds to pass separation_distance.
        time_gap_insec = timedelta(seconds=time_gap_insec)
        t_sat_ref = ref_time
        t_sat_new = ts.utc( t_sat_ref.utc_datetime() - time_gap_insec )
        geocentric_sat_new = satellite.at(t_sat_new)
        sats_dict[new_sat] = geocentric_sat_new
        psat1 = sats_dict[ref_sat]
        psat2 = geocentric_sat_new    
        km = (psat2 - psat1).distance().km
        if verbos:
            print('In {} seconds, the satellite moved {} km'.format(time_gap_insec,km))          
        total_sats += 1
        ref_sat = new_sat
        ref_time = t_sat_new
        
        # JUSST FOR TEST:
        if verbos:
            print(new_sat)
            difference = satellite - lookat_lat_lon # relative position sat - topos
            topocentric = difference.at(t_sat_new)
            # topocentric now is of type - Geometric ICRS position, what is it?
            # Ask the topocentric position for its altitude and azimuth:
            alt, az, distance = topocentric.altaz() # The altaz doesnot work on geocentric (“Earth centered”) positions.
            print('Sat: Altitude: {}, Azimuth: {}, distance: {}'.format(90 - alt.degrees, az.degrees, distance.km))
        
    
    # Go in oposit direction:
    ref_sat = 'sat_0'
    ref_time = init_ref_time 
    for new_sat in list2:
        # Satellite velocity:
        sat_velocity = np.linalg.norm( sats_dict[ref_sat].velocity.km_per_s ) # Still in Geocentric ICRS.
        # time to pas 100km:
        time_gap_insec = separation_distance/sat_velocity # seconds to pass separation_distance.
        time_gap_insec = timedelta(seconds=time_gap_insec)
        t_sat_ref = ref_time
        t_sat_new = ts.utc( t_sat_ref.utc_datetime() + time_gap_insec )
        geocentric_sat_new = satellite.at(t_sat_new)
        sats_dict[new_sat] = geocentric_sat_new
        psat1 = sats_dict[ref_sat]
        psat2 = geocentric_sat_new    
        km = (psat2 - psat1).distance().km
        if verbos:
            print('In {} seconds, the satellite moved {} km'.format(time_gap_insec,km))          
        total_sats += 1
        ref_sat = new_sat
        ref_time = t_sat_new
        
        # JUSST FOR TEST:
        if verbos:        
            print(new_sat)
            difference = satellite - lookat_lat_lon # relative position sat - topos
            topocentric = difference.at(t_sat_new)
            # topocentric now is of type - Geometric ICRS position, what is it?
            # Ask the topocentric position for its altitude and azimuth:
            alt, az, distance = topocentric.altaz() # The altaz doesnot work on geocentric (“Earth centered”) positions.
            print('Sat: Altitude: {}, Azimuth: {}, distance: {}'.format(90 - alt.degrees, az.degrees, distance.        km))
            
    
    # Note that the order in sats_dict doesnot matter yet.
    df = pd.DataFrame([])
    data = []    
    COLUMNS = ['sat name','utc time','sun zenith [deg]','sun azimuth [deg]','sat zenith [deg]',\
               'sat azimuth [deg]', 'scattering angle [deg]', \
               'sat ENU coordinates [km]', 'lookat ENU coordinates [km]']
    
    for sat_name in sat_names:
        sat = sats_dict[sat_name]
        # Calculate satellite azimuth and zenith:
        """
        In SHDOM the camera position is given in the following convention:
        Location in global x coordinates [km] (North)
        Location in global y coordinates [km] (East)
        Location in global z coordinates [km] (Up)
        
        So, our axis are left hand axis as following:
        x - North - n of ENU
        y - East  - e of ENU 
        z - Up    - u of ENU 
        """
        subsat_on_earth = wgs84.subpoint_of(sat) # GeographicPosition WGS84   
        
        GeographicPositionObject = wgs84.geographic_position_of(sat)
        satellite_lat_ = GeographicPositionObject.latitude.degrees
        satellite_lon_ = GeographicPositionObject.longitude.degrees
        elevation      = 1e3*GeographicPositionObject.elevation.km # in meters
        
        e,n,u = pm.geodetic2enu(satellite_lat_, satellite_lon_, elevation, lat0, lon0, h0, ell=None, deg=True) 
        # transforms involving ENU East North Up in meters
        e *= 1e-3 # meter to km
        n *= 1e-3 # meter to km
        u *= 1e-3 # meter to km  
        
        x = n
        y = e
        z = u
        
        # You can specify a location on Earth by giving its latitude and 
        # longitude to a standard “geodetic system” that models the Earth’s 
        # shape. The most popular model is WGS84, used by most modern maps and 
        # also by the Global Positioning System (GPS). If you are given a
        # longitude and latitude without a datum specified, 
        # they are probably WGS84 coordinates.
        sat_direction_ENU = norm(np.array([x,y,z]))
        sat_zenith  = np.rad2deg(np.arccos(sat_direction_ENU[2]))
        sat_azimuth = np.rad2deg(np.arctan2(sat_direction_ENU[1],sat_direction_ENU[0])) # y-coordinates first
        assert np.allclose(np.cos(np.deg2rad(sat_zenith)), sat_direction_ENU[2]), "Error in direction calculations!"
        assert np.allclose(np.sin(np.deg2rad(sat_zenith))*np.cos(np.deg2rad(sat_azimuth)), sat_direction_ENU[0]), "Error in direction calculations!"
        assert np.allclose(np.sin(np.deg2rad(sat_zenith))*np.sin(np.deg2rad(sat_azimuth)), sat_direction_ENU[1]), "Error in direction calculations!"
        
        if verbos:
            print('{}: Altitude: {}, Azimuth: {}, distance: {}'.format(sat_name, sat_zenith, sat_azimuth, np.linalg.norm([x,y,z])))
        
        # scattering angle [deg]:
        # my calculations -> acos(sin(SAT_THETA)sin(SUN_THETA)cos(SAT_PHI−SUN_PHI)+cos(SAT_THETA)cos(SUN_THETA))
        SUN_THETA = np.asscalar(SZA)
        SUN_PHI = np.asscalar(SAA)
        a = np.sin(np.deg2rad(sat_zenith))*np.sin(np.deg2rad(SUN_THETA))*np.cos(np.deg2rad(sat_azimuth - SUN_PHI))
        b = np.cos(np.deg2rad(sat_zenith))*np.cos(np.deg2rad(SUN_THETA))       
        scattering_angle = np.rad2deg(np.arccos(a+b))
        
        # Sun direction:
        sun_x = np.sin(np.deg2rad(SUN_THETA))*np.cos(np.deg2rad(SUN_PHI))
        sun_y = np.sin(np.deg2rad(SUN_THETA))*np.sin(np.deg2rad(SUN_PHI))
        sun_z = np.cos(np.deg2rad(SUN_THETA))
        sun_direction = np.array([sun_x,sun_y,sun_z])
        
        assert np.allclose(np.rad2deg(np.arccos(np.dot(sun_direction, sat_direction_ENU))), scattering_angle), \
                "Error in direction calculations!"
        
        # Within this convention, np.rad2deg(np.arccos(np.dot(sun_direction, sat_direction_ENU))) 
        # I demand that the sun direction (used for the scattering_angle calculations) 
        # is considered from the sun to object to, So:
        scattering_angle = 180 - scattering_angle
        # The scattering angle is 0 for forward scattering.
        # The scattering angle is 180 for backward scattering. 
        
        ENU_coordinates = [x,y,z]
        lookat_ENU_coordinates = [0,0,0] # temporary, TODO add nosiy implementation
        data.append([\
            sat_name,\
            init_ref_time.utc_datetime(),\
            SUN_THETA,\
            SUN_PHI,\
            np.asscalar(sat_zenith),\
            np.asscalar(sat_azimuth),\
            scattering_angle,\
            ENU_coordinates,\
            lookat_ENU_coordinates])
            
            
    # fill the pandas table:
    df = pd.DataFrame(data, columns=COLUMNS)  
    df = df.set_index('sat name')
    # now, sta_names = df.index
    
    # sat_name, ref_time, SUN_THETA, SUN_PHI, sat_zenith, sat_azimuth, scattering_angle, ENU_coordinates, lookat_ENU_coordinates
    sat_positions =np.array([np.array(xi) for xi in df['sat ENU coordinates [km]'].values])
    sat_look_ats =np.array([np.array(xi) for xi in df['lookat ENU coordinates [km]'].values])
    sat_directions = sat_positions - sat_look_ats # from object to sat.
    gs = np.linalg.norm(sat_directions.copy(), axis =1)
    sat_directions = sat_directions/gs[:,np.newaxis] # from object to sat.
    scaled_direction = sat_directions*gs[:,np.newaxis]
    if verbos:
        print("scattering angles are: {}".format(180 - df['scattering angle [deg]'].values))
    
    #fig = mlab.figure(1, bgcolor=(0.48, 0.48, 0.48), fgcolor=(0, 0, 0),
                      #size=(400, 400))    
    #mlab.points3d(sat_positions[:,0],\
                  #sat_positions[:,1],\
                  #sat_positions[:,2],\
                  #scale_factor = 1,color=(0,0,0), opacity = 0.5)
    
    #mlab.quiver3d(sat_look_ats[:,0],\
                  #sat_look_ats[:,1],\
                  #sat_look_ats[:,2],\
                  #scaled_direction[:,0],\
                  #scaled_direction[:,1],\
                  #scaled_direction[:,2],\
                  #line_width=2.0,color = (1,1,1), opacity=1,mode='2ddash',scale_factor=1) 
        
    #mlab.quiver3d(0,0,0,\
                 #sun_direction[0],\
                 #sun_direction[1],\
                 #sun_direction[2],\
                 #line_width=2.0,color = (1., 0.82745098, 0.2627451 ), opacity=1,mode='2ddash',scale_factor=700) 
    #mlab.show()

    return df    
 

def set_string_of_pearls_orbit(SatNumber = 10, separation_distance = 100, VISUALIZE = True, verbos = False):
    """
    TODO:
    
    inputs:
    be_near_SZA - float in degrees.
    SatNumber - integer.
    separation_distance = float in km approximate separation distance between satellites.
    VISUALIZE - boolian flag, If tru open all visualizations for debug and reports.
    """
    # Define when the satellite should take pictures when the sun is in sun_zenith_treshold
    sun_zenith_treshold = 70
    cosain_treshold =  np.cos(np.deg2rad(sun_zenith_treshold))
    
    assert SatNumber>3, "minimum 3 satellites in this formation."
    # ------------------------------------------------------------------------
    # -----------------------START FUNCTION-----------------------------------
    # ------------------------------------------------------------------------
    
    
        
    # use spesific TLE file.       
    TLE = """ISS (ZARYA)
    1 25544U 98067A   21340.39932169  .00004577  00000+0  91351-4 0  9997
    2 25544  51.6435 211.2046 0003988 277.6071 254.8235 15.48946544315288"""
    
    
    """
    Here the object satellite refers to the ISS.
    We want to change the altitude from the ISS's (~450km) to ~500km.
    So we will modifay the mean motion (which is very strongly related to atlitude) as in 
    https://github.com/uhho/orbit_simulator/blob/master/4_custom_satellites.ipynb
    Full documentation of satellite model parameters can be found in 
    https://rhodesmill.org/skyfield/api-satellites.html#skyfield.sgp4lib.EarthSatellite
    
    According to https://en.wikipedia.org/wiki/Two-line_element_set, the mean motion in the 
    TLE file is at line 2 string indexes of 53-63
    
    """
    # change CloudCT satellites to altitude to 500 km (above Earth's surface)
    a = 500 # km
    
    G_CONST = 6.674e-11 # m^3 kg^-1 s^-2 
    EARTH_MASS = 5.97237e24 # kg
    EARTH_RADIUS_METERS = 6.371e6
    MU_EARTH = G_CONST * EARTH_MASS    # m^3 s^-2
    MU_EARTH_km3 = MU_EARTH*1e-9 # units of km^3 s^-2
    a = EARTH_RADIUS_METERS + (a * (10**3)) # meters
    orbital_period = 2 * np.pi * np.sqrt(np.power(a, 3) / MU_EARTH) # sec
    MeanMotion = (np.pi * 2) / orbital_period # rad / s
    new_MeanMotion = MeanMotion * 60 # Mean motion in radians per minute [rad / min]
    
    name, L1, L2 = TLE.splitlines()
    L1 = L1.strip()
    L2 = L2.strip() 
    
    # ----------------------------------------------------------------------
    line1 = L1
    line2 = L2
    
    satellite_number                                        = int(line1[2:7])
    classification                                          = line1[7:8]
    international_designator_year                           = int(line1[9:11])
    international_designator_launch_number                  = int(line1[11:14])
    international_designator_piece_of_launch                = line1[14:17]
    epoch_year                                              = int(line1[18:20])
    epoch                                                   = float(line1[20:32])
    first_time_derivative_of_the_mean_motion_divided_by_two = float(line1[33:43].strip())
    second_time_derivative_of_mean_motion_divided_by_six    = stringScientificNotationToFloat(line1[44:52])
    bstar_drag_term                                         = stringScientificNotationToFloat(line1[53:61])
    the_number_0                                            = float(line1[62:63])
    element_number                                          = float(line1[64:68])
    checksum1                                               = float(line1[68:69])

    satellite        = int(line2[2:7])
    inclination      = float(line2[8:16]) # yes - means in the Classical Orbital Elements (COEs)
    right_ascension  = float(line2[17:25]) # yes, it is in degrees == np.rad2deg(satellite.model.nodeo)
    eccentricity     = float(line2[26:33]) * 0.0000001 # yes
    argument_perigee = float(line2[34:42]) # yes, degrees
    mean_anomaly     = float(line2[43:51]) # tes, deg
    """
    Mean Anomaly (M) The concept of the mean anomaly appears to be confusing is a 
    parameter that is used e.g., by the SPICE library (instead of the true anomaly). 
    The angular speed (degrees per second) of the true anomaly varies with time 
    (higher at the periapsis, lower at the apoapsis). The mean anomaly M describes a
    virtual position of the object drawn on an imaginary circle around the focal point 
    of the ellipse. The idea: In contrary to the true anomaly, the mean anomaly’s angular 
    speed is constant.
    """
    mean_motion      = float(line2[52:63]) # periods in a day
    revolution       = float(line2[63:68])
    checksum2        = float(line2[68:69])
    
    # Inferred :
    # Inferred period
    day_seconds = 24*60*60
    period = day_seconds * 1./mean_motion

    # Inferred semi-major axis (in km)
    semi_major_axis = ((period/(2*np.pi))**2 * MU_EARTH_km3)**(1./3) # indicates size of the orbit.
    # http://orbitsimulator.com/gravity/articles/smaCalculator.html
    
    #----------------------------------------------------------
    #----------------------------------------------------------
    #----------------------------------------------------------
        
    # setup one satellite. Consider this satellite is sat number int(SatNumber/2) and this satellite
    # is at nadir view.
    
    ref_satellite = EarthSatellite(L1, L2, name,ts)
    d1 = ts.utc(ref_satellite.epoch.utc_datetime())
    d0 = ts.utc(1949, 12, 31, 0,0)
    delta_days = d1 - d0
    
    # Dafine new satellite:
    from sgp4.api import Satrec, WGS72
    
    satrec = Satrec()
    satrec.sgp4init(
        WGS72,           # gravity model , SGP4 algorithm defaults to WGS72
        'i',             # 'a' = old AFSPC mode, 'i' = improved mode
        0,               # satnum: Satellite number, 0 means nadir
        delta_days,       # epoch: days since 1949 December 31 00:00 UT
        ref_satellite.model.bstar,      # bstar: drag coefficient (/earth radii)
        0, # ndot: ballistic coefficient (revs/day)
        0.0,             # nddot: second derivative of mean motion (revs/day^3)
        ref_satellite.model.ecco,       # ecco: eccentricity
        ref_satellite.model.argpo, # argpo: argument of perigee (radians)
        ref_satellite.model.inclo, # inclo: inclination (radians)
        ref_satellite.model.mo, # mo: mean anomaly (radians)
        new_MeanMotion, # no_kozai: mean motion (radians/minute)
        ref_satellite.model.nodeo, # nodeo: right ascension of ascending node (radians)
    )
    
    satellite = EarthSatellite.from_satrec(satrec, ts)
    #satellite = EarthSatellite(L1, L2, name,ts)
    if verbos:
        print('Satellite number:', satellite.model.satnum)
        print('new Epoch:', satellite.epoch.utc_jpl())    
        print('ref Epoch:', ref_satellite.epoch.utc_jpl())    
    
    
        print('Simulated Orbit: Right ascension of ascending node: {:}'.format(np.rad2deg(satellite.model.nodeo)))
        print('Simulated Orbit: Inclination: {:}'.format(np.rad2deg(satellite.model.inclo)))    
        print('Simulated Orbit: Orbital period: {:0.0f}h {:0.0f}m'.format(orbital_period // 3600, ((orbital_period % 3600) * 60) / 3600))
        print('Simulated Orbit: Revolutions per day: {:0.1f}'.format((3600 * 24) / orbital_period))
    
    
    
    # Find in which time from the TLE epoch, the satellite (at nadir) image a ground patch 
    # which illuminated by the Sun with SZA near to the (input) parameter be_near_SZA.
    
    # Define times range to seek this spesific occurrence.
    #minutes = np.arange(0, 2*214*(orbital_period/60), 1) # about two weeks?
    minutes = np.arange(0, 12*2*214*(orbital_period/60), 2) # about two weeks?
    
    #minutes = np.arange(0, 10*(orbital_period/60), 1) # about two weeks?
    simulation_days = (minutes.max())/(24*60)
    fdate = satellite.epoch.utc_strftime('%Y,%m,%d,%H,%M,%S')
    fdate = fdate.split(',')
    times = ts.utc(int(fdate[0]),\
                int(fdate[1]), int(fdate[2]), int(fdate[3]), int(fdate[4]) + minutes)
    
    
    # inregion_ means that it distinguishs between a just sample and a sample in the interisting regions
    inregion_st_lons   = []
    inregion_st_lats   = []
    inregion_SZA       = []
    inregion_SAA       = []
    ANG_TO_STRAYLIGHT  = []
    # add regions of interests as are described in the proposal:
    region_1 = [-37.274,-147.143,8.656,-72.403]
    region_2 = [-15.422,-82.456,4.773,-38.862]
    region_3 = [-29.964,-38.686,7.22,11.586]
    region_4 = [30.325,28.461,37.744,38,656]
    region_list = [region_1,region_2,region_3,region_4]
    region_name_list = ['region_1','region_2','region_3','region_4']
    
    colors = ['deepskyblue','green','magenta','blue']
    # --------------------------------------------
    # --------------------------------------------
    # --------------------------------------------
    save_data = []
    
    for t_indx,t in enumerate(tqdm(times)):
        geocentric = satellite.at(t) # Geocentric ICRS position at t,
        # Now Compute the subpoint directly below the satellite  
        # the point on the Earth with the same latitude and longitude
        # as the satellite,
        # but with a height above the WGS84 ellipsoid of zero:
        subsat_on_earth = wgs84.subpoint_of(geocentric) # GeographicPosition WGS84     
                
        position1 = subsat_on_earth.itrs_xyz.km
        
        """
        Alternativly the distance compare would be as following:
        e,n,u = pm.geodetic2enu(subsat_on_earth.latitude.degrees, subsat_on_earth.longitude.degrees, the sat hieght, lat0, lon0, h0, ell=None, deg=True) 
        # transforms involving ENU East North Up in meters
        e *= 1e-3 # meter to km
        n *= 1e-3 # meter to km
        u *= 1e-3 # meter to km
        st_dists[t_indx] = np.linalg.norm(np.array([e,n,u]))
        
        """
        
        # Find where the subsat_on_earth is close to the lookat_lat_lon:      
        st_lons  = subsat_on_earth.longitude.degrees
        st_lats  = subsat_on_earth.latitude.degrees
        
        # Take care of the Sun direction in the spesific position lookat_lat_lon:
        astro = (earth + subsat_on_earth).at(t).observe(sun)
        app = astro.apparent()   
        
        """
        Find when a satellite is in sunlight:
        Skyfield provides a simple geometric estimate for this through the is_sunlit() method. 
        Given an ephemeris with which it can compute the Sun’s position, it will return 
        True when the satellite is in sunlight and False otherwise.
        """
        sunlit = satellite.at(t).is_sunlit(planets)
        
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        
        alt, az, distance = app.altaz()
        #print("Sun altitude {}, azimuth {} at time {}".format(alt.degrees, az.degrees, t.utc_datetime()))
        sza_ = 90 - alt.degrees # sun zenith angle as in shdom
        SZA = sza_
        SAA = az.degrees
        
        # assert sunlit == (sza_ <= 90), "Strange result, the satellite should be in dark?"
        # means that SZA[t_indx] <= 90 it is in sunlight.

        # Altitude measures the angle above or below the horizon. 
        # The zenith is at +90[deg], an object on the horizon’s great circle is
        # at 0[deg], and the nadir beneath your feet is at −90[deg].
    
        # Azimuth measures the angle around the sky from the north pole: 
        # 0[deg] means exactly north, 90[deg] is east, 180[deg] is south, 
        # and 270[deg] is west.
        # =============================================================        
        
        # Check when the satellite is in the ranges:
        INSIDE_FLAG = False
        INSIDE_dict = dict(zip(region_name_list,len(region_list)*[False]))
        # chech if we inside the region of interest:
        for ingex,(region,region_name) in enumerate(zip(region_list,region_name_list)):
            if(st_lats  >= region[0] and st_lats <= region[2]):
                if(st_lons >= region[1] and st_lons <= region[3]):
                    # if we here, we inside the good region.
                    INSIDE_dict[region_name] = True
                    INSIDE_FLAG = True

        cosain =  np.cos(np.deg2rad(SZA))            
        if INSIDE_FLAG and (SZA <= 90) and (cosain >= cosain_treshold):
            inregion_st_lons.append(st_lons)
            inregion_st_lats.append(st_lats)
            inregion_SZA.append(SZA)
            inregion_SAA.append(SAA)   
            
            df = generate_rest_views(satellite, SatNumber, t, SZA, SAA, separation_distance, verbos = False)
            MAX_ANG_TO_STRAYLIGHT = max(180 - df['scattering angle [deg]'].values)
            # print("scattering angles are: {}".format(180 - df['scattering angle [deg]'].values))
            ANG_TO_STRAYLIGHT.append(MAX_ANG_TO_STRAYLIGHT)
            
            save_data.append([t.utc_datetime(),SZA,MAX_ANG_TO_STRAYLIGHT,list(INSIDE_dict.values())])

    # save data:
    # fill the pandas table:
    COLUMNS = ['utc time','sun zenith','stary light angle','regions']
    save_df = pd.DataFrame(save_data,columns=COLUMNS)    
    with open("stary_data_{}_days.pkl".format(float_round(simulation_days)) , 'wb') as f:
        pickle.dump(save_df, f)
        
        
    #if CLOSE_ENOUGH and VISUALIZE:
    if VISUALIZE:
        
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.stock_img() # Add a standard image to the map. Currently, the only (and default) option is a downsampled version of the Natural Earth shaded relief raster.

        for ingex,region in enumerate(region_list):
            
            ax.add_patch(mpatches.Rectangle(xy=[region[1], region[0]], width=region[3]-region[1], height=region[2]-region[0],
                                            fill=False,edgecolor=colors[ingex],linewidth=5,facecolor=None,transform=ccrs.PlateCarree())    )
            
            ax.add_patch(mpatches.Rectangle(xy=[region[1], region[0]], width=region[3]-region[1], height=region[2]-region[0],
                                            facecolor=colors[ingex],alpha=0.1,transform=ccrs.PlateCarree())    )    


        cm = plt.cm.get_cmap('RdYlBu')
        # sat
        ax.scatter(st_lons, st_lats,\
                c = 'w', alpha=0.1, # st_dists
                transform=ccrs.PlateCarree())     
        
        
    
    im = ax.scatter(inregion_st_lons, inregion_st_lats,\
                c = inregion_SZA, # 
                transform=ccrs.PlateCarree(),
                cmap=cm)     
    fig.colorbar(im, ax=ax)
        
    
    # Solar plot
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1, projection='polar')    
    # draw the analemma loops
    points = ax.scatter(inregion_SAA, inregion_SZA,
                        s=10, label=None, c=  inregion_SZA)
    ax.figure.colorbar(points)
    # change coordinates to be like a compass
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rmax(90)    
    
    
    # hist of angles:
    fig = plt.figure(figsize=(20, 10))    
    n, bins, patches = plt.hist(x=ANG_TO_STRAYLIGHT, bins=100, color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    
    
    #plt.xlim(min(bins), max(bins))
    plt.xlim(0, 180)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Angle',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.title('Stray light angle Histogram',fontsize=15)
    
    
    # -----------------------------------------------------:
    fig = plt.figure(figsize=(20, 10))    
    n, bins, patches = plt.hist(x=inregion_SZA, bins=100, color='r',
                                alpha=0.7, rwidth=0.85)
    
    
    #plt.xlim(min(bins), max(bins))
    plt.xlim(0, 180)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Angle',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.title('Sun zenith angle Histogram',fontsize=15)    
    plt.show()
    print('done')
    
        
            

    




if __name__ == '__main__':
    #example_ISS_orbit()
    
    csv_table_path = 'ISS_like_10sats_formation.csv'
    orbitd_data_frame = set_string_of_pearls_orbit(verbos=True)
    orbitd_data_frame.to_csv(csv_table_path, index=False)  # use index=False to avoide indexindg od raws.
    print('done')