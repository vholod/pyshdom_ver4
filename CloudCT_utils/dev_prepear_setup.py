import pyshdom
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import scipy.io as sio
import os
import logging
from collections import OrderedDict
import xarray as xr



"""
The default sensor generator:

The basic sensor generator (pyshdom.sensor.make_sensor_dataset)takes 
np.array's of the angles and positions of pixels. It makes no 
assumptions about the FOV of the pixels. A possible default 
behaviour is to assume that each pixel is modeled by a single 
infinitesmal ray at the pixel's location and pointing direction. 
This default behaviour can be chosen by selecting 
fill_ray_variables=True. Note that ray variables must be defined in 
the sensor for it to be valid. So you must use this option or 
generate the rays from the pixels according to a model.
"""

mu = phi = x = y = z = np.array([1.0])
stokes = ['I']
wavelength = 0.86

sensor = pyshdom.sensor.make_sensor_dataset(x, y, z, \
                                            mu, phi, stokes, \
                                            wavelength, \
                                            fill_ray_variables=True)
pyshdom.checks.check_sensor(sensor)


"""
The Perspective Sensor:

The perspective sensor generator (pyshdom.sensor.perspective_projection) 
is a pinhole sensor and is defined by its location, pointing direction 
and resolution. This generator also includes explicit method for 
generating sub-pixel rays to model the FOV.

Note that the attributes contain the input information used to
generate this sensor.
"""

sensor_dict = pyshdom.containers.SensorsDict()

"""
SensorDict
Groups of sensors which share an uncertainty model are designated as instruments. 
And groups of instruments are stored in pyshdom.containers.SensorsDict, which is a glorified OrderedDict.
These groups of sensors are labeled. The pyshdom.containers.SensorsDict is the container which interacts 
with the other high level objects. Individual datasets that contain pixel geometries can still be used to 
directly interface with
a pyshdom.solver.RTE object to obtain observables at the specified sensor geometry.

Currently, I want to avoid using sensors names in nither sensor nor SensorDict.
I will follow the sensor_index es, measn, link a name to sensor_index.
"""


#---------------------------------------
sensor = pyshdom.sensor.perspective_projection(wavelength = 0.86, fov = 25.0, x_resolution = 100, y_resolution = 100,
                           position_vector = [0,0,10], lookat_vector = [0,0,1], up_vector = [0,1,0],
                           stokes=['I'], sub_pixel_ray_args={'method':pyshdom.sensor.stochastic,'nrays':1})


sensor_dict.add_sensor('TEST1', sensor)
#---------------------------------------
#sensor = pyshdom.sensor.perspective_projection(wavelength = 0.86, fov = 25.0, x_resolution = 100, y_resolution = 100,
                           #position_vector = [0,5,8], lookat_vector = [0,0,0], up_vector = [0,1,0],
                           #stokes=['I'], sub_pixel_ray_args={'method':pyshdom.sensor.stochastic,'nrays':1})


#sensor_dict.add_sensor('TEST1', sensor)

sensor_list = sensor_dict['TEST1']['sensor_list']


pyshdom.sensor.show_sensors(sensor_list, scale = 0.6, axisWidth = 3.0, axisLenght=1.0, Show_Rays =  True, FullCone = True)
mlab.orientation_axes()

mlab.show()
print('done')