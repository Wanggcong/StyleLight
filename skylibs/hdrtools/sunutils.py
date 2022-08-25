import numpy as np
import scipy, scipy.misc, scipy.ndimage, scipy.ndimage.filters
import scipy.spatial, scipy.interpolate, scipy.spatial.distance
from scipy.ndimage.interpolation import map_coordinates as map_coords

from pysolar import solar

import envmap


def findBrightestSpot(image, minpct=99.99):
    """
    Find the sun position (in pixels, in the current projection) using the image.
    """
    if isinstance(image, envmap.EnvironmentMap):
        image = image.data

    # Gaussian filter
    filteredimg = scipy.ndimage.filters.gaussian_filter(image, (5, 5, 0))

    # Intensity image
    if filteredimg.ndim == 2 or filteredimg.shape[2] > 1:
        intensityimg = np.dot( filteredimg[:,:,:3], [.299, .587, .114] )
    else:
        intensityimg = filteredimg
    intensityimg[~np.isfinite(intensityimg)] = 0

    # Look for the value a the *minpct* percentage and threshold at this value
    # We do not take into account the pixels with a value of 0
    minval = np.percentile(intensityimg[intensityimg > 0], minpct)
    thresholdmap = intensityimg >= minval

    # Label the regions in the thresholded image
    labelarray, n = scipy.ndimage.measurements.label(thresholdmap, np.ones((3, 3), dtype="bool8"))

    # Find the size of each of them
    funcsize = lambda x: x.size
    patchsizes = scipy.ndimage.measurements.labeled_comprehension(intensityimg,
                                                                  labelarray,
                                                                  index=np.arange(1, n+1),
                                                                  func=funcsize,
                                                                  out_dtype=np.uint32,
                                                                  default=0.0)

    # Find the biggest one (we must add 1 because the label 0 is the background)
    biggestPatchIdx = np.argmax(patchsizes) + 1

    # Obtain the center of mass of the said biggest patch (we suppose that it is the sun)
    centerpos = scipy.ndimage.measurements.center_of_mass(intensityimg, labelarray, biggestPatchIdx)

    return centerpos


def sunPosFromEnvmap(envmapInput):
    """
    Find the azimuth and elevation of the sun using the environnement map provided.
    Return a tuple containing (elevation, azimuth)
    """
    c = findBrightestSpot(envmapInput.data)
    u, v = c[1] / envmapInput.data.shape[1], c[0] / envmapInput.data.shape[0]

    x, y, z, _ = envmapInput.image2world(u, v)

    elev = np.arcsin(y)
    azim = np.arctan2(x, -z)

    return elev, azim


def sunPosFromCoord(latitude, longitude, time_, elevation=0):
    """
    Find azimuth annd elevation of the sun using the pysolar library.
    Takes latitude(deg), longitude(deg) and a datetime object.
    Return tuple conaining (elevation, azimuth)

    TODO verify if timezone influences the results.
    """
    # import datetime
    # time_ = datetime.datetime(2014, 10, 11, 9, 55, 28)
    azim = solar.get_azimuth(latitude, longitude, time_, elevation)
    alti = solar.get_altitude(latitude, longitude, time_, elevation)

    # Convert to radians
    azim = np.radians(-azim)
    elev = np.radians(90-alti)

    if azim > np.pi: azim = azim - 2*np.pi
    if elev > np.pi: elev = elev - 2*np.pi

    return elev, azim
