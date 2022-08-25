import os

import numpy as np
from scipy.special import sph_harm
from pyshtools.shtools import SHExpandDH, MakeGridDH

from envmap import EnvironmentMap

# Sanity check, sph_harm was bogus in some versions of scipy / Anaconda Python
# http://stackoverflow.com/questions/33149777/scipy-spherical-harmonics-imaginary-part
#assert np.isclose(sph_harm(2, 5, 2.1, 0.4), -0.17931012976432356-0.31877392205957022j), \
#    "Please update your SciPy version, the current version has a bug in its " \
#    "spherical harmonics basis computation."


class SphericalHarmonic:
    def __init__(self, input_, copy_=True, max_l=None, norm=4):
        """
        Projects `input_` to its spherical harmonics basis up to degree `max_l`.
        
        norm = 4 means orthonormal harmonics.
        For more details, please see https://shtools.oca.eu/shtools/pyshexpanddh.html
        """

        if copy_:
            self.spatial = input_.copy()
        else:
            self.spatial = input_

        if not isinstance(self.spatial, EnvironmentMap):
            self.spatial = EnvironmentMap(self.spatial, 'LatLong')

        if self.spatial.format_ != "latlong":
            self.spatial = self.spatial.convertTo("latlong")

        self.norm = norm

        self.coeffs = []
        for i in range(self.spatial.data.shape[2]):
            self.coeffs.append(SHExpandDH(self.spatial.data[:,:,i], norm=norm, sampling=2, lmax_calc=max_l))

    def reconstruct(self, height=None, max_l=None, clamp_negative=True):
        """
        :height: height of the reconstructed image
        :clamp_negative: Remove reconstructed values under 0
        """

        retval = []
        for i in range(len(self.coeffs)):
            retval.append(MakeGridDH(self.coeffs[i], norm=self.norm, sampling=2, lmax=height, lmax_calc=max_l))

        retval = np.asarray(retval).transpose((1,2,0))

        if clamp_negative:
            retval = np.maximum(retval, 0)

        return retval


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    e = EnvironmentMap('envmap.exr', 'angular')
    e.resize((64, 64))
    e.convertTo('latlong')

    se = SphericalHarmonic(e)

    err = []
    from tqdm import tqdm
    for i in tqdm(range(32)):
        recons = se.reconstruct(max_l=i)
        err.append(np.sum((recons - e.data)**2))

    plt.plot(err)

    plt.figure()
    plt.imshow(recons);
    plt.show()
