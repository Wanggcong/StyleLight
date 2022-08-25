## skylibs

Tools used for LDR/HDR environment map (IBL) handling, conversion and I/O.


### Install & Develop

Install using:
```
pip install skylibs
```

To develop skylibs, clone the repository and execute `python setup.py develop`


### OpenEXR & Spherical harmonics

To read and save `exr` files, install the following dependencies (works on win/mac/linux):

```
conda install -c conda-forge openexr-python openexr
```


### Spherical Harmonics

To use the spherical harmonics functionalities, install the following dependency (works on mac/linux):

```
conda install -c conda-forge pyshtools
```

### envmap

Example usage:
```
from envmap import EnvironmentMap

e = EnvironmentMap('envmap.exr', 'latlong')
e_angular = e.copy().convertTo('angular')
e_angular_sa = e_angular.solidAngles()
```

`envmap.EnvironmentMap` Environment map class. Converts easily between those formats:

- latlong (equirectangular)
- angular 
- sphere
- cube
- skyangular
- skylatlong

Available methods:

- `.copy()`: Deepcopy the instance.
- `.solidAngles()`: Computes the per-pixel solid angles of the current representation.
- `.convertTo(targetFormat)`: Convert to the `targetFormat`.
- `.rotate(format, rotation)`: Rotate the environment map using format DCM. Soon will support Euler Angles, Euler Vector and Quaternions.
- `.resize(targetSize)`: Resize the environment map. Be cautious, this function does not ensure energy is preserved!
- `.toIntensity()`: Convert to grayscale.
- `.getMeanLightVectors(normals)`: Compute the mean light vector of the environment map for the given normals.
- `.project(self, vfov, rotation_matrix, ar=4./3., resolution=(640, 480), projection="perspective", mode="normal")`: Extract a rectified image from the panorama. See [the code](https://github.com/soravux/skylibs/blob/master/envmap/environmentmap.py#L402) for details.

Internal functions:
- `.imageCoordinates()`: returns the (u, v) coordinates at teach pixel center.
- `.worldCoordinates()`: returns the (x, y, z) world coordinates for each pixel center.
- `.interpolate(u, v, valid, method='linear')`: interpolates


### Projection, cropping, simulating a camera

To perform a crop from a `pano.jpg`:

```
import numpy as np
from imageio import imread, imsave
from envmap import EnvironmentMap, rotation_matrix


e = EnvironmentMap('pano.jpg', 'latlong')

dcm = rotation_matrix(azimuth=np.pi/6,
                      elevation=np.pi/8,
                      roll=np.pi/12)
crop = e.project(vfov=85., # degrees
                 rotation_matrix=dcm,
                 ar=4./3.,
                 resolution=(640, 480),
                 projection="perspective",
                 mode="normal")

crop = np.clip(255.*crop, 0, 255).astype('uint8')
imsave("crop.jpg", crop, quality=90)
```

### hdrio

`imread` and `imwrite`/`imsave` supporting the folloring formats:

- exr (ezexr)
- cr2, nef, raw (dcraw)
- hdr, pic (custom, beta)
- tiff (tifffile or scipy)
- All the formats supported by `scipy.io`

### ezexr

Internal exr reader and writer.

### tools3d

- `getMaskDerivatives(mask)`: creates the dx+dy from a binary `mask`.
- `NfromZ`: derivates the normals from a depth map `surf`.
- `ZfromN`: Integrates a depth map from a normal map `normals`.
- `display.plotDepth`: Creates a 3-subplot figure that shows the depth map `Z` and two side views.
- `spharm.SphericalHarmonic` Spherical Harmonic Transform (uses `pyshtools`)


Example usage of `spharm`:
```
from envmap import EnvironmentMap
from tools3d import spharm

e = EnvironmentMap('envmap.exr', 'latlong')
sh = spharm.SphericalHarmonic(e)
print(sh.coeffs)
reconstruction = sh.reconstruct(height=64)
```

### hdrtools

Tonemapping using `pfstools`.


## Changelog

0.6.6: Fixed aspect ratio when embedding
0.6.5: Added envmap embed feature
0.6.4: Removed `pyshtools` as mandatory dependency
0.6.3: Removed custom OpenEXR bindings (can be easily installed using conda)
0.6.2: Removed `rotlib` dependency
0.6.1: Aspect ratio in `project()` now in pixels
0.6: Updated the transport matrix Blender plugin to 2.8+


## Roadmap

- Improved display for environment maps (change intensity with keystroke/button)
- Standalone `ezexr` on all platforms
- add `worldCoordinates()` output in spherical coordinates instead of (x, y, z)
- Add assert that data is float32 in convertTo/resize (internal bugs in scipy interpolation)
- bugfix: `.rotate()` not working on grayscale (2D) data (current fix: make the array 3D with 1 channel)
- bugfix: `.convertTo()` not working on grayscale (2D) data (current fix: make the array 3D with 1 channel)

