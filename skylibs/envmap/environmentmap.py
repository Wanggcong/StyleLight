import os
from copy import deepcopy

import numpy as np
from scipy.ndimage.interpolation import map_coordinates, zoom

from skylibs.hdrio import imread

from .tetrahedronSolidAngle import tetrahedronSolidAngle
from .projections import *
from .xmlhelper import EnvmapXMLParser
from .rotations import rotx, roty, rotz


SUPPORTED_FORMATS = [
    'angular',
    'skyangular',
    'latlong',
    'skylatlong',
    'sphere',
    'cube',
]

ROTATION_FORMATS = [
    'DCM',
    'EA###',    # TODO
    'EV',       # TODO
    'Q',        # TODO
]


#From Dan:
#  I've generated these using the monochromatic albedo values from here:
#  http://agsys.cra-cin.it/tools/solarradiation/help/Albedo.html
#  (they cite some books as references). Since these monochromatic,
#  I got unscaled r, g, b values from internet textures and scaled them
#  so that their mean matches the expected monochromatic albedo. Hopefully
#  this is a valid thing to do.
GROUND_ALBEDOS = {
    "GreenGrass": np.array([ 0.291801, 0.344855, 0.113344 ]).T,
    "FreshSnow": np.array([ 0.797356, 0.835876, 0.916767 ]).T,
    "Asphalt": np.array([ 0.148077, 0.150000, 0.151923 ]).T,
}


class EnvironmentMap:
    def __init__(self, im, format_=None, copy=True, color=True):
        """
        Creates an EnvironmentMap.

        :param im: Image path or data to be converted to an EnvironmentMap, or 
                   the height of an empty EnvironmentMap.
        :param format_: EnvironmentMap format. Can be `Angular`, ...
        :param copy: When a numpy array is given, should it be copied.
        :param color: When providing an integer, create an empty color or
                      grayscale EnvironmentMap.
        :type im: float, numpy array
        :type format_: string
        :type copy: bool
        """
        if not format_ and isinstance(im, str):
            filename = os.path.splitext(im)[0]
            metadata = EnvmapXMLParser("{}.meta.xml".format(filename))
            format_ = metadata.getFormat()

        assert format_ is not None, (
            "Please provide format (metadata file not found).")
        assert format_.lower() in SUPPORTED_FORMATS, (
            "Unknown format: {}".format(format_))

        self.format_ = format_.lower()

        if isinstance(im, str):
            # We received the filename
            self.data = imread(im)
        elif isinstance(im, int):
            # We received a single scalar
            if self.format_ == 'latlong':
                if color:
                    self.data = np.zeros((im, im*2, 3))
                else:
                    self.data = np.zeros((im, im*2))
            elif self.format_ == 'skylatlong':
                if color:
                    self.data = np.zeros((im, im*4, 3))
                else:
                    self.data = np.zeros((im, im*4))
            elif self.format_ == 'cube':
                if color:
                    self.data = np.zeros((im, round(3/4*im), 3))
                else:
                    self.data = np.zeros((im, round(3/4*im)))
            else:
                if color:
                    self.data = np.zeros((im, im, 3))
                else:
                    self.data = np.zeros((im, im))
        elif type(im).__module__ == np.__name__:
            # We received a numpy array
            self.data = np.asarray(im, dtype='double')

            if copy:
                self.data = self.data.copy()
        else:
            raise Exception('Could not understand input. Please provide a '
                            'filename, a single size (height) or an image.')

        self.backgroundColor = np.array([0, 0, 0]) if self.data.ndim == 2 else np.zeros(self.data.shape[-1])

        # Ensure size is valid
        if self.format_ in ['sphere', 'angular', 'skysphere', 'skyangular']:
            assert self.data.shape[0] == self.data.shape[1], (
                "Sphere/Angular formats must have the same width/height")
        elif self.format_ == 'latlong':
            assert 2*self.data.shape[0] == self.data.shape[1], (
                "LatLong format width should be twice the height")
        elif self.format_ == 'skylatlong':
            assert 4*self.data.shape[0] == self.data.shape[1], (
                "SkyLatLong format width should be four times the height")

    @classmethod
    def fromSkybox(cls, top, bottom, left, right, front, back):
        """Create an environment map from skybox (cube) captures. Six images
        must be provided, one for each side of the virtual cube. This function
        will return an EnvironmentMap object.

        All the images must be square (width==height), have the same size 
        and number of channels.
        """
        basedim = top.shape[0]
        channels = 1 if len(top.shape) == 2 else top.shape[2]

        cube = np.zeros((basedim*4, basedim*3, channels), dtype=top.dtype)
        cube[0:basedim, basedim:2*basedim] = top
        cube[basedim:2*basedim, basedim:2*basedim] = front
        cube[basedim:2*basedim, 2*basedim:3*basedim] = right
        cube[3*basedim:4*basedim, basedim:2*basedim] = np.fliplr(np.flipud(back))
        cube[1*basedim:2*basedim, 0:basedim] = left
        cube[2*basedim:3*basedim, basedim:2*basedim] = bottom

        # We ensure that there is no visible artifacts at the junction of the images
        # due to the interpolation process
        cube[0:basedim, basedim-1] = left[0,...] # top-left
        cube[0:basedim, 2*basedim] = right[0,...][::-1] # top-right
        cube[basedim-1, 2*basedim:3*basedim] = top[:,-1][::-1] #right-top
        cube[2*basedim, 2*basedim:3*basedim] = bottom[:,-1] #right-bottom
        cube[1*basedim-1, 0:basedim] = top[:,0] # left-top
        cube[2*basedim, 0:basedim] = bottom[:,0][::-1] # left-bottom
        cube[2*basedim:3*basedim, basedim-1] = left[-1,...][::-1] #bottom-left
        cube[2*basedim:3*basedim, 2*basedim] = right[-1,...] #bottom-right
        cube[3*basedim:, basedim-1] = left[:,0][::-1] # back-left
        cube[3*basedim:, 2*basedim] = right[:,-1][::-1] # back-right

        return cls(cube, format_="cube")

    def __hash__(self):
        """Provide a hash of the environment map type and size.
        Warning: doesn't take into account the data, just the type,
                 and the size (useful for soldAngles)."""
        return hash((self.data.shape, self.format_))

    def copy(self):
        """Returns a copy of the current environment map."""
        return deepcopy(self)

    def solidAngles(self):
        """Computes the solid angle subtended by each pixel."""
        # If already computed, take it
        if hasattr(self, '_solidAngles') and hash(self) == self._solidAngles_hash:
            return self._solidAngles

        # Compute coordinates of pixel borders
        cols = np.linspace(0, 1, self.data.shape[1] + 1)
        rows = np.linspace(0, 1, self.data.shape[0] + 1)

        u, v = np.meshgrid(cols, rows)
        dx, dy, dz, _ = self.image2world(u, v)

        # Split each pixel into two triangles and compute the solid angle
        # subtended by the two tetrahedron
        a = np.vstack((dx[:-1,:-1].ravel(), dy[:-1,:-1].ravel(), dz[:-1,:-1].ravel()))
        b = np.vstack((dx[:-1,1:].ravel(), dy[:-1,1:].ravel(), dz[:-1,1:].ravel()))
        c = np.vstack((dx[1:,:-1].ravel(), dy[1:,:-1].ravel(), dz[1:,:-1].ravel()))
        d = np.vstack((dx[1:,1:].ravel(), dy[1:,1:].ravel(), dz[1:,1:].ravel()))
        omega = tetrahedronSolidAngle(a, b, c)
        omega += tetrahedronSolidAngle(b, c, d)

        # Get pixel center coordinates
        _, _, _, valid = self.worldCoordinates()
        omega[~valid.ravel()] = np.nan

        self._solidAngles = omega.reshape(self.data.shape[0:2])
        self._solidAngles_hash = hash(self)
        return self._solidAngles

    def imageCoordinates(self):
        """Returns the (u, v) coordinates for each pixel center."""
        cols = np.linspace(0, 1, self.data.shape[1]*2 + 1)
        rows = np.linspace(0, 1, self.data.shape[0]*2 + 1)

        cols = cols[1::2]
        rows = rows[1::2]

        return [d.astype('float32') for d in np.meshgrid(cols, rows)]

    def worldCoordinates(self):
        """Returns the (x, y, z) world coordinates for each pixel center."""
        u, v = self.imageCoordinates()
        x, y, z, valid = self.image2world(u, v)
        return x, y, z, valid

    def image2world(self, u, v):
        """Returns the (x, y, z) coordinates in the [-1, 1] interval."""
        func = {
            'angular': angular2world,
            'skyangular': skyangular2world,
            'latlong': latlong2world,
            'skylatlong': skylatlong2world,
            'sphere': sphere2world,
            'cube': cube2world,
        }.get(self.format_)
        return func(u, v)

    def world2image(self, x, y, z):
        """Returns the (u, v) coordinates (in the [0, 1] interval)."""
        func = {
            'angular': world2angular,
            'skyangular': world2skyangular,
            'latlong': world2latlong,
            'skylatlong': world2skylatlong,
            'sphere': world2sphere,
            'cube': world2cube,
        }.get(self.format_)
        return func(x, y, z)

    def interpolate(self, u, v, valid=None, order=1, filter=True, return_coor=False):
        """"Interpolate to get the desired pixel values."""
        # Repeat the first and last rows/columns for interpolation purposes
        if len(self.data.shape) == 2:
            h, w = self.data.shape
            d = 0
            source = np.empty((h + 2, w + 2))
        else:
            h, w, d = self.data.shape
            source = np.empty((h + 2, w + 2, d))

        source[1:-1, 1:-1] = self.data
        source[0,1:-1] = self.data[0,:]; source[0,0] = self.data[0,0]; source[0,-1] = self.data[0,-1]
        source[-1,1:-1] = self.data[-1,:]; source[-1,0] = self.data[-1,0]; source[-1,-1] = self.data[-1,-1]
        source[1:-1,0] = self.data[:,0]
        source[1:-1,-1] = self.data[:,-1]

        # To avoid displacement due to the padding
        u += 1./self.data.shape[1]
        v += 1./self.data.shape[0]
        # flatten, then float (0~1)-->int index
        target = np.vstack((v.flatten()*self.data.shape[0], u.flatten()*self.data.shape[1]))

        # print('target shape:',target)
        # print('target shape:',target.shape) # target shape: (2, 131072)

        if d == 0:
            data = np.zeros((u.shape[0], u.shape[1]))
            map_coordinates(source, target, output=data.ravel(), cval=np.nan, order=order, prefilter=filter)
        else:
            data = np.zeros((u.shape[0], u.shape[1], d))
            for c in range(d):
                map_coordinates(source[:,:,c], target, output=data[:,:,c].reshape(-1), cval=np.nan, order=order, prefilter=filter)
        self.data = data

        # In original: valid &= ~isnan(data)...
        # I haven't included it here because it may mask potential problems...
        if valid is not None:
            self.setBackgroundColor(self.backgroundColor, valid)
        if return_coor:
            return self, target
        return self

    def setBackgroundColor(self, color, valid):
        """Sets the area defined by ~valid to color."""
        assert valid.dtype == 'bool', "`valid` must be a boolean array."
        assert valid.shape[:2] == self.data.shape[:2], "`valid` must be the same size as the EnvironmentMap."

        self.backgroundColor = np.asarray(color)

        grayscale = len(self.data.shape) == 2

        if grayscale:
            self.data[:,:][np.invert(valid)] = self.backgroundColor.dot(np.array([0.299, 0.587, 0.114]).T)
        else:
            nb_channels = self.data.shape[2]
            for c in range(nb_channels):
                self.data[:,:,c][np.invert(valid)] = self.backgroundColor[c]

        return self

    def convertTo(self, targetFormat, targetDim=None):
        """
        Convert to another format.

        :param targetFormat: Target format.
        :param targetDim: Target dimension.
        :type targetFormat: string
        :type targetFormat: integer

        """
        assert targetFormat.lower() in SUPPORTED_FORMATS, (
            "Unknown format: {}".format(targetFormat))

        if not targetDim:
            # By default, number of rows
            targetDim = self.data.shape[0]

        eTmp = EnvironmentMap(targetDim, targetFormat)
        dx, dy, dz, valid = eTmp.worldCoordinates()
        u, v = self.world2image(dx, dy, dz)
        self.format_ = targetFormat.lower()
        self.interpolate(u, v, valid)

        return self

    def rotate(self, format, input_):
        """
        Rotate the environment map.

        :param format: Rotation type
        :param input: Rotation information (currently only 3x3 numpy matrix)
        """
        assert format.upper() in ROTATION_FORMATS, "Unknown rotation type '{}'".format(format)
        dx, dy, dz, valid = self.worldCoordinates()

        ptR = np.dot(input_, np.vstack((dx.flatten(), dy.flatten(), dz.flatten())))
        dx, dy, dz = ptR[0].reshape(dx.shape), ptR[1].reshape(dy.shape), ptR[2].reshape(dz.shape)

        dx = np.clip(dx, -1, 1)
        dy = np.clip(dy, -1, 1)
        dz = np.clip(dz, -1, 1)

        u, v = self.world2image(dx, dy, dz)
        self.interpolate(u, v, valid)

        return self

    def resize(self, targetSize, order=1):
        """
        Resize the current environnement map to targetSize.
        targetSize can be a tuple or a single number, in which case the same factor is assumed
        for both u and v.
        If 0 < targetSize < 1, treat it as a ratio
        If targetSize > 1, treat it as new dimensions to use
        """
        if not isinstance(targetSize, tuple):
            if self.format_ == 'latlong':
                targetSize = (targetSize, 2*targetSize)
            elif self.format_ == 'skylatlong':
                targetSize = (targetSize, 4*targetSize)
            elif self.format_ == 'cube':
                targetSize = (targetSize, round(3/4*targetSize))
            else:
                targetSize = (targetSize, targetSize)

        _size = []
        for i in range(2):
            _size.append(targetSize[i] / self.data.shape[i] if targetSize[i] > 1. else targetSize[i])

        if len(self.data.shape) > 2:
            _size.append(1.0)   # To ensure we do not "scale" the color axis...

        self.data = zoom(self.data, _size, order=order)
        return self

    def toIntensity(self):
        """
        Returns intensity-version of the environment map.
        This function assumes the CCIR 601 standard to perform internsity conversion.
        """
        assert len(self.data.shape) == 3, "Data should have exactly 3 dimensions"

        if self.data.shape[2] != 3:
            print("Envmap doesn't have 3 channels. This function won't do anything.")
        else:
            self.data = 0.299 * self.data[...,0] + 0.587 * self.data[...,1] + 0.114 * self.data[...,2]
            self.data = self.data[:,:,np.newaxis]
        return self

    def setHemisphereAlbedo(self, normal, value):
        """Sets an whole hemisphere defined by `normal` to a given `value`.
        Useful to set the ground albedo."""
        raise NotImplementedError()

    def getMeanLightVectors(self, normals):
        """Compute the mean light vector of the environment map for the normals given.
        Normals should be 3xN.
        Output is 3xN.
        """
        normals = np.asarray(normals)
        solidAngles = self.solidAngles()
        solidAngles /= np.nansum(solidAngles) # Normalize to 1
        normals /= np.linalg.norm(normals, 1)

        x, y, z, _ = self.worldCoordinates()

        xyz = np.dstack((x, y, z))

        visibility = xyz.dot(normals) > 0

        intensity = deepcopy(self).toIntensity()
        meanlight = visibility * intensity.data * solidAngles[:,:,np.newaxis]
        meanlight = np.nansum(xyz[...,np.newaxis] * meanlight[...,np.newaxis].transpose((0,1,3,2)), (0, 1))

        return meanlight

    def embed(self, vfov, rotation_matrix, image):
        """Projects an image onto the environment map.

        :vfov: Vertical Field of View (degrees).
        :rotation_matrix: Camera rotation matrix.
        :image: The image to project."""

        targetDim = self.data.shape[0]
        targetFormat = self.format_

        eTmp = EnvironmentMap(targetDim, targetFormat)
        dx, dy, dz, valid = eTmp.worldCoordinates()
        ar = image.shape[1]/image.shape[0]

        vfov_half_rad = vfov/2.*np.pi/180.
        hfov_half_rad = np.arctan(np.tan(vfov_half_rad)*ar)

        fx = 0.5/np.tan(hfov_half_rad)
        fy = 0.5/np.tan(vfov_half_rad)
        u0 = 0.5
        v0 = 0.5

        # world2image for a camera
        K = np.array([[fx, 0, u0],
                      [0, fy, v0],
                      [0,  0,  1]])
        M = K.dot(rotation_matrix)

        xyz = np.dstack((-dx, dy, dz)).reshape((-1, 3)).T

        # mask behind the camera
        forward_vector = rotation_matrix.T.dot(np.array([0, 0, -1]).T)
        mask = forward_vector.dot(xyz) <= 0
        xyz[:,mask] = np.inf

        uv = M.dot(xyz)
        u = (uv[0,:]/uv[2,:]).reshape(self.data.shape[:2])
        v = (uv[1,:]/uv[2,:]).reshape(self.data.shape[:2])

        self.format_ = targetFormat.lower()
        self.data = image.copy()
        self.interpolate(u, v, valid)
        return self

    def project(self, vfov, rotation_matrix, ar=4./3., resolution=(640, 480),
                projection="perspective", mode="normal"):
        """Perform a projection onto a plane ("_simulates_" a camera).

        Note: this function does not modify the foreshortening present in the
        environment map.

        :vfov: Vertical Field of View (degrees).
        :rotation_matrix: Camera rotation matrix.
        :ar: Aspect ratio (width / height), defaults to 4/3.
        :resolution: Output size in (cols, rows), defaults to (640,480).
        :projection: perspective (default) or orthographic.
        :mode: "normal": perform crop (default)
               "mask": show pixel mask in the envmap,
               "normal+uv": returns (crop, u, v), where (u,v) are the coordinates
                            of the crop."""

        coords = self._cameraCoordinates(vfov, rotation_matrix, ar,
                                         resolution, projection, mode)

        target = self.copy()
        if target.format_ != "latlong":
            target = target.convertTo("LatLong")
        
        if mode == 'mask':
            # mask_ = coords
            target_masked = coords.reshape(coords.shape[0],coords.shape[1],1)*target.data
            return coords
            # return target_masked

        # Get image coordinates from world sphere coordinates
        u, v = target.world2image(coords[0,:], coords[1,:], coords[2,:])
        # print('resolution[::-1]:',resolution[::-1])
        # print('coords:',coords.shape)
        # print('u v shape:',u.shape, v.shape)
        # print('u v:',u, v) #####
        # import pdb
        # pdb.set_trace()         
        u, v = u.reshape(resolution[::-1]), v.reshape(resolution[::-1])
        
        # import pdb
        # pdb.set_trace()   
        crop = target.interpolate(u, v).data

        if mode == "normal+uv":
            return crop, u, v

        return crop

    def Fov2MaskedPano(self, cropped_img, vfov, rotation_matrix, ar=4./3., resolution=(640, 480),
                projection="perspective", mode="normal"):
        """Perform a projection onto a plane ("_simulates_" a camera).

        Note: this function does not modify the foreshortening present in the
        environment map.

        :vfov: Vertical Field of View (degrees).
        :rotation_matrix: Camera rotation matrix.
        :ar: Aspect ratio (width / height), defaults to 4/3.
        :resolution: Output size in (cols, rows), defaults to (640,480).
        :projection: perspective (default) or orthographic.
        :mode: "normal": perform crop (default)
               "mask": show pixel mask in the envmap,
               "normal+uv": returns (crop, u, v), where (u,v) are the coordinates
                            of the crop."""

        coords = self._cameraCoordinates(vfov, rotation_matrix, ar,
                                         resolution, projection, mode)

        target = self.copy()
        if target.format_ != "latlong":
            target = target.convertTo("LatLong")
        
        if mode == 'mask':
            # mask_ = coords
            target_masked = coords.reshape(coords.shape[0],coords.shape[1],1)*target.data
            return coords
            # return target_masked

        # Get image coordinates from world sphere coordinates
        u, v = target.world2image(coords[0,:], coords[1,:], coords[2,:])
        # print('resolution[::-1]:',resolution[::-1])
        # print('coords:',coords.shape)
        # print('u v shape:',u.shape, v.shape)
        # print('u v:',u, v)
        # import pdb
        # pdb.set_trace()         
        u, v = u.reshape(resolution[::-1]), v.reshape(resolution[::-1])
        
        # import pdb
        # pdb.set_trace()   

        ####################################### cropped_img
        _, coor = target.interpolate(u, v, return_coor=True)

        # if mode == "normal+uv":
        #     return crop, u, v
        # return crop

        # given coor, cropped_img, we compute masked_pano
        masked_pano = np.zeros_like(self.data)
        # masked_pano = np.ones_like(self.data)*255
        # masked_pano[coor] = masked_pano, where # coor shape: (2, 131072)
        coor_ = coor.reshape(-1, resolution[::-1][0], resolution[::-1][1]).astype(np.int32)
        # print('coor_ shape:',coor_)
        # print('self.data shape:',self.data.shape)
        # coor_ = np.int(coor.reshape(-1, coor.shape[0], coor.shape[1]))

        # print('coor_[:,i,j]:',coor_[:,0,0])
        for i in range(cropped_img.shape[0]):
            for j in range(cropped_img.shape[1]):
                masked_pano[coor_[0,i,j],coor_[1,i,j],:] = cropped_img[i,j,:]


        return masked_pano



    def _cameraCoordinates(self, vfov, rotation_matrix, ar=4./3., resolution=(640, 480),
            projection="perspective", mode="normal"):

        if mode not in ("normal", "mask", "normal+uv"):
            raise Exception("Unknown mode: {}.".format(mode))

        if projection == "orthographic":
            vfov = np.arctan(np.sin(vfov*np.pi/180.))*180/np.pi

        # aspect ratio in pixels
        hfov = 2 * np.arctan(np.tan(vfov*np.pi/180./2)*ar)*180/np.pi

        # Project angle on the sphere to the +Z plane (distance=1 from the camera)
        mu = np.tan(hfov/2.*np.pi/180.)
        mv = np.tan(vfov/2.*np.pi/180.)

        if mode == "mask":
            x, y, z, _ = self.worldCoordinates()
            xy = np.sqrt( (x**2 + y**2) / (-x**2 - y**2 + 1) )
            theta = np.arctan2(x, y)
            x = xy*np.sin(theta)
            y = xy*np.cos(theta)

            hmask = (x > -mu) & (x < mu)
            vmask = (y > -mv) & (y < mv)
            dmask = z < 0
            mask = hmask & vmask & dmask

            e = EnvironmentMap(mask[:,:,np.newaxis], 'LatLong')
            e.rotate("DCM", rotation_matrix)
            # print('e.data shape:',e.data.shape)
            mask = e.data[:,:,0]
            return mask

        # Uniform sampling on the plane
        dy = np.linspace(mv, -mv, resolution[1])
        dx = np.linspace(-mu, mu, resolution[0])
        x, y = np.meshgrid(dx, dy)

        # Compute unit length vector (project back to sphere) from the plane
        x, y = x.ravel(), y.ravel()
        if projection == "perspective":
            xy = np.sqrt( (x**2 + y**2) / (x**2 + y**2 + 1) )
            theta = np.arctan2(x, y)
            x = xy*np.sin(theta)
            y = xy*np.cos(theta)
        elif projection == "orthographic":
            pass
        else:
            raise Exception("Unknown projection: {}.".format(projection))
        z = -np.sqrt(1 - (x**2 + y**2))
        coords = np.vstack((x, y, z))

        # Perform rotation
        coords = rotation_matrix.T.dot(coords)

        return coords


def rotation_matrix(azimuth, elevation, roll=0):
    """Returns a camera rotation matrix.
    :azimuth: left (negative) to right (positive) [rad]
    :elevation: upward (negative) to downward (positive) [rad]
    :roll: counter-clockwise (negative) to clockwise (positive) [rad]"""
    return rotz(roll).dot(rotx(elevation)).dot(roty(-azimuth))

def downscaleEnvmap(nenvmap, sao, sat, times):
    """Energy-preserving environment map downscaling by factors of 2.
    Usage:
    sao = EnvironmentMap(512, 'LatLong').solidAngles() # Source envmap solid angles, could be replaced by `sao = envmap.soldAngles()`
    sat = EnvironmentMap(128, 'LatLong').solidAngles() # Target envmap solid angles
    downscaleEnvmap(envmap, sao, sat, 3)
    Note : `times` is the number of downscales, so the total downscaling factor is 2**times"""
    nenvmap.data *= sao[:, :, np.newaxis]
    nenvmap.data = np.pad(nenvmap.data, [(0, 1), (0, 1), (0, 0)], 'constant')
    sx = np.cumsum(nenvmap.data, axis=1)
    tmp = sx[:, 2**times::2**times, ...] - sx[:, :-2**times:2**times, ...]
    sy = np.cumsum(tmp, axis=0)
    nenvmap.data = sy[2**times::2**times, :, ...] - sy[:-2**times:2**times, :, ...]
    if nenvmap.data.shape[1] > 2*nenvmap.data.shape[0]:
        nenvmap.data[:, -2, :] += nenvmap[:, -1, :]
        nenvmap.data = nenvmap.data[:, :-1, :]
    nenvmap.data /= sat[:, :, np.newaxis]
    nenvmap.data = np.ascontiguousarray(nenvmap.data)
    return nenvmap
