import os
import subprocess

import numpy as np
import imageio

__version__ = "0.6.2"


try:
    import ezexr
except ImportError as e:
    print("Could not import exr module:", e)

imsave_ldr = imageio.imwrite


def imwrite(data, filename):
    _, ext = os.path.splitext(filename.lower())
    if ext == '.exr':
        ezexr.imwrite(filename, data)
    elif ext in ['.hdr', '.pic']:
        _hdr_write(filename, data)
    else:
        imsave_ldr(filename, np.clip(255.*data, 0, 255).astype('uint8'))


def imsave(filename, data):
    imwrite(data, filename)


def imread(filename, format_="float32"):
    """Reads an image. Supports exr, hdr, cr2, tiff, jpg, png and
    everything SciPy/PIL supports.

    :filename: file path.
    :format_: format in which to return the value. If set to "native", the
              native format of the file will be given (e.g. uint8 for jpg).
    """
    ldr = False
    _, ext = os.path.splitext(filename.lower())

    if ext == '.exr':
        im = ezexr.imread(filename)
    elif ext in ['.hdr', '.pic']:
        im = _hdr_read(filename)
    elif ext in ['.cr2', '.nef', '.raw']:
        im = _raw_read(filename)
    elif ext in ['.tiff', '.tif']:
        try:
            import tifffile as tiff
        except ImportError:
            print('Install tifffile for better tiff support. Fallbacking to '
                  'imageio.')
            im = imageio.imread(filename)
        else:
            im = tiff.imread(filename)
    else:
        im = imageio.imread(filename)
        ldr = True

    if format_ == "native":
        return im
    elif ldr and not 'int' in format_:
        return im.astype(format_) / 255.
    else:
        return im.astype(format_)


def _raw_read(filename):
    """Calls the dcraw program to unmosaic the raw image."""
    fn, _ = os.path.splitext(filename.lower())
    target_file = "{}.tiff".format(fn)
    if not os.path.exists(target_file):
        ret = subprocess.call('dcraw -v -T -4 -t 0 -j {}'.format(filename))
        if ret != 0:
            raise Exception('Could not execute dcraw. Make sure the executable'
                            ' is available.')
    try:
        import tifffile as tiff
    except ImportError:
        raise Exception('Install tifffile to read the converted tiff file.')
    else:
        return tiff.imread(target_file)


def _hdr_write(filename, data, **kwargs):
    """Write a Radiance hdr file.
Refer to the ImageIO API ( http://imageio.readthedocs.io/en/latest/userapi.html
) for parameter description."""

    imageio.imwrite(filename, data, **kwargs)


def _hdr_read(filename, use_imageio=False):
    """Read hdr file.

.. TODO:

    * Support axis other than -Y +X
"""
    if use_imageio:
        return imageio.imread(filename, **kwargs)

    with open(filename, "rb") as f:
        MAGIC = f.readline().strip()
        assert MAGIC == b'#?RADIANCE', "Wrong header found in {}".format(filename)
        comments = b""
        while comments[:6] != b"FORMAT":
            comments = f.readline().strip()
            assert comments[:3] != b"-Y ", "Could not find data format"
        assert comments == b'FORMAT=32-bit_rle_rgbe', "Format not supported"
        while comments[:3] != b"-Y ":
            comments = f.readline().strip()
        _, height, _, width = comments.decode("ascii").split(" ")
        height, width = int(height), int(width)
        rgbe = np.fromfile(f, dtype=np.uint8).reshape((height, width, 4))
        rgb = np.empty((height, width, 3), dtype=np.float)
        rgb[...,0] = np.ldexp(rgbe[...,0], rgbe[...,3].astype('int') - 128)
        rgb[...,1] = np.ldexp(rgbe[...,1], rgbe[...,3].astype('int') - 128)
        rgb[...,2] = np.ldexp(rgbe[...,2], rgbe[...,3].astype('int') - 128)
        # TODO: This will rescale all the values to be in [0, 1]. Find a way to retrieve the original values.
        rgb /= rgb.max()
    return rgb


__all__ = ['imwrite', 'imsave', 'imread']
