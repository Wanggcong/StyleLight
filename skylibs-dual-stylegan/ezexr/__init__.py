import warnings

import numpy as np


try:
    import OpenEXR
    import Imath

except Exception as e:
    pass


def imread(filename, bufferImage=None, rgb=True):
    """
    Read an .exr image and returns a numpy matrix or a dict of channels.

    Does not support .exr with varying channels sizes.

    :bufferImage: If not None, then it should be a numpy array
                  of a sufficient size to contain the data.
                  If it is None, a new array is created and returned.
    :rgb: If True: tries to get the RGB(A) channels as an image
          If False: Returns all channels in a dict()
          If "hybrid": "<identifier>.[R|G|B|A|X|Y|Z]" -> merged to an image
                       Useful for Blender Cycles' output.
    """

    if 'OpenEXR' not in globals():
        print(">>> Install OpenEXR-Python with `conda install -c conda-forge openexr openexr-python`\n\n")
        raise Exception("Please Install OpenEXR-Python")

    # Open the input file
    f = OpenEXR.InputFile(filename)

    # Get the header (we store it in a variable because this function read the file each time it is called)
    header = f.header()

    # Compute the size
    dw = header['dataWindow']
    h, w = dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1

    # Use the attribute "v" of PixelType objects because they have no __eq__
    pixformat_mapping = {Imath.PixelType(Imath.PixelType.FLOAT).v: np.float32,
                         Imath.PixelType(Imath.PixelType.HALF).v: np.float16,
                         Imath.PixelType(Imath.PixelType.UINT).v: np.uint32}

    # Get the number of channels
    nc = len(header['channels'])
    #print(nc)

    # Check the data type
    dtGlobal = list(header['channels'].values())[0].type

    if rgb is True:
        # Create the read buffer if needed
        data = bufferImage if bufferImage is not None else np.empty((h, w, nc), dtype=pixformat_mapping[dtGlobal.v])

        if nc == 1:  # Greyscale
            cname = list(header['channels'].keys())[0]
            data = np.fromstring(f.channel(cname), dtype=pixformat_mapping[dtGlobal.v]).reshape(h, w, 1)
        else:
            assert 'R' in header['channels'] and 'G' in header['channels'] and 'B' in header['channels'], "Not a grayscale image, but no RGB data!"
            channelsToUse = ('R', 'G', 'B', 'A') if 'A' in header['channels'] else ('R', 'G', 'B')
            nc = len(channelsToUse)
            for i, c in enumerate(channelsToUse):
                # Check the data type
                dt = header['channels'][c].type
                if dt.v != dtGlobal.v:
                    data[:, :, i] = np.fromstring(f.channel(c), dtype=pixformat_mapping[dt.v]).reshape((h, w)).astype(pixformat_mapping[dtGlobal.v])
                else:
                    data[:, :, i] = np.fromstring(f.channel(c), dtype=pixformat_mapping[dt.v]).reshape((h, w))
    else:
        data = {}

        for i, c in enumerate(header['channels']):
            dt = header['channels'][c].type
            data[c] = np.fromstring(f.channel(c), dtype=pixformat_mapping[dt.v]).reshape((h, w))

        if rgb == "hybrid":
            ordering = {key: i for i, key in enumerate("RGBAXYZ")}

            new_data = {}
            for c in data.keys():

                ident = c.split(".")[0]
                try:
                    chan = c.split(".")[1]
                except IndexError:
                    chan = "R"

                if ident not in new_data:
                    all_chans = [x.split(".")[1] for x in data if x.startswith(ident + ".")]
                    nc = len(all_chans)
                    new_data[ident] = np.empty((h, w, nc), dtype=np.float32)
                    for i, chan in enumerate(sorted(all_chans, key=lambda v: ordering.get(v, len(ordering)))):
                        new_data[ident][:,:,i] = data["{}.{}".format(ident, chan)].astype(new_data[ident].dtype)

            data = new_data

    f.close()
    
    return data


def imwrite(filename, arr, **params):
    """
    Write an .exr file from an input array.

    Optional params : 
    channel_names = name of the channels, defaults to "RGB" for 3-channel, "Y" for grayscale, and "Y{n}" for N channels.
    compression = 'NONE' | 'RLE' | 'ZIPS' | 'ZIP' | 'PIZ' | 'PXR24' (default PIZ)
    pixeltype = 'HALF' | 'FLOAT' | 'UINT' (default : dtype of the input array if float16, float32 or uint32, else float16)

    """

    if arr.ndim == 3:
        h, w, d = arr.shape
    elif arr.ndim == 2:
        h, w = arr.shape
        d = 1
    else:
        raise Exception("Could not understand dimensions in array.")
    
    if "channel_names" in params:
        ch_names = params["channel_names"]
        assert ch_names >= d, "Provide as many channel names as channels in the array."
    else:
        if d == 1:
            ch_names = ["Y"]
        elif d == 3:
            ch_names = ["R","G","B"]
        else:
            ch_names = ['Y{}'.format(idx) for idx in range(d)]

    if 'OpenEXR' not in globals():
        print(">>> Install OpenEXR-Python with `conda install -c conda-forge openexr openexr-python`\n\n")
        raise Exception("Please Install OpenEXR-Python")

    compression = 'PIZ' if not 'compression' in params or \
                     params['compression'] not in ('NONE', 'RLE', 'ZIPS', 'ZIP', 'PIZ', 'PXR24') else params['compression']
    imath_compression = {'NONE' : Imath.Compression(Imath.Compression.NO_COMPRESSION),
                            'RLE' : Imath.Compression(Imath.Compression.RLE_COMPRESSION),
                            'ZIPS' : Imath.Compression(Imath.Compression.ZIPS_COMPRESSION),
                            'ZIP' : Imath.Compression(Imath.Compression.ZIP_COMPRESSION),
                            'PIZ' : Imath.Compression(Imath.Compression.PIZ_COMPRESSION),
                            'PXR24' : Imath.Compression(Imath.Compression.PXR24_COMPRESSION)}[compression]


    if 'pixeltype' in params and params['pixeltype'] in ('HALF', 'FLOAT', 'UINT'):
        # User-defined pixel type
        pixformat = params['pixeltype']
    elif arr.dtype == np.float32:
        pixformat = 'FLOAT'
    elif arr.dtype == np.uint32:
        pixformat = 'UINT'
    elif arr.dtype == np.float16:
        pixformat = 'HALF'
    else:
        # Default : Auto detect
        arr_fin = arr[np.isfinite(arr)]
        the_max = np.abs(arr_fin).max()
        the_min = np.abs(arr_fin[arr_fin > 0]).min()

        if the_max <= 65504. and the_min >= 1e-7:
            print("Autodetected HALF (FLOAT16) format")
            pixformat = 'HALF'
        elif the_max < 3.402823e+38 and the_min >= 1.18e-38:
            print("Autodetected FLOAT32 format")
            pixformat = 'FLOAT'
        else:
            raise Exception('Could not convert array into exr without loss of information '
                            '(a value would be rounded to infinity or 0)')
        warnings.warn("imwrite received an array with dtype={}, which cannot be saved in EXR format."
                      "Will fallback to {}, which can represent all the values in the array.".format(arr.dtype, pixformat), RuntimeWarning)

    imath_pixformat = {'HALF' : Imath.PixelType(Imath.PixelType.HALF),
                        'FLOAT' : Imath.PixelType(Imath.PixelType.FLOAT),
                        'UINT' : Imath.PixelType(Imath.PixelType.UINT)}[pixformat]
    numpy_pixformat = {'HALF' : 'float16',
                        'FLOAT' : 'float32',
                        'UINT' : 'uint32'}[pixformat]      # Not sure for the last one...

    # Convert to strings
    if d == 1:
        data = [ arr.astype(numpy_pixformat).tostring() ]
    else:
        data = [ arr[:,:,c].astype(numpy_pixformat).tostring() for c in range(d) ]

    outHeader = OpenEXR.Header(w, h)
    outHeader['compression'] = imath_compression        # Apply compression
    outHeader['channels'] = {                           # Apply pixel format
        ch_names[i]: Imath.Channel(imath_pixformat, 1, 1) for i in range(d)
    }

    # Write the three color channels to the output file
    out = OpenEXR.OutputFile(filename, outHeader)
    if d == 1:
        out.writePixels({ch_names[0] : data[0] })
    elif d == 3:
        out.writePixels({ch_names[0] : data[0], ch_names[1] : data[1], ch_names[2] : data[2] })
    else:
        out.writePixels({ch_names[c] : data[c] for c in range(d)})

    out.close()


imsave = imwrite


__all__ = ['imread', 'imwrite', 'imsave']
