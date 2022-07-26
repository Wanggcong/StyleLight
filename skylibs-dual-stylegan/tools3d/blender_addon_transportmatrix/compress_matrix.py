import time
import numpy as np
from imageio import imsave
import lz4.frame
import msgpack
import msgpack_numpy
from ezexr import imread
from scipy.ndimage.interpolation import zoom


from functools import partial
lz4open = partial(lz4.frame.open, block_size=lz4.frame.BLOCKSIZE_MAX1MB,
                                  compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)


ts = time.time()
with open("output.npz", "rb") as fhdl:
    data = np.load(fhdl)
    T = data["arr_0"]
    normals = data["arr_1"]
    image_size = data["arr_2"]
print("Loading time: {:.03f}".format(time.time() - ts))


ts = time.time()
with lz4open("output.msgpack", "wb") as fhdl:
   msgpack.pack([T.astype('float16'), normals.astype('float16'), image_size], fhdl, default=msgpack_numpy.encode)
print("Writing time: {:.03f}".format(time.time() - ts))
