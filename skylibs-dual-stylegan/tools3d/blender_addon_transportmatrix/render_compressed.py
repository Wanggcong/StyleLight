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


# Loading the transport matrix
ts = time.time()
with lz4open("output.msgpack", "rb") as fhdl:
   T, normals, img_size = msgpack.unpack(fhdl, object_hook=msgpack_numpy.decode, max_str_len=2**32-1)
print("time: {:.03f}".format(time.time() - ts))
T = T.astype('float32')
normals = normals.astype('float32')

# Save the normals
imsave('normals.png', (normals + 1)/2)
#from matplotlib import pyplot as plt
#plt.imshow((normals + 1)/2); plt.show()

# Load the envmap
ts = time.time()
envmap = imread("envmap.exr", rgb=True)
envmap = zoom(envmap, (128/envmap.shape[0], 256/envmap.shape[1], 1), order=1, prefilter=True)
envmap = envmap[:64,:,:].astype('float32')
print("time: {:.03f}".format(time.time() - ts))


for i in range(256):
    envmap = np.roll(envmap, shift=1, axis=1)
    # Perform the rendering
    ts = time.time()
    im = T.dot(envmap.reshape((-1, 3)))
    # Tonemap & reshape
    im = im.reshape((img_size[0], img_size[1], 3))**(1./2.2)
    print("Render performed in {:.3f}s".format(time.time() - ts))


    # Save the images
    im *= 1/200
    imsave('render_{:03d}.png'.format(i), np.clip(im, 0, 1))
    imsave('envmap_{:03d}.png'.format(i), np.clip(0.7*envmap**(1./2.2), 0, 1))
    print("Saved images render.png & render_envmap.png")

