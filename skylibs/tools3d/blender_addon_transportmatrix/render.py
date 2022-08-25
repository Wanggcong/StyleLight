import time
import numpy as np
from imageio import imsave
from ezexr import imread
from scipy.ndimage.interpolation import zoom


# Load the transport matrix
ts = time.time()
with open("output.npz", "rb") as fhdl:
    data = np.load(fhdl)
    T = data["arr_0"]
    normals = data["arr_1"]
    img_size = data["arr_2"]
print("time: {:.03f}".format(time.time() - ts))

# Save the normals
imsave('normals.png', (normals + 1)/2)
#from matplotlib import pyplot as plt
#plt.imshow((normals + 1)/2); plt.show()

# Load the envmap
ts = time.time()
envmap = imread("envmap.exr", rgb=True)
envmap = zoom(envmap, (128/envmap.shape[0], 256/envmap.shape[1], 1), order=1, prefilter=True)
envmap = envmap[:64,:,:]
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

