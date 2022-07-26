import numpy as np
from imageio import imread, imsave
from ezexr import imsave as exrsave, imread as exrread
from matplotlib import pyplot as plt

from envmap import EnvironmentMap, rotation_matrix


e = EnvironmentMap('pano.jpg', 'latlong')

dcm = rotation_matrix(azimuth=85./180*np.pi,
                      elevation=58./180*np.pi,
                      roll=-25./180*np.pi)
crop = e.project(vfov=85., # degrees
                 rotation_matrix=dcm,
                 ar=4./3.,
                 resolution=(640, 480),
                 projection="perspective",
                 mode="normal")

crop = np.clip(255.*crop, 0, 255).astype('uint8')
imsave("crop.jpg", crop, quality=90)

e_embed = EnvironmentMap(e.data.shape[0], 'latlong')
e_embed = e_embed.embed(vfov=85.,
                        rotation_matrix=dcm,
                        image=crop.astype('float32')/255)


plt.imshow(e_embed.data); plt.show()
e_embed.data[~np.isfinite(e_embed.data)] = 0.
imsave("embed.jpg", e_embed.data, quality=90)


# Test ezexr

exrsave('embed.exr', e_embed.data)
data = exrread('embed.exr')
print(np.sum(e_embed.data - data))

exrsave('embed_gray.exr', e_embed.data[:,:,0])
data1 = exrread('embed_gray.exr')

exrsave('embed_2.exr', e_embed.data[:,:,:2])
data2 = exrread('embed_2.exr', rgb=False)
import pdb; pdb.set_trace()
