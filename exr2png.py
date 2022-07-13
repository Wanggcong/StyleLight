
# https://github.com/soravux/skylibs
from envmap import EnvironmentMap
from hdrio import imread, imwrite, imsave

import os

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=99, max_mapping=0.99):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img.astype('float32'), alpha


'''
image_name ="9C4A2347-df771e2f13.exr"

e = EnvironmentMap(image_name, 'latlong')

import numpy as np

ee = e.resize((128,256))
imsave('9C4A2347-df771e2f13_resized.exr',ee.data)

# hdr = TonemapHDR()
hdr = TonemapHDR(gamma=2.4, percentile=99, max_mapping=0.9)
hdr_rgb,_ = hdr(ee.data)

# power_numpy_img = np.power(ee.data, 1 / 2.4)
# imsave('9C4A2347-df771e2f13_gama_correct.exr',power_numpy_img)
# tonemapped_img = np.clip(power_numpy_img, 0, 1)
# imsave('9C4A2347-df771e2f13_gama_correct_clip.exr',tonemapped_img)

imsave('9C4A2347-df771e2f13_gama_correct_clip_.exr',hdr_rgb)
'''



root_path = "/mnt/disks/data/datasets/IndoorHDRDataset2018/"
to_path = "/mnt/disks/data/datasets/IndoorHDRDataset2018-128x256-debug"

if not os.path.exists(to_path):
    os.mkdir(to_path)


image_paths = sorted(os.listdir(root_path))

for one_path in image_paths[:10]:
    full_path = os.path.join(root_path, one_path)
    e = EnvironmentMap(full_path, 'latlong')

    ee = e.resize((128,256))
    imsave(os.path.join(to_path, one_path),ee.data)























