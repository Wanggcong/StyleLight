# import torch
import numpy as np
# import OpenEXR
# import Imath
# import cv2
# from scipy import interpolate
# import vtk
# from vtk.util import numpy_support
# import imageio
# imageio.plugins.freeimage.download()
import os
import glob

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
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
            tonemapped_img_clip = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img_clip.astype('float32'), alpha,tonemapped_img



tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)


import PIL.Image
# from skylibs.hdrio import imread, imsave
from hdrio import imread, imsave

# exr_path = '/home/deep/projects/mini-stylegan2/laval/test/*exr'

'''
exr_path = '/mnt/disks/data/datasets/IndoorHDRDataset2018_crop_from_origin/*exr'
image_paths =  glob.glob(exr_path)
to_path = '/mnt/disks/data/datasets/IndoorHDRDataset2018_crop_from_origin_png'
'''
exr_path = '/mnt/disks/data/datasets/IndoorHDRDataset2018-256x512-data-splits2/test_select/*exr'
image_paths =  glob.glob(exr_path)
to_path = '/mnt/disks/data/datasets/IndoorHDRDataset2018-256x512-data-splits2/test_select_png'


if not os.path.exists(to_path):
    os.mkdir(to_path)
    
for img_path in image_paths:
    # read img
    img_ = imread(img_path)
    img, alpha, img_hdr = tonemap(img_)
    print('alpha:', alpha, img.max(), img_.max(),img_hdr.max()**(1/2.4)-1)
    # print('alpha:', alpha, img.max(), img_.max(),img_hdr.max())

    file_name = os.path.join(to_path,img_path.split('/')[-1].split('.')[0]+'.png')
    print('file_name:',file_name)
    # imsave(file_name, np.clip(img_hdr**(1/2.4)-1,0,100))
    # imsave(file_name, np.clip(img_hdr-1,0,100))
    
    img = (img*255).astype(np.uint8) 
    # PIL.Image.save(img*255)
    PIL.Image.fromarray(img, 'RGB').save(file_name)





