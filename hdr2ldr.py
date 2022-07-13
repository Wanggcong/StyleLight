
import glob
from envmap import EnvironmentMap
import numpy as np
import PIL.Image as Image
import os

image_paths = '/mnt/disks/data/datasets/IndoorHDRDataset2018-128x256-data-splits2/train/9C4A97*'
to_paths = '/home/guangcongwang/projects/mini-stylegan2-hdr/experiments/examples_ldr'

file_paths = glob.glob(image_paths)

for image_path in file_paths:

    e = EnvironmentMap(image_path, 'latlong')
    image_hdr = e.data
    # print('image:', image)
    # lo, hi = image.min(),image.max()
    # print('############## dataset image lo, hi:',lo, hi) #

    # comment
    gamma=2.4
    image = np.clip(image_hdr,1e-10,1e8)

    is_single_crop = False
    if is_single_crop:
        image = np.power(image, 1 / gamma)*5.0#*255c  ######
        image = np.clip(image,0,1)
    else:
        level = [0.01,0.02,0.04]
        aa = np.clip(np.power(image/level[0], 1/gamma), 0, 1)
        bb = np.clip(np.power(image/level[1], 1/gamma), 0, 1)
        cc = np.clip(np.power(image/level[2], 1/gamma), 0, 1)
        image = (aa+bb+cc)/3.0

    # image=image*2-1
    # comment

    # image = np.concatenate((image, image_hdr), axis=2)

    # image = image.transpose(2, 0, 1)


    im_ = Image.fromarray((image*255).astype(np.uint8))

    image_path_split = image_path.split('/')[-1].split('.')[0]

    im_.save(os.path.join(to_paths,f'{image_path_split}_ldr.png'))