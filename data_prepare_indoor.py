import os
from PIL import Image
from envmap import EnvironmentMap
import numpy as np
# from hdrio import imsave
from skylibs.hdrio import imsave

#to_path_folder = '/mnt/disks/data/datasets/IndoorHDRDataset2018-256x512'
to_path_folder = '/mnt/disks/data/datasets/IndoorHDRDataset2018-512x1024'
if not os.path.exists(to_path_folder):
    os.mkdir(to_path_folder)

root_path = '/mnt/disks/data/datasets/IndoorHDRDataset2018'
train_list = sorted(os.listdir(root_path))


for file_ in train_list:
    file_path = os.path.join(root_path, file_)   #dataset_name
    e = EnvironmentMap(file_path, 'latlong')
    e.resize((512, 1024))
    imsave(os.path.join(to_path_folder,file_), e.data)
    # imsave("crop.jpg", e, quality=90)






